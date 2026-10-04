"""
Stage 2c (preferred) - Assign districts by point-in-polygon against official
OpenStreetMap administrative boundaries.

Why this supersedes the kNN method
----------------------------------
Two earlier approaches both failed on real data:

  * Nearest-centroid used each district's principal TOWN as its reference point.
    Sri Lankan districts are not compact, so Sigiriya, Dambulla and Pidurangala
    were all assigned to Polonnaruwa instead of Matale.

  * Nearest-neighbour over SLTDA hotel coordinates fixed those, but inherits
    SLTDA's coverage: Mullaitivu has no registered accommodation with published
    coordinates, so its POIs were absorbed by Kilinochchi. Roughly 19% of POIs
    were reassigned with no way to verify the border cases.

Point-in-polygon against the actual administrative boundary is not an
approximation at all. A POI is in Matale district if and only if it lies inside
the Matale polygon.

The module degrades gracefully: if Overpass is unreachable or the boundaries
cannot be assembled, it reports the failure and leaves the existing assignment
untouched rather than writing something worse.

Run:  python -m etl.boundaries
"""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import DATA_RAW, DATA_PROC
from .districts import canonicalise_district, CANONICAL

log = logging.getLogger(__name__)

# Sri Lankan admin levels in OSM:
#   2 = country, 4 = province, 5 = DISTRICT, 6 = DS (divisional secretariat)
# An earlier version queried level 6 and received DS Divisions such as
# "Horana DS Division" and "Panadura DS Division", which are sub-district units.
# Levels 5 and 6 are both requested and non-district names are rejected in
# build_polygons(), so the module still works if a relation is tagged level 6.
OVERPASS_DISTRICTS = """
[out:json][timeout:300];
area["ISO3166-1"="LK"][admin_level=2]->.lk;
(
  relation["admin_level"="5"]["boundary"="administrative"](area.lk);
  relation["admin_level"="6"]["boundary"="administrative"]["name"~"District"](area.lk);
);
out geom;
"""

# Names that are administrative units other than districts.
REJECT_TOKENS = ("ds division", "divisional", "secretariat", "province",
                 "gn division", "grama", "municipal", "urban council",
                 "pradeshiya")


def is_district_name(raw: str) -> bool:
    """Reject sub-district and super-district units before canonicalisation."""
    low = " ".join(str(raw).strip().lower().split())
    if not low:
        return False
    return not any(tok in low for tok in REJECT_TOKENS)


def extract_boundaries() -> Path:
    from .extract_osm import fetch, save_raw
    log.info("Extracting district boundaries from Overpass")
    return save_raw(fetch(OVERPASS_DISTRICTS), "osm_districts")


def build_polygons(raw_path: Path) -> dict:
    """
    Assemble one polygon per district from the relation's member ways.

    OSM boundary relations store their outline as unordered way fragments, so
    the ways are polygonized rather than assumed to be in order. Anything that
    fails to close is skipped and reported.
    """
    from shapely.geometry import LineString, MultiPolygon
    from shapely.ops import polygonize, unary_union

    payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    polys: dict = {}
    skipped = []

    for el in payload.get("elements", []):
        if el.get("type") != "relation":
            continue
        tags = el.get("tags") or {}
        raw_name = (tags.get("name:en") or tags.get("name") or "").strip()
        if not is_district_name(raw_name):
            skipped.append(f"{raw_name} (not a district unit)")
            continue
        # strip a trailing "District" so "Matale District" canonicalises cleanly
        cleaned = raw_name
        for suffix in (" district", " disctrict"):
            if cleaned.lower().endswith(suffix):
                cleaned = cleaned[: -len(suffix)].strip()
        name = canonicalise_district(cleaned)
        if not name:
            skipped.append(raw_name or "(unnamed)")
            continue
        if name in polys:
            skipped.append(f"{raw_name} (duplicate of {name})")
            continue

        lines = []
        for m in el.get("members", []):
            if m.get("type") != "way" or m.get("role") not in ("outer", "", None):
                continue
            geom = m.get("geometry") or []
            pts = [(p["lon"], p["lat"]) for p in geom
                   if p.get("lon") is not None and p.get("lat") is not None]
            if len(pts) >= 2:
                lines.append(LineString(pts))

        if not lines:
            skipped.append(f"{raw_name} (no geometry)")
            continue

        built = list(polygonize(unary_union(lines)))
        if not built:
            skipped.append(f"{raw_name} (rings did not close)")
            continue

        geom = unary_union(built)
        # keep the largest part if the union is fragmented
        if isinstance(geom, MultiPolygon):
            geom = max(geom.geoms, key=lambda g: g.area) if len(geom.geoms) == 1 else geom
        polys[name] = geom

    log.info("Assembled %d district polygons", len(polys))
    if skipped:
        log.warning("Skipped %d relations: %s", len(skipped), skipped[:8])

    missing = set(CANONICAL) - set(polys)
    if missing:
        log.warning("No polygon for: %s", sorted(missing))
    return polys


def assign(df: pd.DataFrame, polys: dict) -> pd.DataFrame:
    """Point-in-polygon, with a nearest-polygon fallback for coastal points."""
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    names = list(polys.keys())
    geoms = [polys[n] for n in names]
    tree = STRtree(geoms)

    out_d, out_m = [], []
    inside = nearest = 0

    for lat, lon in zip(df.lat.to_numpy(float), df.lon.to_numpy(float)):
        p = Point(float(lon), float(lat))
        hit = None
        for idx in tree.query(p):
            if geoms[int(idx)].contains(p):
                hit = names[int(idx)]
                break
        if hit is not None:
            inside += 1
            out_d.append(hit)
            out_m.append("boundary_contains")
        else:
            # Coastal POIs and small islands can fall marginally outside.
            j = int(tree.nearest(p))
            nearest += 1
            out_d.append(names[j])
            out_m.append("boundary_nearest")

    df = df.copy()
    if "district" in df.columns:
        df["district_previous"] = df["district"]
    df["district"] = out_d
    df["district_method"] = out_m
    df["district_confidence"] = [1.0 if m == "boundary_contains" else 0.7
                                 for m in out_m]

    log.info("Point-in-polygon: %d inside a boundary, %d snapped to nearest "
             "(coastal or offshore)", inside, nearest)
    if "district_previous" in df.columns:
        changed = int((df.district != df.district_previous).sum())
        log.info("District changed for %d of %d POIs (%.1f%%)",
                 changed, len(df), 100 * changed / max(len(df), 1))
    return df


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    try:
        import shapely  # noqa: F401
    except ImportError:
        log.error("shapely is required: pip install shapely")
        log.error("Falling back: keep the existing assignment from etl.districts")
        return 1

    existing = sorted(DATA_RAW.glob("osm_districts_*.json"))
    try:
        raw = existing[-1] if existing else extract_boundaries()
        polys = build_polygons(raw)
    except Exception as e:                       # noqa: BLE001
        log.error("Boundary extraction failed: %s", e)
        log.error("Existing district assignment left untouched. "
                  "Run `python -m etl.districts` for the SLTDA-based fallback.")
        return 1

    if len(polys) < 20:
        log.error("Only %d polygons assembled, expected 25. Aborting rather "
                  "than writing a worse assignment.", len(polys))
        return 1

    for fname, label in (("pois.parquet", "POIs"),
                         ("accommodation.parquet", "Accommodation")):
        path = DATA_PROC / fname
        if not path.exists():
            log.warning("%s not found, skipping", fname)
            continue
        df = assign(pd.read_parquet(path), polys)
        df.to_parquet(path, index=False)
        log.info("Updated %s (%s)", fname, label)

        counts = df.district.value_counts()
        thin = [d for d in CANONICAL if counts.get(d, 0) < 5]
        log.info("%s: %d/25 districts with >=5 records%s", label,
                 25 - len(thin), f"; thin: {thin}" if thin else "")

    pois = pd.read_parquet(DATA_PROC / "pois.parquet")
    print("\nSpot check (all should be Matale):")
    for n in ("Sigiriya", "Dambulla", "Pidurangala"):
        for _, r in pois[pois.name.str.contains(n, case=False, na=False)].head(2).iterrows():
            print(f"  {r['name'][:38]:<40s} {r['district']:<14s} ({r['district_method']})")

    print("\nDistrict counts:")
    vc = pois.district.value_counts()
    for d in CANONICAL:
        c = int(vc.get(d, 0))
        print(f"  {d:<15s} {c:>5d}{'   <-- BELOW 5' if c < 5 else ''}")

    print("\nNext: python -m etl.enrich")
    return 0


if __name__ == "__main__":
    sys.exit(main())
