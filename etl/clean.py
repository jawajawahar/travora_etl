"""
Stage 2 - Clean, categorise, deduplicate and impute.

Every imputed value is FLAGGED. Silent imputation is what lets a dataset look
complete while being fabricated; flagged imputation is an honest, reportable
statistic ("38% of opening hours imputed from category defaults").

Run:  python -m etl.clean
"""
from __future__ import annotations
import json
import logging
import re
import sys
from math import radians, sin, cos, asin, sqrt
from pathlib import Path
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz

from .config import (LK_BBOX, DATA_RAW, DATA_PROC, DEDUPE_NAME_THRESHOLD,
                     DEDUPE_DISTANCE_M, load_categories)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sri Lankan districts. Assignment is by nearest centroid, which is adequate for
# coverage reporting (criterion A2). Replace with an official boundary shapefile
# and a point-in-polygon test if higher precision is required; the method used is
# recorded in the output as `district_method`.
# ---------------------------------------------------------------------------
DISTRICTS = {
    "Colombo": (6.9271, 79.8612),      "Gampaha": (7.0873, 80.0144),
    "Kalutara": (6.5854, 79.9607),     "Kandy": (7.2906, 80.6337),
    "Matale": (7.7500, 80.6800),       "Nuwara Eliya": (6.9497, 80.7891),
    "Galle": (6.0535, 80.2210),        "Matara": (5.9549, 80.5550),
    "Hambantota": (6.1241, 81.1185),   "Jaffna": (9.6615, 80.0255),
    "Kilinochchi": (9.3803, 80.3770),  "Mannar": (8.9810, 79.9044),
    "Vavuniya": (8.7514, 80.4971),     "Mullaitivu": (9.2671, 80.8142),
    "Batticaloa": (7.7102, 81.6924),   "Ampara": (7.2917, 81.6747),
    "Trincomalee": (8.5874, 81.2152),  "Kurunegala": (7.4863, 80.3623),
    "Puttalam": (8.0362, 79.8283),     "Anuradhapura": (8.3114, 80.4037),
    "Polonnaruwa": (7.9403, 81.0188),  "Badulla": (6.9934, 81.0550),
    "Monaragala": (6.8728, 81.3510),   "Ratnapura": (6.6828, 80.3992),
    "Kegalle": (7.2513, 80.3464),
}


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    la1, lo1, la2, lo2 = map(radians, (lat1, lon1, lat2, lon2))
    h = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371000.0 * asin(sqrt(h))


def nearest_district(lat: float, lon: float) -> str:
    return min(DISTRICTS, key=lambda d: haversine_m(lat, lon, *DISTRICTS[d]))


# ---------------------------------------------------------------------------
# Opening hours
# ---------------------------------------------------------------------------
_TIME_RANGE = re.compile(r"(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})")


def parse_opening_hours(value: Optional[str]) -> Optional[tuple[int, int]]:
    """
    Pragmatic parser for the OSM `opening_hours` patterns that actually occur in
    Sri Lankan data. Returns (open_min, close_min) as minutes from midnight, or
    None if unparseable so the caller can impute and flag.

    Takes the widest window across all rules, which is the correct conservative
    choice: a scheduler that assumes a narrower window than reality only rejects
    feasible plans, whereas assuming a wider window produces plans that fail.
    """
    if not value or not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in {"24/7", "24 hours", "24hrs"}:
        return (0, 1440)

    matches = _TIME_RANGE.findall(v)
    if not matches:
        return None

    opens, closes = [], []
    for h1, m1, h2, m2 in matches:
        o = int(h1) * 60 + int(m1)
        c = int(h2) * 60 + int(m2)
        if c <= o:          # crosses midnight, e.g. 18:00-02:00
            c += 1440
        if 0 <= o <= 1440 and o < c <= 2880:
            opens.append(o)
            closes.append(min(c, 1440))
    if not opens:
        return None
    return (min(opens), max(closes))


def parse_fee(tags: dict) -> Optional[float]:
    """Extract an entry fee in LKR from OSM `charge`/`fee` tags."""
    charge = tags.get("charge") or tags.get("fee:amount")
    if charge:
        m = re.search(r"(\d[\d,\.]*)", str(charge))
        if m:
            try:
                amount = float(m.group(1).replace(",", ""))
                if "usd" in str(charge).lower():
                    amount *= 300          # rough LKR conversion; flagged downstream
                return amount
            except ValueError:
                pass
    if str(tags.get("fee", "")).lower() in {"no", "false"}:
        return 0.0
    return None


# ---------------------------------------------------------------------------
# Categorisation
# ---------------------------------------------------------------------------
def categorise(tags: dict, rules: list) -> Optional[str]:
    """
    First matching rule wins; order in categories.yaml is significant.

    A rule may also declare `require_any_key`: a list of tag keys of which at
    least one must be present for the match to count. This gates the noisiest
    OSM classes - amenity=place_of_worship and natural=wood - which are real
    features but not visitor destinations. Without the gate they made up 83% of
    the dataset and drowned out genuine attractions.
    """
    for rule in rules:
        matched = False
        for cond in rule.get("any", []):
            val = tags.get(cond["key"])
            if val is not None and str(val) in {str(v) for v in cond["values"]}:
                matched = True
                break
        if not matched:
            continue

        required = rule.get("require_any_key")
        if required and not any(tags.get(k) for k in required):
            continue                 # matched the class but lacks corroboration

        return rule["category"]
    return None


def pick_name(tags: dict) -> Optional[str]:
    for key in ("name:en", "name", "int_name", "official_name"):
        n = tags.get(key)
        if n and isinstance(n, str) and len(n.strip()) >= 3:
            return n.strip()
    return None


# ---------------------------------------------------------------------------
# Element normalisation
# ---------------------------------------------------------------------------
def normalise_elements(elements: list, cfg: dict) -> pd.DataFrame:
    rules = cfg["rules"]
    excludes = [re.compile(p, re.I) for p in cfg.get("exclude_name_patterns", [])]
    rows = []
    stats = {"no_name": 0, "no_coords": 0, "no_category": 0, "excluded": 0, "out_of_bbox": 0}

    for el in elements:
        tags = el.get("tags") or {}

        name = pick_name(tags)
        if not name:
            stats["no_name"] += 1
            continue
        if any(rx.search(name) for rx in excludes):
            stats["excluded"] += 1
            continue

        if el.get("type") == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            centre = el.get("center") or {}
            lat, lon = centre.get("lat"), centre.get("lon")
        if lat is None or lon is None:
            stats["no_coords"] += 1
            continue
        if not (LK_BBOX["min_lat"] <= lat <= LK_BBOX["max_lat"]
                and LK_BBOX["min_lon"] <= lon <= LK_BBOX["max_lon"]):
            stats["out_of_bbox"] += 1
            continue

        cat = categorise(tags, rules)
        if cat is None:
            stats["no_category"] += 1
            continue

        hours = parse_opening_hours(tags.get("opening_hours"))
        fee = parse_fee(tags)

        rows.append({
            "poi_id": f"{el['type']}/{el['id']}",
            "name": name,
            "lat": float(lat), "lon": float(lon),
            "category": cat,
            "open_min": hours[0] if hours else None,
            "close_min": hours[1] if hours else None,
            "hours_parsed": hours is not None,
            "entry_fee_lkr": fee,
            "fee_parsed": fee is not None,
            "wikidata": tags.get("wikidata"),
            "wikipedia": tags.get("wikipedia"),
            "unesco": bool(tags.get("heritage:operator") == "whc"
                           or "world heritage" in str(tags.get("heritage:website", "")).lower()),
            "tag_count": len(tags),
            "osm_tags_kept": json.dumps(
                {k: v for k, v in tags.items()
                 if k in ("tourism", "historic", "natural", "leisure", "amenity",
                          "boundary", "waterway", "man_made", "website", "phone")},
                ensure_ascii=False),
        })

    log.info("Normalised %d POIs. Dropped: %s", len(rows), stats)
    df = pd.DataFrame(rows)
    df.attrs["drop_stats"] = stats
    return df


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def normalise_name(name: str) -> str:
    """
    Reduce a name to its Latin-script tokens for fuzzy comparison.

    Sri Lankan OSM records frequently carry bilingual names such as
    "Fort Hammenhiel \u0b95\u0b9f\u0bb2\u0bcd \u0b95\u0bcb\u0b9f\u0bcd\u0b9f\u0bc8". Compared raw against
    "Fort Hammenhiel" this scores 71, below the 85 threshold, so the duplicate
    survived. Stripping non-Latin characters before comparison raises it to 100
    and the pair merges correctly.
    """
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(name).lower()).split())


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge records that are the same physical place appearing as both a node and a
    way, or duplicated across tag schemes. Two records merge when the names are
    similar AND they are physically close - either test alone produces false
    merges ("Buddha Statue" appears in many towns).
    """
    if df.empty:
        return df

    df = df.sort_values("poi_id").reset_index(drop=True)
    # Coarse spatial bucketing keeps this near-linear instead of O(n^2).
    df["_bucket"] = (df.lat.round(2).astype(str) + "_" + df.lon.round(2).astype(str))

    keep, dropped = [], 0
    for _, bucket in df.groupby("_bucket"):
        chosen: list[dict] = []
        for rec in bucket.to_dict("records"):
            merged = False
            for c in chosen:
                n_new = normalise_name(rec["name"])
                n_old = normalise_name(c["name"])
                similar = (fuzz.token_sort_ratio(n_new, n_old) >= DEDUPE_NAME_THRESHOLD
                           # one name fully containing the other is the bilingual case
                           or (n_new and n_old and (n_new in n_old or n_old in n_new)))
                if (similar
                        and haversine_m(rec["lat"], rec["lon"], c["lat"], c["lon"]) < DEDUPE_DISTANCE_M):
                    # Prefer the richer record: parsed hours, then parsed fee, then wikidata.
                    ascii_new = rec["name"].isascii()
                    ascii_old = c["name"].isascii()
                    score_new = (rec["hours_parsed"], rec["fee_parsed"],
                                 bool(rec["wikidata"]), ascii_new)
                    score_old = (c["hours_parsed"], c["fee_parsed"],
                                 bool(c["wikidata"]), ascii_old)
                    if score_new > score_old:
                        c.update(rec)
                    dropped += 1
                    merged = True
                    break
            if not merged:
                chosen.append(rec)
        keep.extend(chosen)

    out = pd.DataFrame(keep).drop(columns=["_bucket"], errors="ignore")
    log.info("Deduplication merged %d records -> %d remain", dropped, len(out))
    out.attrs["merged"] = dropped
    return out


# ---------------------------------------------------------------------------
# Imputation (always flagged)
# ---------------------------------------------------------------------------
def impute(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    defaults = cfg["defaults"]
    df = df.copy()

    df["open_min"] = df.apply(
        lambda r: r.open_min if r.hours_parsed else defaults[r.category]["open_min"], axis=1)
    df["close_min"] = df.apply(
        lambda r: r.close_min if r.hours_parsed else defaults[r.category]["close_min"], axis=1)
    df["hours_estimated"] = ~df["hours_parsed"]

    df["entry_fee_lkr"] = df.apply(
        lambda r: r.entry_fee_lkr if r.fee_parsed else defaults[r.category]["fee_lkr"], axis=1)
    df["fee_estimated"] = ~df["fee_parsed"]

    df["typical_dwell_min"] = df["category"].map(lambda c: defaults[c]["dwell_min"])
    df["dwell_estimated"] = True          # always a category default; stated honestly

    df["district"] = df.apply(lambda r: nearest_district(r.lat, r.lon), axis=1)
    df["district_method"] = "nearest_centroid"

    df["popularity_index"] = 0.0          # populated by etl.enrich
    df["eco_certified"] = "unknown"       # never False - absence of data is not absence of certification
    df["community_run"] = "unknown"

    return df.drop(columns=["hours_parsed", "fee_parsed"])


# ---------------------------------------------------------------------------
def run(raw_path: Optional[Path] = None) -> pd.DataFrame:
    cfg = load_categories()

    if raw_path is None:
        candidates = sorted(DATA_RAW.glob("osm_pois_*.json"))
        if not candidates:
            raise FileNotFoundError(
                "No raw OSM dump found. Run `python -m etl.extract_osm` first.")
        raw_path = candidates[-1]

    log.info("Cleaning %s", raw_path.name)
    payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))

    df = normalise_elements(payload.get("elements", []), cfg)
    drop_stats = df.attrs.get("drop_stats", {})
    df = deduplicate(df)
    merged = df.attrs.get("merged", 0)
    df = impute(df, cfg)

    df.attrs["drop_stats"] = drop_stats
    df.attrs["merged"] = merged

    out = DATA_PROC / "pois.parquet"
    df.to_parquet(out, index=False)
    log.info("Wrote %s with %d POIs", out.name, len(df))
    return df


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    df = run()
    print("\nCategory distribution:")
    print(df.category.value_counts().to_string())
    print(f"\nDistricts covered: {df.district.nunique()}/25")
    print(f"Opening hours parsed: {100 * (~df.hours_estimated).mean():.1f}%")
    print("\nNext: python -m etl.enrich")
    return 0


if __name__ == "__main__":
    sys.exit(main())
