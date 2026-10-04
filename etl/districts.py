"""
Stage 2c - Assign districts using SLTDA's government-labelled coordinates.

Why this replaces nearest-centroid
----------------------------------
The first implementation assigned each POI to the district whose reference point
was closest. That reference point was the district's principal TOWN, not its
geographic centre, and Sri Lankan districts are not compact. Matale town sits in
the south of Matale district, while the district extends north past Dambulla and
Sigiriya. The result was that Sigiriya, Dambulla Cave Temple and Pidurangala -
the three best-known sites of the Cultural Triangle - were all assigned to
Polonnaruwa.

The SLTDA register solves this without any new data source. It contains 1,367
accommodation records with BOTH a published coordinate and a district assigned
by the national tourism authority. That is an authoritative labelled sample
covering all 25 districts, so district assignment becomes a nearest-neighbour
classification against government-labelled points rather than a guess against an
arbitrary reference.

Only records whose coordinates were published by SLTDA are used as training
points. Records whose coordinates this pipeline recovered from division or
district centroids are excluded, because their positions were derived from their
district labels and would make the classifier circular.

Run:  python -m etl.districts
"""
from __future__ import annotations
import logging
import sys
from collections import Counter

import numpy as np
import pandas as pd

from .config import DATA_PROC

log = logging.getLogger(__name__)

EARTH_KM = 6371.0
K_NEIGHBOURS = 7
MAX_TRUST_KM = 60.0     # beyond this, fall back to the centroid method

# Weight applied to synthetic centroid anchors relative to real SLTDA points.
# Low enough that a genuine government-labelled hotel always outvotes an anchor,
# high enough that a district with no SLTDA coverage can still be assigned.
ANCHOR_WEIGHT = 0.25

# Districts with fewer real reference points than this also receive an anchor.
MIN_REFERENCE_POINTS = 8


# ---------------------------------------------------------------------------
# District name canonicalisation
#
# SLTDA and OpenStreetMap disagree on the romanisation of several Sri Lankan
# district names. SLTDA writes "Moneragala" where this pipeline used
# "Monaragala", and the mismatch silently produced a district that held zero
# POIs while a near-identical name held 49. Every district name is therefore
# normalised through this map before use.
# ---------------------------------------------------------------------------
CANONICAL = [
    "Colombo", "Gampaha", "Kalutara", "Kandy", "Matale", "Nuwara Eliya",
    "Galle", "Matara", "Hambantota", "Jaffna", "Kilinochchi", "Mannar",
    "Vavuniya", "Mullaitivu", "Batticaloa", "Ampara", "Trincomalee",
    "Kurunegala", "Puttalam", "Anuradhapura", "Polonnaruwa", "Badulla",
    "Monaragala", "Ratnapura", "Kegalle",
]

ALIASES = {
    "moneragala": "Monaragala", "monaragala": "Monaragala",
    "mullaithivu": "Mullaitivu", "mullaittivu": "Mullaitivu",
    "mulaitivu": "Mullaitivu", "mullativu": "Mullaitivu",
    "killinochchi": "Kilinochchi", "kilinochci": "Kilinochchi",
    "kilinochchi": "Kilinochchi",
    "nuwaraeliya": "Nuwara Eliya", "nuwara eliya": "Nuwara Eliya",
    "amparai": "Ampara", "ampara": "Ampara",
    "kaluthara": "Kalutara", "puttlam": "Puttalam", "puttalama": "Puttalam",
    "kegalla": "Kegalle", "pollonnaruwa": "Polonnaruwa",
    "trincomali": "Trincomalee", "thrincomalee": "Trincomalee",
    "vavuniyawa": "Vavuniya", "mannaram": "Mannar",
    "maha nuwara": "Kandy", "anuradapura": "Anuradhapura",
}


def canonicalise_district(name) -> str | None:
    """Map any spelling variant onto the canonical district name."""
    if not isinstance(name, str) or not name.strip():
        return None
    key = " ".join(name.strip().lower().replace("-", " ").split())
    if key in ALIASES:
        return ALIASES[key]
    squashed = key.replace(" ", "")
    for c in CANONICAL:
        if squashed == c.lower().replace(" ", ""):
            return c
    for c in CANONICAL:                      # last resort: prefix match
        if squashed.startswith(c.lower().replace(" ", "")[:6]):
            return c
    return None


# Fallback anchors, one per district. Used only where SLTDA coverage is thin or
# absent, so that every district remains reachable by the classifier.
DISTRICT_ANCHORS = {
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


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorised: (n,) against (m,) -> (n, m)."""
    la1 = np.radians(np.asarray(lat1, dtype=float))[:, None]
    lo1 = np.radians(np.asarray(lon1, dtype=float))[:, None]
    la2 = np.radians(np.asarray(lat2, dtype=float))[None, :]
    lo2 = np.radians(np.asarray(lon2, dtype=float))[None, :]
    h = (np.sin((la2 - la1) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def load_reference() -> pd.DataFrame:
    """
    SLTDA records with published coordinates and a district label, plus a
    synthetic anchor for any district that SLTDA does not cover adequately.

    The anchors matter: Mullaitivu has no SLTDA accommodation with a published
    coordinate, so without an anchor the classifier could never vote for it and
    every POI in the district was absorbed by its neighbours.
    """
    path = DATA_PROC / "accommodation.parquet"
    if not path.exists():
        raise SystemExit(
            "accommodation.parquet not found. Run `python -m etl.extract_sltda` first - "
            "district assignment uses SLTDA's government-labelled coordinates.")

    acc = pd.read_parquet(path)
    ref = acc[(acc.get("location_method") == "sltda_published")
              & acc.district.notna()][["district", "lat", "lon"]].dropna().copy()

    ref["district"] = ref.district.map(canonicalise_district)
    unmapped = int(ref.district.isna().sum())
    if unmapped:
        log.warning("%d SLTDA records had an unrecognised district name and were "
                    "excluded from the reference set", unmapped)
    ref = ref.dropna(subset=["district"])
    ref["weight"] = 1.0

    per = ref.district.value_counts()
    log.info("SLTDA reference points: %d across %d districts",
             len(ref), ref.district.nunique())

    anchors = []
    for d, (lat, lon) in DISTRICT_ANCHORS.items():
        count = int(per.get(d, 0))
        if count < MIN_REFERENCE_POINTS:
            anchors.append({"district": d, "lat": lat, "lon": lon,
                            "weight": ANCHOR_WEIGHT})
            log.info("  anchor added for %s (only %d SLTDA points)", d, count)

    if anchors:
        ref = pd.concat([ref, pd.DataFrame(anchors)], ignore_index=True)

    missing = set(DISTRICT_ANCHORS) - set(ref.district)
    if missing:
        log.error("Districts still unrepresented: %s", sorted(missing))

    log.info("Reference set: %d points, %d districts (%d anchors)",
             len(ref), ref.district.nunique(), len(anchors))
    return ref.reset_index(drop=True)


def assign(df: pd.DataFrame, ref: pd.DataFrame,
           k: int = K_NEIGHBOURS, chunk: int = 1000) -> pd.DataFrame:
    """
    Inverse-distance-weighted k-nearest-neighbour vote over the reference points.

    Weighting by 1/distance matters: an unweighted vote lets a cluster of hotels
    just over a district border outvote the single closest point.
    """
    df = df.copy()
    if "district" in df.columns:
        df["district"] = df["district"].map(
            lambda x: canonicalise_district(x) or x)
    ref_lat = ref.lat.to_numpy(float)
    ref_lon = ref.lon.to_numpy(float)
    ref_dist = ref.district.to_numpy()
    ref_w = (ref.weight.to_numpy(float) if "weight" in ref.columns
             else np.ones(len(ref)))

    out_district, out_conf, out_method = [], [], []
    n = len(df)

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = haversine_km(df.lat.to_numpy(float)[start:stop],
                         df.lon.to_numpy(float)[start:stop],
                         ref_lat, ref_lon)
        kk = min(k, d.shape[1])
        idx = np.argpartition(d, kk - 1, axis=1)[:, :kk]

        for row in range(d.shape[0]):
            nb = idx[row]
            dist = d[row, nb]
            if dist.min() > MAX_TRUST_KM:
                out_district.append(None)
                out_conf.append(0.0)
                out_method.append("unassigned")
                continue

            weights: Counter = Counter()
            for j, dd in zip(nb, dist):
                weights[ref_dist[j]] += ref_w[j] / max(dd, 0.25)

            total = sum(weights.values())
            best, w = weights.most_common(1)[0]
            out_district.append(best)
            out_conf.append(round(w / total, 3))
            out_method.append("sltda_knn")

        log.info("  %d/%d POIs assigned", stop, n)

    df["district_knn"] = out_district
    df["district_confidence"] = out_conf
    df["district_method"] = out_method

    # Keep the old centroid result where kNN could not decide.
    fallback = df.district_knn.isna()
    if fallback.any():
        log.warning("%d POIs had no reference point within %.0f km; "
                    "retaining the centroid assignment",
                    int(fallback.sum()), MAX_TRUST_KM)
        df.loc[fallback, "district_knn"] = df.loc[fallback, "district"]
        df.loc[fallback, "district_method"] = "nearest_centroid_fallback"

    changed = int((df.district_knn != df.district).sum())
    log.info("District reassigned for %d of %d POIs (%.1f%%)",
             changed, len(df), 100 * changed / max(len(df), 1))
    log.info("Mean confidence: %.2f; %d assignments below 0.5 confidence",
             df.district_confidence.mean(),
             int((df.district_confidence < 0.5).sum()))

    df["district_previous"] = df["district"]
    df["district"] = df["district_knn"]
    return df.drop(columns=["district_knn"])


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ref = load_reference()

    src = DATA_PROC / "pois.parquet"
    if not src.exists():
        raise SystemExit("Run `python -m etl.clean` first.")

    df = assign(pd.read_parquet(src), ref)
    df.to_parquet(src, index=False)

    moved = df[df.district != df.district_previous]
    if len(moved):
        print("\nLargest reassignments:")
        summary = (moved.groupby(["district_previous", "district"])
                        .size().sort_values(ascending=False).head(12))
        for (old, new), cnt in summary.items():
            print(f"  {old:<15s} -> {new:<15s} {cnt:>5d}")

    print("\nSpot check (expected: Matale for all three):")
    for name in ("Sigiriya", "Dambulla", "Pidurangala"):
        hit = df[df.name.str.contains(name, case=False, na=False)]
        for _, r in hit.head(2).iterrows():
            print(f"  {r['name'][:38]:<40s} {r['district']:<14s} "
                  f"(was {r['district_previous']}, conf {r['district_confidence']:.2f})")

    print(f"\nDistricts covered: {df.district.nunique()}/25")
    print("\nNext: python -m etl.enrich")
    return 0


if __name__ == "__main__":
    sys.exit(main())
