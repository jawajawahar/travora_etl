"""
Stage 2b - Ingest the SLTDA registered accommodation register.

Why this replaces the TripAdvisor plan
--------------------------------------
TripAdvisor ratings measure guest SATISFACTION, not sustainability. A 200-room
resort with three pools outscores a three-room family homestay on rating, so a
sustainability score built on ratings would measure the wrong construct.

SLTDA's registration categories are a sustainability instrument by the
authority's own statement: the Home Stay category exists to empower local
communities, distribute economic benefit, extend tourism into rural areas and
support "sustainable and responsible development of Eco and rural tourism".
That is a citable, authoritative basis for the accommodation term in S_sust.

The ordering is corroborated inside the register itself: median room count rises
monotonically across the category scale. This module reports the Spearman
correlation between the assigned category score and observed median rooms, which
is an empirical validation of the model rather than an assumption.

Coordinate recovery
-------------------
Roughly 36% of records lack coordinates, and the gap is NOT random: 100% of
Classified Hotels are geocoded versus 56% of Home Stays and 33% of Rented
Apartments. Dropping ungeocoded rows would delete precisely the small,
community-scale properties the sustainability objective exists to surface, and
would bake luxury bias into the dataset. Missing coordinates are therefore
recovered from AGA-division centroids computed from the geocoded records in the
same file, falling back to district centroids, and every recovered location is
flagged.

Run:  python -m etl.extract_sltda --csv path/to/sltda_accommodations.csv
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import CONFIG_DIR, DATA_PROC, DATA_RAW, LK_BBOX

log = logging.getLogger(__name__)

RENAME = {
    "Type": "sltda_type", "Name": "name", "Address": "address",
    "Rooms": "rooms", "Grade": "grade", "District": "district",
    "AGA Division": "aga_division", "PS/MC/UC": "local_authority",
    "Logitiute": "lon", "Latitude": "lat",       # SLTDA's spelling of Longitude
}


def load_config() -> dict:
    with open(CONFIG_DIR / "sustainability.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(RENAME) - set(df.columns)
    if missing:
        raise ValueError(f"SLTDA CSV missing expected columns: {sorted(missing)}")

    df = df.rename(columns=RENAME)[list(RENAME.values())].copy()
    df["name"] = df["name"].astype(str).str.strip()
    df["sltda_type"] = df["sltda_type"].astype(str).str.strip()
    df["grade"] = df["grade"].astype(str).str.strip().str.upper().replace({"NAN": None})
    df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # A stable id: SLTDA publishes no identifier column.
    df["acc_id"] = ("sltda/" + df.index.astype(str).str.zfill(5))

    log.info("Loaded %d accommodation records", len(df))
    return df


def validate_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Null out anything outside Sri Lanka rather than trusting it."""
    inside = (df.lat.between(LK_BBOX["min_lat"], LK_BBOX["max_lat"])
              & df.lon.between(LK_BBOX["min_lon"], LK_BBOX["max_lon"]))
    bad = df.lat.notna() & ~inside
    if bad.any():
        log.warning("Discarding %d coordinates outside Sri Lanka", int(bad.sum()))
        df.loc[bad, ["lat", "lon"]] = np.nan
    return df


def recover_coords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing coordinates from centroids computed within this same dataset:
    AGA division first, then district. Every filled row is flagged.
    """
    df = df.copy()
    df["location_estimated"] = df.lat.isna() | df.lon.isna()
    before = int(df.location_estimated.sum())

    geo = df[~df.location_estimated]
    div = geo.groupby("aga_division")[["lat", "lon"]].median()
    dis = geo.groupby("district")[["lat", "lon"]].median()

    filled_div = filled_dis = 0
    for i in df.index[df.location_estimated]:
        d = df.at[i, "aga_division"]
        if isinstance(d, str) and d in div.index:
            df.at[i, "lat"], df.at[i, "lon"] = div.at[d, "lat"], div.at[d, "lon"]
            df.at[i, "location_method"] = "aga_division_centroid"
            filled_div += 1
            continue
        k = df.at[i, "district"]
        if isinstance(k, str) and k in dis.index:
            df.at[i, "lat"], df.at[i, "lon"] = dis.at[k, "lat"], dis.at[k, "lon"]
            df.at[i, "location_method"] = "district_centroid"
            filled_dis += 1

    df.loc[~df.location_estimated, "location_method"] = "sltda_published"
    unresolved = int(df.lat.isna().sum())

    log.info("Coordinates: %d published, %d from division centroid, "
             "%d from district centroid, %d unresolved",
             len(df) - before, filled_div, filled_dis, unresolved)
    if unresolved:
        log.warning("%d records still have no location and will be dropped", unresolved)
    return df[df.lat.notna()].copy()


# ---------------------------------------------------------------------------
def score_size(rooms: pd.Series, cfg: dict) -> pd.Series:
    """
    Smaller properties score higher. Log scale, because the difference between
    3 and 10 rooms matters far more than between 300 and 307.
    """
    s = cfg["size"]
    r = rooms.fillna(rooms.median()).clip(s["rooms_floor"], s["rooms_ceiling"])
    lo = np.log(s["rooms_floor"])
    hi = np.log(s["rooms_ceiling"])
    return (1.0 - (np.log(r) - lo) / (hi - lo)).clip(0, 1)


def score_accommodation(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    cat_map = cfg["accommodation_category"]
    default = cfg["_default_category_score"]

    df["category_score"] = df.sltda_type.map(cat_map).fillna(default)
    unknown = df.loc[~df.sltda_type.isin(cat_map), "sltda_type"].unique()
    if len(unknown):
        log.warning("Unmapped SLTDA categories scored at default %.2f: %s",
                    default, list(unknown))

    df["size_score"] = score_size(df.rooms, cfg)
    df["grade_adj"] = df.grade.map(cfg["grade_adjustment"]).fillna(0.0)

    w = cfg["accommodation_weights"]
    df["sustainability_score"] = (
        w["category"] * df.category_score + w["size"] * df.size_score + df.grade_adj
    ).clip(0, 1).round(4)

    # C_acc is a COST: higher means worse.
    df["acc_impact"] = (1.0 - df.sustainability_score).round(4)
    return df


def validate_ordering(df: pd.DataFrame) -> dict:
    """
    Empirical check that the category ordering is not arbitrary: does the
    assigned category score correlate with observed median room count?
    A strong negative Spearman rho means small categories really are small.
    """
    from scipy import stats as st
    g = df.groupby("sltda_type").agg(
        category_score=("category_score", "first"),
        median_rooms=("rooms", "median"),
        n=("acc_id", "size"),
    ).dropna()
    if len(g) < 3:
        return {}
    rho, p = st.spearmanr(g.category_score, g.median_rooms)
    log.info("Category-score vs median-rooms Spearman rho = %.3f (p = %.4f, n = %d categories)",
             rho, p, len(g))
    return {"rho": float(rho), "p": float(p), "n_categories": int(len(g)),
            "table": g.sort_values("category_score", ascending=False)}


# ---------------------------------------------------------------------------
def run(csv_path: Path) -> pd.DataFrame:
    cfg = load_config()
    df = load_raw(csv_path)
    df = validate_coords(df)
    df = recover_coords(df)
    df = score_accommodation(df, cfg)

    out = DATA_PROC / "accommodation.parquet"
    df.to_parquet(out, index=False)
    log.info("Wrote %s with %d records", out.name, len(df))
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest the SLTDA accommodation register")
    ap.add_argument("--csv", default=None,
                    help="path to sltda_accommodations.csv "
                         "(default: newest match in data/raw/)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.csv:
        path = Path(args.csv)
    else:
        found = sorted(DATA_RAW.glob("*sltda*accommodation*.csv"))
        if not found:
            raise SystemExit(
                "No SLTDA CSV found. Place it in data/raw/ or pass --csv PATH.")
        path = found[-1]

    df = run(path)
    cfg = load_config()
    check = validate_ordering(df)

    print("\n" + "=" * 72)
    print("SLTDA ACCOMMODATION  -  sustainability model")
    print("=" * 72)
    if check:
        print(check["table"].to_string(
            float_format=lambda x: f"{x:.2f}" if isinstance(x, float) else x))
        print(f"\nOrdering validation: Spearman rho = {check['rho']:.3f} "
              f"(p = {check['p']:.4f}) between assigned category score and "
              f"observed median rooms.")
        if check["rho"] < -0.8:
            print("Strong negative correlation: the category ordering is "
                  "corroborated by independent room-count data in the register.")

    print(f"\nLocation provenance:")
    print(df.location_method.value_counts().to_string())

    print(f"\nSustainability score by category:")
    print(df.groupby("sltda_type").sustainability_score
            .agg(["count", "mean"]).round(3).sort_values("mean", ascending=False)
            .to_string())

    print(f"\nDistricts covered: {df.district.nunique()}/25")
    print("\nNext: python -m etl.rail   (then etl.load_neo4j)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
