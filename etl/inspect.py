"""
Quick dataset summary. Useful for reporting progress to a supervisor and for
spotting problems before the slow stages run.

Run:  python -m etl.inspect
"""
from __future__ import annotations
import sys

import pandas as pd

from .config import DATA_PROC

from .districts import CANONICAL as ALL_DISTRICTS


def main() -> int:
    src = DATA_PROC / "pois.parquet"
    if not src.exists():
        raise SystemExit("Run `python -m etl.clean` first.")
    df = pd.read_parquet(src)

    bar = "=" * 62
    print(bar)
    print(f"TRAVORA DATASET  -  {len(df):,} POIs")
    print(bar)

    print("\nBy category")
    for c, v in df.category.value_counts().items():
        print(f"  {c:<12s} {v:>6,}  {100 * v / len(df):5.1f}%  {'#' * int(40 * v / len(df))}")

    print("\nBy district")
    counts = df.district.value_counts()
    for d in ALL_DISTRICTS:
        v = int(counts.get(d, 0))
        flag = "  <-- BELOW 5" if v < 5 else ""
        print(f"  {d:<15s} {v:>6,}{flag}")
    missing = [d for d in ALL_DISTRICTS if counts.get(d, 0) < 5]
    print(f"\n  Districts with >=5 POIs: {25 - len(missing)}/25")

    print("\nData quality")
    print(f"  Real opening hours   {100 * (~df.hours_estimated).mean():5.1f}%")
    print(f"  Real entry fees      {100 * (~df.fee_estimated).mean():5.1f}%")
    if "popularity_source" in df.columns:
        got = (df.popularity_source == "wikipedia").mean()
        print(f"  Popularity resolved  {100 * got:5.1f}%")
    if "wikidata" in df.columns:
        print(f"  Has wikidata tag     {100 * df.wikidata.notna().mean():5.1f}%")
        print(f"  Has wikipedia tag    {100 * df.wikipedia.notna().mean():5.1f}%")

    if "prominence" in df.columns:
        print("\nProminence distribution")
        for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
            v = int(((df.prominence >= lo) & (df.prominence < hi)).sum())
            print(f"  {lo:.1f}-{hi if hi <= 1 else 1.0:.1f}  {v:>6,}  "
                  f"{'#' * int(40 * v / len(df))}")
        print("\nTop 15 by prominence")
        cols = ["name", "category", "district", "prominence"]
        if "popularity_index" in df.columns:
            cols.append("popularity_index")
        print(df.nlargest(15, "prominence")[cols].to_string(index=False))

    mpath = DATA_PROC / "travel_matrix.parquet"
    if mpath.exists():
        m = pd.read_parquet(mpath)
        print(f"\nTravel graph: {len(m):,} relationships "
              f"({100 * (m.method == 'osrm').mean():.1f}% OSRM), "
              f"median {m.distance_km.median():.1f} km")
        if len(m) > 400_000:
            print("  WARNING: exceeds the AuraDB Free limit of 400,000")

    print("\n" + bar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
