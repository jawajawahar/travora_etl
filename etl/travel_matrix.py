"""
Stage 4 - Build the travel graph.

Rewritten because the all-pairs approach does not scale. With 11,515 POIs there
are 66.3 million pairs, and Neo4j AuraDB Free permits 400,000 relationships. The
all-pairs matrix would run for hours and then fail to load.

k-nearest-neighbour graph instead
---------------------------------
Each POI is connected to its k nearest neighbours. At k=25 that is roughly
178,000 relationships after deduplication, which fits comfortably.

This is not a compromise, it is the correct model. An itinerary never needs a
direct edge between two POIs on opposite ends of the island: a day's stops are
geographically clustered, and multi-day travel is handled by chaining edges. Any
pair the solver actually needs is either a kNN edge or reachable by traversal,
and the solver can fall back to the distance model for a pair with no stored
edge.

Run:  python -m etl.travel_matrix
"""
from __future__ import annotations
import logging
import sys
import time

import numpy as np
import pandas as pd
import requests

from .config import (DATA_PROC, OSRM_URL, USER_AGENT,
                     FALLBACK_AVG_KMH, FALLBACK_DETOUR_FACTOR)

log = logging.getLogger(__name__)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})

K_NEIGHBOURS = 25          # ~178k relationships at 11.5k POIs
CHUNK = 500                # POIs per vectorised distance block
OSRM_TOP_N = 400           # most prominent POIs get real road durations
OSRM_TABLE_SIZE = 80       # coordinates per OSRM /table request

EARTH_KM = 6371.0


def haversine_matrix(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorised great-circle distance in km. Shapes (n,1) against (1,m)."""
    la1 = np.radians(lat1)[:, None]
    lo1 = np.radians(lon1)[:, None]
    la2 = np.radians(lat2)[None, :]
    lo2 = np.radians(lon2)[None, :]
    dlat = la2 - la1
    dlon = lo2 - lo1
    h = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def fallback_minutes(km: np.ndarray | float):
    """Straight-line distance corrected for road detour and average speed."""
    return (km * FALLBACK_DETOUR_FACTOR / FALLBACK_AVG_KMH) * 60.0


def knn_edges(df: pd.DataFrame, k: int = K_NEIGHBOURS) -> pd.DataFrame:
    """Each POI to its k nearest neighbours, deduplicated to one direction."""
    lat = df.lat.to_numpy(dtype=float)
    lon = df.lon.to_numpy(dtype=float)
    ids = df.poi_id.to_numpy()
    n = len(df)
    log.info("Building kNN graph: %d POIs, k=%d", n, k)

    pairs: set[tuple[int, int]] = set()
    dist_lookup: dict[tuple[int, int], float] = {}

    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        d = haversine_matrix(lat[start:stop], lon[start:stop], lat, lon)

        # exclude self-distance
        for local, glob in enumerate(range(start, stop)):
            d[local, glob] = np.inf

        take = min(k, n - 1)
        nearest = np.argpartition(d, take, axis=1)[:, :take]

        for local, glob in enumerate(range(start, stop)):
            for j in nearest[local]:
                a, b = (glob, int(j)) if glob < j else (int(j), glob)
                if a == b:
                    continue
                if (a, b) not in pairs:
                    pairs.add((a, b))
                    dist_lookup[(a, b)] = float(d[local, j])

        log.info("  %d/%d POIs processed, %d unique pairs", stop, n, len(pairs))

    rows = [(ids[a], ids[b], round(km, 2), round(float(fallback_minutes(km)), 1),
             "haversine", a, b)
            for (a, b), km in dist_lookup.items()]

    out = pd.DataFrame(rows, columns=["from_id", "to_id", "distance_km",
                                      "duration_min", "method", "_i", "_j"])
    log.info("kNN graph: %d relationships", len(out))
    return out


def osrm_table(coords: list[tuple[float, float]]) -> np.ndarray | None:
    """Full duration matrix (seconds) for a small coordinate set."""
    coord_str = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in coords)
    try:
        r = SESSION.get(f"{OSRM_URL}/table/v1/driving/{coord_str}",
                        params={"annotations": "duration"}, timeout=120)
        if r.status_code == 429:
            log.warning("OSRM rate limited; pausing 30s")
            time.sleep(30)
            return None
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "Ok":
            return None
        return np.array(data["durations"], dtype=float)
    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning("OSRM request failed: %s", e)
        return None


def refine_with_osrm(df: pd.DataFrame, edges: pd.DataFrame,
                     top_n: int = OSRM_TOP_N) -> pd.DataFrame:
    """
    Replace estimated durations with real road durations for edges between the
    most prominent POIs. These are the POIs most likely to appear in itineraries,
    so accuracy matters most there; everything else keeps the distance model and
    is labelled accordingly.
    """
    if "prominence" not in df.columns:
        log.warning("No prominence column - run `python -m etl.enrich` first. "
                    "Skipping OSRM refinement.")
        return edges

    top_idx = set(df.prominence.nlargest(min(top_n, len(df))).index.tolist())
    mask = edges._i.isin(top_idx) & edges._j.isin(top_idx)
    target = edges[mask]
    log.info("OSRM refinement: %d edges among the %d most prominent POIs",
             len(target), len(top_idx))
    if target.empty:
        return edges

    ordered = sorted(top_idx)
    pos = {g: p for p, g in enumerate(ordered)}
    coords = [(float(df.lat.iloc[g]), float(df.lon.iloc[g])) for g in ordered]

    durations = np.full((len(ordered), len(ordered)), np.nan)
    blocks = range(0, len(ordered), OSRM_TABLE_SIZE)
    for bi, start in enumerate(blocks, 1):
        stop = min(start + OSRM_TABLE_SIZE, len(ordered))
        sub = osrm_table(coords[start:stop])
        if sub is None:
            log.warning("OSRM unavailable - keeping estimated durations")
            break
        durations[start:stop, start:stop] = sub
        log.info("  OSRM block %d/%d", bi, len(list(blocks)))
        time.sleep(1.0)

    edges = edges.copy()
    updated = 0
    for row_idx, row in target.iterrows():
        a, b = pos.get(row._i), pos.get(row._j)
        if a is None or b is None:
            continue
        sec = durations[a, b]
        if np.isfinite(sec) and sec > 0:
            edges.at[row_idx, "duration_min"] = round(sec / 60.0, 1)
            edges.at[row_idx, "method"] = "osrm"
            updated += 1

    log.info("OSRM durations applied to %d edges (%.1f%% of graph)",
             updated, 100 * updated / max(len(edges), 1))
    return edges


def build(df: pd.DataFrame, use_osrm: bool = True,
          k: int = K_NEIGHBOURS) -> pd.DataFrame:
    edges = knn_edges(df, k=k)
    if use_osrm:
        edges = refine_with_osrm(df, edges)
    return edges.drop(columns=["_i", "_j"])


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    src = DATA_PROC / "pois.parquet"
    if not src.exists():
        raise SystemExit("Run `python -m etl.clean` first.")

    df = pd.read_parquet(src).reset_index(drop=True)
    edges = build(df)
    dest = DATA_PROC / "travel_matrix.parquet"
    edges.to_parquet(dest, index=False)

    share = 100 * (edges.method == "osrm").mean() if len(edges) else 0.0
    log.info("Wrote %s: %d relationships, %.1f%% from OSRM",
             dest.name, len(edges), share)
    if len(edges) > 380_000:
        log.warning("Relationship count is close to the AuraDB Free limit of "
                    "400,000. Lower K_NEIGHBOURS if the load fails.")
    print("\nNext: python -m etl.load_neo4j")
    return 0


if __name__ == "__main__":
    sys.exit(main())
