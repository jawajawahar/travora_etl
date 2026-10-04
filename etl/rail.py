"""
Stage 4b - Rail accessibility.

Train is the lowest-emission intercity transport mode in Sri Lanka, and the
network reaches most of the major visitor regions: the coastal line to Galle and
Matara, the main line through Kandy to Badulla via Ella, the northern line to
Jaffna, and the Anuradhapura and Trincomalee branches.

A POI reachable by rail can genuinely be visited at lower carbon cost. This
module extracts every station from OpenStreetMap and computes each POI's and
each accommodation's distance to the nearest one, which feeds the transport
component of S_sust with real data instead of an assumed travel mode.

This is also a distinctly Sri Lankan signal. A sustainability model built on
European city transit assumptions would not transfer; rail accessibility is the
locally meaningful equivalent.

Run:  python -m etl.rail
"""
from __future__ import annotations
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import CONFIG_DIR, DATA_RAW, DATA_PROC

log = logging.getLogger(__name__)

OVERPASS_RAIL = """
[out:json][timeout:180];
area["ISO3166-1"="LK"][admin_level=2]->.lk;
(
  node["railway"="station"](area.lk);
  node["railway"="halt"](area.lk);
  way ["railway"="station"](area.lk);
);
out center tags;
"""

EARTH_KM = 6371.0


def load_sustainability_config() -> dict:
    with open(CONFIG_DIR / "sustainability.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_stations() -> Path:
    """Fetch stations from Overpass and archive the raw response."""
    from .extract_osm import fetch, save_raw
    log.info("Extracting railway stations from Overpass")
    return save_raw(fetch(OVERPASS_RAIL), "osm_rail")


def parse_stations(raw_path: Path) -> pd.DataFrame:
    payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    rows = []
    for el in payload.get("elements", []):
        tags = el.get("tags") or {}
        name = tags.get("name:en") or tags.get("name")
        if el.get("type") == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            c = el.get("center") or {}
            lat, lon = c.get("lat"), c.get("lon")
        if lat is None or lon is None:
            continue
        rows.append({
            "station_id": f"{el['type']}/{el['id']}",
            "name": name or "(unnamed halt)",
            "lat": float(lat), "lon": float(lon),
            "kind": tags.get("railway", "station"),
        })
    df = pd.DataFrame(rows)
    log.info("Parsed %d railway stations/halts", len(df))
    return df


def nearest_station_km(lat: np.ndarray, lon: np.ndarray,
                       st_lat: np.ndarray, st_lon: np.ndarray,
                       chunk: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised nearest-station distance. Returns (km, station_index)."""
    n = len(lat)
    best_km = np.full(n, np.inf)
    best_ix = np.full(n, -1, dtype=int)
    if len(st_lat) == 0:
        return best_km, best_ix

    sla = np.radians(st_lat)[None, :]
    slo = np.radians(st_lon)[None, :]

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        la = np.radians(lat[start:stop])[:, None]
        lo = np.radians(lon[start:stop])[:, None]
        h = (np.sin((sla - la) / 2) ** 2
             + np.cos(la) * np.cos(sla) * np.sin((slo - lo) / 2) ** 2)
        d = 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
        best_ix[start:stop] = np.argmin(d, axis=1)
        best_km[start:stop] = np.min(d, axis=1)
    return best_km, best_ix


def rail_access_score(km: np.ndarray, cfg: dict) -> np.ndarray:
    """
    1.0 = walkable from a station, tapering to 0 beyond the moderate threshold.
    Piecewise linear rather than a hard cut-off, so a POI 8.1 km from a station
    is not treated identically to one 80 km away.
    """
    r = cfg["rail"]
    exc, good, mod = r["excellent_km"], r["good_km"], r["moderate_km"]
    score = np.zeros_like(km, dtype=float)

    score = np.where(km <= exc, 1.0, score)
    m = (km > exc) & (km <= good)
    score = np.where(m, 1.0 - 0.4 * (km - exc) / (good - exc), score)
    m = (km > good) & (km <= mod)
    score = np.where(m, 0.6 * (1.0 - (km - good) / (mod - good)), score)
    return np.clip(score, 0.0, 1.0).round(4)


def annotate(df: pd.DataFrame, stations: pd.DataFrame, cfg: dict,
             label: str) -> pd.DataFrame:
    df = df.copy()
    km, ix = nearest_station_km(
        df.lat.to_numpy(float), df.lon.to_numpy(float),
        stations.lat.to_numpy(float), stations.lon.to_numpy(float))

    df["nearest_station_km"] = np.round(km, 2)
    df["nearest_station"] = [stations.name.iloc[i] if i >= 0 else None for i in ix]
    df["rail_access_score"] = rail_access_score(km, cfg)

    r = cfg["rail"]
    log.info("%s rail access: %.1f%% within %.0f km (walkable), "
             "%.1f%% within %.0f km, %.1f%% within %.0f km",
             label,
             100 * (km <= r["excellent_km"]).mean(), r["excellent_km"],
             100 * (km <= r["good_km"]).mean(), r["good_km"],
             100 * (km <= r["moderate_km"]).mean(), r["moderate_km"])
    return df


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_sustainability_config()

    existing = sorted(DATA_RAW.glob("osm_rail_*.json"))
    raw = existing[-1] if existing else extract_stations()
    stations = parse_stations(raw)
    stations.to_parquet(DATA_PROC / "rail_stations.parquet", index=False)

    for fname, label in (("pois.parquet", "POIs"),
                         ("accommodation.parquet", "Accommodation")):
        path = DATA_PROC / fname
        if not path.exists():
            log.warning("%s not found, skipping", fname)
            continue
        df = annotate(pd.read_parquet(path), stations, cfg, label)
        df.to_parquet(path, index=False)
        log.info("Updated %s", fname)

    print("\nNext: python -m etl.load_neo4j")
    return 0


if __name__ == "__main__":
    sys.exit(main())
