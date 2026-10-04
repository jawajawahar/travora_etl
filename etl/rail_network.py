"""
Stage 5c - Sri Lanka rail network as a routable graph.

    python -m etl.rail_network          # fetch from Overpass (if needed) and build
    python -m etl.rail_network --build  # rebuild from the cached raw dump

Station coordinates alone cannot say whether two stations are on the same line,
so "both stops are near a station" was being read as "a train connects them".
This builds the track itself: every railway=rail way becomes edges between its
consecutive nodes, weighted by length. A train leg is then offered only when a
path exists along the track, and its length is the track distance, not the
straight line.

Output: data/processed/rail_graph.json
  nodes:    {osm_id: [lat, lon]}         only nodes used by track
  edges:    [[a, b, km], ...]             undirected
  stations: [{name, lat, lon, node}]      snapped to the nearest track node
"""
from __future__ import annotations
import argparse
import json
import logging
import math
import sys
from datetime import date

import requests

from .config import DATA_RAW, DATA_PROC, OVERPASS_URL, USER_AGENT

log = logging.getLogger(__name__)

OUT = DATA_PROC / "rail_graph.json"
SNAP_KM = 0.6          # a station further than this from any track is dropped
HEAL_KM = 0.05         # dead-end track this close to other track is joined
# Lines rebuilt for higher running speeds (OSM carries 80-120 km/h limits on
# them); their edges are classed "fast" so they are not timed like older track.
FAST_LINES = {"Northern Line", "Mannar Line"}
# Older single-track lines in the east, where trains wait at crossing loops.
SLOW_LINES = {"Trincomalee Line", "Batticaloa Line"}

QUERY = """
[out:json][timeout:240];
area["ISO3166-1"="LK"][admin_level=2]->.lk;
(
  way["railway"="rail"](area.lk);
  node["railway"~"^(station|halt)$"](area.lk);
  way["railway"~"^(station|halt)$"](area.lk);
);
out body geom;
"""


def _km(a, b) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def fetch() -> dict:
    path = DATA_RAW / f"osm_rail_lines_{date.today():%Y-%m-%d}.json"
    log.info("Fetching rail lines from Overpass (%s)...", OVERPASS_URL)
    r = requests.post(OVERPASS_URL, data={"data": QUERY},
                      headers={"User-Agent": USER_AGENT}, timeout=300)
    r.raise_for_status()
    data = r.json()
    path.write_text(json.dumps(data), encoding="utf-8")
    log.info("Saved %s (%d elements)", path.name, len(data.get("elements", [])))
    return data


def latest_raw() -> dict | None:
    files = sorted(DATA_RAW.glob("osm_rail_lines_*.json"))
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


def build(data: dict) -> dict:
    nodes: dict[str, list[float]] = {}
    edges: dict[tuple[str, str], float] = {}
    fast: set[tuple[str, str]] = set()
    slow: set[tuple[str, str]] = set()
    stations_raw = []

    for el in data.get("elements", []):
        tags = el.get("tags") or {}
        if el["type"] == "way" and tags.get("railway") == "rail":
            # Disused, abandoned or construction track carries no trains.
            if any(k in tags for k in ("disused", "abandoned", "construction")):
                continue
            ids = [str(n) for n in el.get("nodes", [])]
            geom = el.get("geometry") or []
            if len(ids) != len(geom):
                continue
            for nid, g in zip(ids, geom):
                nodes[nid] = [round(g["lat"], 6), round(g["lon"], 6)]
            is_fast = tags.get("name") in FAST_LINES
            is_slow = tags.get("name") in SLOW_LINES
            for a, b in zip(ids, ids[1:]):
                if a == b:
                    continue
                k = (a, b) if a < b else (b, a)
                edges[k] = round(_km(nodes[a], nodes[b]), 4)
                if is_fast:
                    fast.add(k)
                if is_slow:
                    slow.add(k)
        elif tags.get("railway") in ("station", "halt"):
            # Stations are mapped both as points and as areas (Batticaloa is an
            # area); an area is represented by the centre of its outline.
            if el["type"] == "node":
                lat, lon, osm = el["lat"], el["lon"], str(el["id"])
            elif el.get("geometry"):
                pts = el["geometry"]
                lat = sum(q["lat"] for q in pts) / len(pts)
                lon = sum(q["lon"] for q in pts) / len(pts)
                osm = f"way/{el['id']}"
            else:
                continue
            stations_raw.append({"name": tags.get("name:en") or tags.get("name") or "Station",
                                 "lat": lat, "lon": lon, "osm": osm})

    grid: dict[tuple[int, int], list[str]] = {}
    for nid, (la, lo) in nodes.items():
        grid.setdefault((int(la * 50), int(lo * 50)), []).append(nid)

    # Heal digitisation gaps: a way that ends within HEAL_KM of other track but
    # does not share its node leaves the network split (the Batticaloa line was
    # unreachable from Colombo for this reason). Only dead ends are joined, so
    # genuinely separate parallel tracks are not merged.
    degree: dict[str, int] = {}
    for a, b in edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    healed = 0
    for nid, deg in list(degree.items()):
        if deg != 1:
            continue
        la, lo = nodes[nid]
        gx, gy = int(la * 50), int(lo * 50)
        best, best_km = None, HEAL_KM
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in grid.get((gx + dx, gy + dy), []):
                    if other == nid or ((nid, other) if nid < other else (other, nid)) in edges:
                        continue
                    d = _km(nodes[nid], nodes[other])
                    if d < best_km:
                        best, best_km = other, d
        if best:
            edges[(nid, best) if nid < best else (best, nid)] = round(best_km, 4)
            healed += 1
    log.info("Healed %d track gaps under %.0f m", healed, HEAL_KM * 1000)

    # Snap each station to the nearest track node. A station node usually sits
    # on the track already; one placed beside it is snapped if close enough.

    stations = []
    for st in stations_raw:
        if st["osm"] in nodes:
            stations.append({"name": st["name"], "lat": st["lat"], "lon": st["lon"],
                             "node": st["osm"]})
            continue
        gx, gy = int(st["lat"] * 50), int(st["lon"] * 50)
        best, best_km = None, SNAP_KM
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for nid in grid.get((gx + dx, gy + dy), []):
                    d = _km((st["lat"], st["lon"]), nodes[nid])
                    if d < best_km:
                        best, best_km = nid, d
        if best:
            stations.append({"name": st["name"], "lat": st["lat"], "lon": st["lon"],
                             "node": best})

    graph = {"nodes": nodes,
             "edges": [[a, b, km, "fast" if (a, b) in fast else "slow" if (a, b) in slow else "std"]
                       for (a, b), km in edges.items()],
             "stations": stations,
             "built": date.today().isoformat()}
    return graph


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="build from the cached dump only")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data = latest_raw() if args.build else None
    if data is None:
        data = fetch()
    g = build(data)
    OUT.write_text(json.dumps(g), encoding="utf-8")
    total_km = sum(e[2] for e in g["edges"])
    print(f"Rail graph: {len(g['nodes']):,} nodes, {len(g['edges']):,} edges, "
          f"{total_km:,.0f} km of track, {len(g['stations'])} stations snapped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
