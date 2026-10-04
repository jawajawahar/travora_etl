"""
Real distances and times for the legs of a finished itinerary.

The solver searches with estimated travel times (a stored OSRM edge where the
graph has one, otherwise straight-line distance x 1.35 at 35 km/h). Once a plan
is chosen there are only a dozen or so legs, so they can be measured properly:

  road  - OSRM routes the whole trip, starting point to last stop, in ONE
          request; each leg of the response is one leg of the trip.
  rail  - a train leg is offered only when both ends are near a station AND
          the two stations are joined by track in the OSM rail graph. Its
          length is the track distance, not the straight line.

Anything that cannot be measured falls back to the estimate and says so.
"""
from __future__ import annotations
import heapq
import json
import logging
import math
import threading
from functools import lru_cache
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RAIL_GRAPH = ROOT / "data" / "processed" / "rail_graph.json"
OSRM_CACHE = ROOT / "data" / "processed" / "osrm_cache.json"

# Estimate used only when a road leg cannot be measured.
DETOUR = 1.35
EST_KMH = 35.0

# Rail model: effective speeds including intermediate stops, one for lowland
# track and one for the hill country, where the line climbs and winds (the
# Kadugannawa incline and the Main Line beyond Kandy to Badulla). Fitted by
# least squares against reference journey times in etl/validate_transport.py,
# which also reports leave-one-out error; see data/evaluation/transport_validation.md.
TRAIN_KMH = {"lowland": 47.6, "hill": 24.1, "fast": 64.0, "slow": 24.0}
HILL_BOX = (6.80, 7.50, 80.38, 81.10)        # lat_min, lat_max, lon_min, lon_max
STATION_ACCESS_KM = 3.0          # further than this, the station is not "near"
ACCESS_KMH = 20.0                # tuk-tuk to or from the station
TRAIN_WAIT_MIN = 20.0            # average wait; no open timetable exists

_lock = threading.Lock()


def haversine_km(a_lat, a_lon, b_lat, b_lon) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a_lat, a_lon, b_lat, b_lon))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def estimate_road(a, b) -> tuple[float, float]:
    km = haversine_km(a[0], a[1], b[0], b[1]) * DETOUR
    return km, km / EST_KMH * 60.0


# ---------------------------------------------------------------------------
# Road (OSRM)
# ---------------------------------------------------------------------------
def _key(a, b) -> str:
    return f"{a[0]:.4f},{a[1]:.4f}|{b[0]:.4f},{b[1]:.4f}"


def _load_cache() -> dict:
    try:
        return json.loads(OSRM_CACHE.read_text(encoding="utf-8"))
    except Exception:                                   # noqa: BLE001
        return {}


_cache: dict = _load_cache()


def road_legs(points: list[tuple[float, float]], osrm_url: str,
              timeout: float = 8.0) -> list[Optional[tuple[float, float]]]:
    """
    Road (km, minutes) for each consecutive pair of points, or None per leg
    that could not be measured. One OSRM request covers the whole sequence.
    """
    pairs = list(zip(points, points[1:]))
    out: list[Optional[tuple[float, float]]] = [None] * len(pairs)
    missing = []
    for i, (a, b) in enumerate(pairs):
        hit = _cache.get(_key(a, b))
        if hit:
            out[i] = (hit[0], hit[1])
        else:
            missing.append(i)
    if not missing or not osrm_url:
        return out

    coords = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in points)
    try:
        r = requests.get(f"{osrm_url.rstrip('/')}/route/v1/driving/{coords}",
                         params={"overview": "false"}, timeout=timeout,
                         headers={"User-Agent": "TravoraResearch/1.0"})
        r.raise_for_status()
        legs = r.json()["routes"][0]["legs"]
    except Exception as e:                              # noqa: BLE001
        log.warning("OSRM unavailable (%s); road legs estimated", str(e)[:80])
        return out

    with _lock:
        for i, leg in enumerate(legs[:len(pairs)]):
            km, mins = leg["distance"] / 1000.0, leg["duration"] / 60.0
            out[i] = (km, mins)
            _cache[_key(*pairs[i])] = [round(km, 3), round(mins, 2)]
        try:
            OSRM_CACHE.write_text(json.dumps(_cache), encoding="utf-8")
        except Exception:                               # noqa: BLE001
            pass
    return out


# ---------------------------------------------------------------------------
# Rail
# ---------------------------------------------------------------------------
def is_hill(lat: float, lon: float) -> bool:
    la0, la1, lo0, lo1 = HILL_BOX
    return la0 <= lat <= la1 and lo0 <= lon <= lo1


class RailNetwork:
    def __init__(self, path: Path = RAIL_GRAPH, speeds: Optional[dict] = None):
        g = json.loads(path.read_text(encoding="utf-8"))
        nodes = g["nodes"]
        # Each edge carries (neighbour, km, hill?) so a path can be timed by
        # terrain as well as measured by length.
        self.adj: dict[str, list[tuple[str, float, bool]]] = {}
        for a, b, km, *rest in g["edges"]:
            (la, lo), (lb, lob) = nodes[a], nodes[b]
            # Terrain first: hill track is slow whatever line it is on.
            if is_hill((la + lb) / 2, (lo + lob) / 2):
                cls = "hill"
            else:
                cls = rest[0] if rest and rest[0] in ("fast", "slow") else "lowland"
            self.adj.setdefault(a, []).append((b, km, cls))
            self.adj.setdefault(b, []).append((a, km, cls))
        self.stations = g["stations"]
        self.speeds = speeds or TRAIN_KMH

    def nearest_station(self, lat: float, lon: float) -> Optional[tuple[dict, float]]:
        best, best_km = None, STATION_ACCESS_KM
        for st in self.stations:
            d = haversine_km(lat, lon, st["lat"], st["lon"])
            if d < best_km:
                best, best_km = st, d
        return (best, best_km) if best else None

    @lru_cache(maxsize=4096)
    def path(self, a: str, b: str) -> Optional[tuple[float, float, float]]:
        """
        Fastest path along the track: (minutes, km, {class: km}), or None when
        the two nodes are not connected.
        """
        if a == b:
            return 0.0, 0.0, {}
        best = {a: 0.0}
        heap = [(0.0, 0.0, a, ())]
        while heap:
            t, km, n, parts = heapq.heappop(heap)
            if n == b:
                split: dict[str, float] = {}
                for cls, w in parts:
                    split[cls] = split.get(cls, 0.0) + w
                return t, km, split
            if t > best.get(n, math.inf):
                continue
            for m, w, cls in self.adj.get(n, ()):
                nt = t + w / self.speeds[cls] * 60.0
                if nt < best.get(m, math.inf):
                    best[m] = nt
                    heapq.heappush(heap, (nt, km + w, m, parts + ((cls, w),)))
        return None

    def station_named(self, name: str) -> Optional[dict]:
        """Exact name, then without "Railway Station", then a whole-word match.
        A plain substring match turned "Ella" into "Avissawella"."""
        import re
        want = name.lower().strip()
        clean = lambda s: re.sub(r"\s+(railway\s+)?station$", "", s.lower().strip())
        for test in (lambda s: s.lower() == want, lambda s: clean(s) == want,
                     lambda s: re.search(r"\b" + re.escape(want) + r"\b", s.lower())):
            hit = [s for s in self.stations if test(s["name"])]
            if hit:
                return hit[0]
        return None

    def train(self, a: tuple[float, float], b: tuple[float, float]) -> Optional[dict]:
        sa, sb = self.nearest_station(*a), self.nearest_station(*b)
        if not sa or not sb or sa[0]["node"] == sb[0]["node"]:
            return None
        p = self.path(sa[0]["node"], sb[0]["node"])
        if p is None:
            return None
        ride, km, _ = p
        access = (sa[1] + sb[1]) * DETOUR / ACCESS_KMH * 60.0
        return {
            "track_km": km,
            "ride_min": ride,
            "minutes": access + TRAIN_WAIT_MIN + ride,
            "from_station": sa[0]["name"], "to_station": sb[0]["name"],
        }


_rail: Optional[RailNetwork] = None


def rail() -> Optional[RailNetwork]:
    """The rail graph, loaded once; None if etl.rail_network has not been run."""
    global _rail
    if _rail is None and RAIL_GRAPH.exists():
        try:
            _rail = RailNetwork()
        except Exception as e:                          # noqa: BLE001
            log.warning("Rail graph unreadable: %s", e)
    return _rail
