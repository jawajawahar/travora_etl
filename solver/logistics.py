"""
Transport mode and lodging.

Two gaps this closes.

First, the sustainability score already contained a transport term derived from
rail accessibility, but the itinerary never said HOW a traveller moves between
stops, and the emission factors in sustainability.yaml were never used. A score
that claims to measure transport impact while the plan omits the mode is
measuring an intention, not a journey.

Second, the graph holds 2,129 registered properties, each with a sustainability
score validated against room counts, and none of them appeared in an itinerary.
Accommodation is where a traveller spends most of a trip budget and where the
community-benefit argument in the tourism register actually applies, so leaving
it out removed the strongest part of the sustainability case.

Lodging is chosen AFTER the stops are fixed rather than inside the search. A
hotel does not change which places are reachable; it changes where the traveller
sleeps. Folding it into the constraint search would multiply the search space
for no gain in feasibility.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from math import radians, sin, cos, asin, sqrt
from pathlib import Path
from typing import Optional

import yaml

from .routing import road_legs, estimate_road, rail
from .emissions import kg_per_traveller, journey_score, trip_sustainability

log = logging.getLogger(__name__)

CONFIG = Path(__file__).resolve().parent.parent / "config" / "sustainability.yaml"

MODE_LABEL = {
    "walk": "Walk", "train": "Train", "bus": "Bus", "tuktuk": "Tuk-tuk",
    "car_shared": "Shared car", "car_private": "Private car",
}




def haversine_km(a_lat, a_lon, b_lat, b_lon) -> float:
    la1, lo1, la2, lo2 = map(radians, (a_lat, a_lon, b_lat, b_lon))
    h = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(h))


# ---------------------------------------------------------------------------
# Travel-time assumptions for modes with no open timing data. They are stated
# here and in the reason shown to the traveller, not hidden in a formula.
WALK_KMH = 4.5
TUKTUK_FACTOR, TUKTUK_WAIT_MIN = 1.1, 5.0
# Buses stop often and wait to fill; bus time is the road driving time scaled.
BUS_FACTOR, BUS_WAIT_MIN = 1.4, 10.0
# Train is chosen when it is no more than this much slower than the bus: it is
# the lowest-emission option, but not at any cost in time.
TRAIN_TOLERANCE = 1.25

START_ID = "__start__"


@dataclass
class Leg:
    from_id: str
    to_id: str
    distance_km: float
    duration_min: float
    mode: str
    emissions_g: float
    rationale: str
    from_name: str = ""
    kind: str = "between"            # start | overnight | between
    basis: str = "estimate"          # road (OSRM) | track (rail graph) | estimate
    stations: Optional[tuple[str, str]] = None

    @property
    def mode_label(self) -> str:
        return MODE_LABEL.get(self.mode, self.mode)

    @property
    def estimated(self) -> bool:
        return self.basis == "estimate"


def plan_leg(a: tuple[float, float], b: tuple[float, float],
             road: Optional[tuple[float, float]], net=None) -> dict:
    """
    Mode, time and distance for one leg.

    `road` is the measured (km, minutes) driving route, or None when it could
    not be measured, in which case the straight-line estimate is used and the
    leg is marked as estimated. A train is offered only when both ends are near
    a station AND the stations are joined by track.
    """
    if road is not None:
        road_km, road_min, basis = road[0], road[1], "road"
    else:
        road_km, road_min = estimate_road(a, b)
        basis = "estimate"
    note = "" if basis == "road" else " Distance estimated: the road route could not be measured."

    if road_km <= 1.5:
        return dict(mode="walk", minutes=road_km / WALK_KMH * 60.0, km=road_km, basis=basis,
                    why="Close enough to walk." + note)
    if road_km <= 6.0:
        return dict(mode="tuktuk", minutes=road_min * TUKTUK_FACTOR + TUKTUK_WAIT_MIN,
                    km=road_km, basis=basis,
                    why="A short hop, normally taken by tuk-tuk." + note)

    bus_min = road_min * BUS_FACTOR + BUS_WAIT_MIN
    if road_km >= 20.0 and net is not None:
        t = net.train(a, b)
        if t and t["minutes"] <= bus_min * TRAIN_TOLERANCE:
            return dict(mode="train", minutes=t["minutes"], km=t["track_km"], basis="track",
                        stations=(t["from_station"], t["to_station"]),
                        why=(f"Train from {t['from_station']} to {t['to_station']}, "
                             f"{t['track_km']:.0f} km of track. The time includes getting to "
                             "and from the stations and a typical wait; check the Sri Lanka "
                             "Railways timetable for departures."))
    if road_km >= 60.0:
        return dict(mode="car_shared", minutes=road_min, km=road_km, basis=basis,
                    why=("A long road journey with no practical rail link; a shared vehicle "
                         "keeps emissions per person lower." + note))
    return dict(mode="bus", minutes=bus_min, km=road_km, basis=basis,
                why="Public bus; the time allows for stops and waiting." + note)


def build_legs(stops_by_day: list, by_id: dict, start: Optional[tuple[float, float]] = None,
               start_name: Optional[str] = None, osrm_url: Optional[str] = None,
               party_size: int = 1) -> list[Leg]:
    """
    Every leg of the trip: from the starting point to the first stop, between
    stops, and each morning from where the previous day ended. The whole trip is
    measured in one road-routing request.
    """
    entries = []            # (point, id, name, kind of the leg INTO this entry or None)
    if start is not None:
        entries.append(((float(start[0]), float(start[1])), START_ID,
                        start_name or "your starting point", None))
    for day_stops in stops_by_day:
        for j, stop in enumerate(day_stops):
            c = by_id.get(stop.poi_id)
            if c is None:
                continue
            kind = "between" if j > 0 else (
                ("start" if entries and entries[-1][1] == START_ID else "overnight")
                if getattr(stop, "from_anchor", False) else None)
            entries.append(((c.lat, c.lon), stop.poi_id, stop.name, kind if entries else None))

    if len(entries) < 2:
        return []
    roads = road_legs([e[0] for e in entries], osrm_url) if osrm_url else [None] * (len(entries) - 1)
    net = rail()

    legs: list[Leg] = []
    for k in range(len(entries) - 1):
        frm, to = entries[k], entries[k + 1]
        if to[3] is None:
            continue
        leg = plan_leg(frm[0], to[0], roads[k], net)
        legs.append(Leg(
            from_id=frm[1], to_id=to[1], from_name=frm[2], kind=to[3],
            distance_km=round(leg["km"], 1), duration_min=round(leg["minutes"], 1),
            # Per traveller: a shared tuk-tuk or car is divided among its riders.
            mode=leg["mode"],
            emissions_g=round(kg_per_traveller(leg["mode"], leg["km"], party_size) * 1000.0, 1),
            rationale=leg["why"], basis=leg["basis"], stations=leg.get("stations"),
        ))
    return legs


def retime(stops_by_day: list, legs: list[Leg], by_id: dict, day_start: int = 480,
           day_end: int = 1080, buffer: int = 15):
    """
    Lay the days out again with the measured leg times.

    The solver schedules with estimated times; once the legs are measured the
    clock can differ. Returns ({poi_id: (arrive, depart)}, {day: minutes past
    day_end}, {poi_ids reached after they close}), so a day that no longer fits
    is reported rather than silently shown with times it cannot keep.
    """
    into = {l.to_id: l for l in legs}
    times, overflow, late = {}, {}, set()
    for d, day_stops in enumerate(stops_by_day, 1):
        clock = float(day_start)
        for stop in day_stops:
            leg = into.get(stop.poi_id)
            if leg is not None:
                clock += leg.duration_min + buffer
            c = by_id.get(stop.poi_id)
            arrive = clock
            if c is not None and getattr(c, "hours_are_hard", False):
                arrive = max(arrive, c.open_min)
                if arrive + stop.dwell_min > c.close_min:
                    late.add(stop.poi_id)
            depart = arrive + stop.dwell_min
            times[stop.poi_id] = (int(round(arrive)), int(round(depart)))
            clock = depart
        overflow[d] = max(0, int(round(clock - day_end)))
    return times, overflow, late


def emissions_summary(legs: list[Leg], party_size: int = 1, days: int = 1,
                      place_mean: Optional[float] = None) -> dict:
    """
    Trip emissions, the same journeys by the party's own car for comparison, and
    the journey part of the sustainability score.

    Leg emissions are per traveller, so the party total multiplies by party
    size. kg per traveller-day is the headline figure: it compares trips of
    different lengths and group sizes on the same scale.
    """
    party = max(party_size, 1)
    per_traveller = sum(l.emissions_g for l in legs) / 1000.0
    total = per_traveller * party * 1000.0
    km = sum(l.distance_km for l in legs)
    baseline = sum(kg_per_traveller("car_private", l.distance_km, party)
                   for l in legs) * party * 1000.0
    saved = max(0.0, baseline - total)
    per_day = per_traveller / max(days, 1)
    by_mode: dict[str, float] = {}
    for l in legs:
        by_mode[l.mode] = round(by_mode.get(l.mode, 0.0) + l.distance_km, 1)
    return {
        "total_km": round(km, 1),
        "total_kg_co2": round(total / 1000.0, 2),
        "private_car_kg_co2": round(baseline / 1000.0, 2),
        "saved_kg_co2": round(saved / 1000.0, 2),
        "saved_pct": round(100.0 * saved / baseline, 1) if baseline else 0.0,
        "km_by_mode": by_mode,
        "kg_co2e_per_traveller_day": round(per_day, 2),
        "journey_score": journey_score(per_day),
        **({"trip_sustainability": trip_sustainability(place_mean, per_day)}
           if place_mean is not None else {}),
    }


# ---------------------------------------------------------------------------
NEARBY_ACCOMMODATION = """
MATCH (a:Accommodation)
WHERE point.distance(a.location,
      point({latitude: $lat, longitude: $lon})) <= $radius_m
  AND a.sustainability_score IS NOT NULL
RETURN a.acc_id AS acc_id, a.name AS name, a.sltda_type AS type,
       a.district AS district, a.rooms AS rooms,
       a.sustainability_score AS sustainability_score,
       coalesce(a.rail_access_score, 0.0) AS rail_access_score,
       coalesce(a.location_estimated, false) AS location_estimated,
       point.distance(a.location,
         point({latitude: $lat, longitude: $lon})) / 1000.0 AS distance_km
ORDER BY a.sustainability_score DESC, distance_km ASC
LIMIT $limit
"""

# Rough nightly rates by register category, in LKR for a double room. These are
# ESTIMATES: the register records category and room count but not price, and the
# commercial pricing APIs cannot be stored. Every figure shown to a traveller is
# flagged as an estimate for this reason.
NIGHTLY_RATE_LKR = {
    "Home Stay Units": 4000, "Heritage Homes": 12000, "Heritage Bungalows": 14000,
    "Rented Homes": 6000, "Bangalows": 8000, "Rented Apartments": 7000,
    "Guest Houses": 6500, "Boutique Villas": 22000, "Boutique Hotels": 25000,
    "Tourist Hotels": 15000, "Classified Hotels( 1-5 Star)": 35000,
}
DEFAULT_RATE = 10000


@dataclass
class Stay:
    night: int
    acc_id: str
    name: str
    type: str
    district: str
    rooms: Optional[int]
    sustainability_score: float
    distance_km: float
    est_rate_lkr: float
    location_estimated: bool = False
    rate_estimated: bool = True


def choose_lodging(session, days, budget_lkr: float,
                   party_size: int = 1, radius_km: float = 25.0) -> tuple[list[Stay], dict]:
    """
    Pick one property per night, near the last stop of that day.

    Selection prefers the most sustainable property that fits the remaining
    budget. If nothing affordable is nearby, the night is left unassigned and
    reported rather than filled with something the traveller cannot afford.
    """
    stays: list[Stay] = []
    unassigned: list[int] = []
    spent = 0.0
    nights = max(len(days) - 1, 0)          # no stay needed after the last day

    for i in range(nights):
        day = days[i]
        if not day.stops:
            unassigned.append(day.day)
            continue
        anchor = day.stops[-1]

        try:
            rows = [dict(r) for r in session.run(
                NEARBY_ACCOMMODATION,
                lat=float(getattr(anchor, "lat", 0.0) or 0.0),
                lon=float(getattr(anchor, "lon", 0.0) or 0.0),
                radius_m=radius_km * 1000.0, limit=25)]
        except Exception as e:                          # noqa: BLE001
            log.warning("accommodation lookup failed for day %d: %s", day.day, e)
            rows = []

        picked = None
        for r in rows:
            rate = NIGHTLY_RATE_LKR.get(r["type"], DEFAULT_RATE)
            if spent + rate <= budget_lkr:
                picked = (r, rate)
                break

        if picked is None:
            unassigned.append(day.day)
            continue

        r, rate = picked
        spent += rate
        stays.append(Stay(
            night=day.day, acc_id=r["acc_id"], name=r["name"],
            type=r["type"], district=r["district"], rooms=r.get("rooms"),
            sustainability_score=round(float(r["sustainability_score"]), 3),
            distance_km=round(float(r["distance_km"]), 1),
            est_rate_lkr=float(rate),
            location_estimated=bool(r.get("location_estimated")),
        ))

    summary = {
        "nights_needed": nights,
        "nights_assigned": len(stays),
        "unassigned_nights": unassigned,
        "total_est_lkr": round(spent, 0),
        "mean_sustainability": round(
            sum(s.sustainability_score for s in stays) / len(stays), 3) if stays else 0.0,
        "note": "Nightly rates are estimates by property category. The register "
                "records category and room count but not price.",
    }
    if unassigned:
        log.info("No affordable accommodation found near stops on night(s) %s", unassigned)
    return stays, summary
