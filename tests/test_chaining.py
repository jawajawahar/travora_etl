"""
Day chaining: day 1 starts at the traveller's starting point, and each later
day starts where the previous one ended.

Before this, the starting point only bounded retrieval, so a traveller in
Kinniya could be given a first stop in Matale (147 km by road) with the
transfer counted nowhere.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.models import Candidate
from solver.models import SolverConfig
from solver.search import solve
from evaluation.metrics import verify, travel_minutes, _Point, MAX_TRANSFER


def cand(i, lat, lon, district="Kandy"):
    return Candidate(
        poi_id=f"node/{i}", name=f"Place {i}", category="heritage", lat=lat, lon=lon,
        district=district, open_min=480, close_min=1020, entry_fee_lkr=0.0,
        typical_dwell_min=60, popularity_index=0.2, prominence=0.6,
        rail_access_score=0.5, hours_estimated=True, fee_estimated=True,
        distance_from_start_km=5.0)


def near(n, lat, lon, start_id, district):
    return [cand(start_id + i, lat + 0.004 * i, lon + 0.004 * i, district) for i in range(n)]


CFG = SolverConfig(max_slots_per_day=3, min_stops_per_day=2, time_budget_s=5.0)
KANDY = (7.2906, 80.6337)


def test_day_one_stays_reachable_from_the_start():
    """Far places score higher, but are out of reach of the starting point."""
    local = near(5, 7.30, 80.64, 0, "Kandy")
    far = near(5, 9.66, 80.02, 100, "Jaffna")          # about 270 km away
    pref = {c.poi_id: 0.4 for c in local} | {c.poi_id: 0.95 for c in far}
    sust = {c.poi_id: 0.5 for c in local + far}
    res = solve("t", local + far, 1, 1e6, pref, sust, 0.7, {}, CFG, start=KANDY)
    assert res.feasible
    assert {s.district for s in res.itinerary.stops} == {"Kandy"}


def test_first_stop_counts_the_transfer():
    local = near(4, 7.40, 80.70, 0, "Matale")           # ~15 km from the start
    scores = {c.poi_id: 0.7 for c in local}
    res = solve("t", local, 1, 1e6, scores, scores, 0.5, {}, CFG, start=KANDY)
    first = res.itinerary.days[0].stops[0]
    assert first.from_anchor
    assert first.travel_from_prev_min > 0
    assert first.arrive_min > CFG.day_start_min, "the transfer must use day-one time"


def test_day_two_starts_where_day_one_ended():
    local = near(8, 7.30, 80.64, 0, "Kandy")
    scores = {c.poi_id: 0.7 for c in local}
    res = solve("t", local, 2, 1e6, scores, scores, 0.5, {}, CFG, start=KANDY)
    d1, d2 = res.itinerary.days
    assert d2.stops[0].from_anchor
    by_id = {c.poi_id: c for c in local}
    expected = travel_minutes(by_id[d1.stops[-1].poi_id], by_id[d2.stops[0].poi_id])
    assert abs(d2.stops[0].travel_from_prev_min - expected) < 1.0


def test_unreachable_start_is_reported_not_fabricated():
    far = near(6, 9.66, 80.02, 0, "Jaffna")
    scores = {c.poi_id: 0.7 for c in far}
    res = solve("t", far, 1, 1e6, scores, scores, 0.5, {}, CFG, start=(6.03, 80.22))
    assert not res.feasible


def test_verifier_flags_a_plan_that_starts_too_far_away():
    class Req:
        days, poi_budget_lkr = 1, 1e6
        start_lat, start_lon = 6.03, 80.22               # Galle
    far = near(2, 9.66, 80.02, 0, "Jaffna")
    assert travel_minutes(_Point(6.03, 80.22), far[0]) > MAX_TRANSFER
    v = verify([[c.poi_id for c in far]], Req(), {c.poi_id: c for c in far})
    assert not v.feasible and v.violations["transfer_too_long"] == 1


def test_travel_cost_prefers_good_nearby_over_slightly_better_far():
    """Without a travel cost a slightly better place 80 km away always wins."""
    local = near(4, 7.30, 80.64, 0, "Kandy")                  # ~1 km from the start
    far = near(4, 7.98, 80.76, 100, "Matale")                 # ~76 km away
    pref = {c.poi_id: 0.70 for c in local} | {c.poi_id: 0.78 for c in far}
    sust = {c.poi_id: 0.5 for c in local + far}
    from dataclasses import replace
    free = solve("t", local + far, 1, 1e6, pref, sust, 0.7, {},
                 replace(CFG, travel_cost_per_hour=0.0), start=KANDY)
    costed = solve("t", local + far, 1, 1e6, pref, sust, 0.7, {}, CFG, start=KANDY)
    assert {s.district for s in free.itinerary.stops} == {"Matale"}
    assert {s.district for s in costed.itinerary.stops} == {"Kandy"}
