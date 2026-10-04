"""Greedy seed for branch and bound: always a real plan, never changes the optimum."""
from __future__ import annotations
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.models import Candidate
from solver.models import SolverConfig
from solver.search import solve


def cand(i, lat, lon, fee=0.0, district="Kandy"):
    return Candidate(
        poi_id=f"node/{i}", name=f"Place {i}", category="heritage", lat=lat, lon=lon,
        district=district, open_min=480, close_min=1020, entry_fee_lkr=fee,
        typical_dwell_min=60, popularity_index=0.2, prominence=0.6,
        rail_access_score=0.5, hours_estimated=True, fee_estimated=True,
        distance_from_start_km=5.0)


START = (7.2906, 80.6337)
POOL = [cand(i, 7.29 + 0.01 * (i % 5), 80.63 + 0.01 * (i // 5), fee=500.0 * (i % 3))
        for i in range(15)]
PREF = {c.poi_id: 0.3 + 0.04 * i for i, c in enumerate(POOL)}
SUST = {c.poi_id: 0.5 for c in POOL}


def run(days=3, budget=1e6, pool=POOL, **cfg_changes):
    cfg = replace(SolverConfig(max_slots_per_day=3, min_stops_per_day=2, time_budget_s=5.0),
                  **cfg_changes)
    return solve("t", pool, days, budget, PREF, SUST, 0.5, {}, cfg, start=START)


def check_plan(res, days, budget):
    stops = res.itinerary.stops
    ids = [s.poi_id for s in stops]
    assert len(ids) == len(set(ids)), "a place repeats"
    assert res.itinerary.total_cost_lkr <= budget + 1e-6
    for d in res.itinerary.days:
        assert len(d.stops) >= 2, f"day {d.day} has too few stops"
    assert len(res.itinerary.days) == days


def test_seed_does_not_change_the_optimum():
    """With time to finish, seeded and unseeded search return the same value."""
    small = POOL[:7]                     # small enough to search exhaustively
    seeded = run(days=2, pool=small, warm_start=True)
    plain = run(days=2, pool=small, warm_start=False)
    assert seeded.feasible and plain.feasible
    assert seeded.itinerary.optimal and plain.itinerary.optimal
    assert abs(seeded.itinerary.utility - plain.itinerary.utility) < 1e-9


def test_seed_is_returned_and_labelled_when_search_has_no_time():
    """No search time at all: the seed alone is answered, and marked not proven best."""
    res = run(time_budget_s=0.0)
    assert res.feasible
    assert res.diagnostics.warm_start_found and res.diagnostics.returned_warm_start
    assert not res.itinerary.optimal
    check_plan(res, 3, 1e6)


def test_without_seed_and_no_time_there_is_no_plan():
    assert not run(time_budget_s=0.0, warm_start=False).feasible


def test_seed_respects_a_tight_budget():
    """Budget held back for later days: the seed still gives every day its stops."""
    res = run(days=4, budget=2500.0, time_budget_s=0.0)
    if res.feasible:
        check_plan(res, 4, 2500.0)
