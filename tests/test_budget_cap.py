"""The domain cap must not turn a request that fits the budget into one that does not."""
from __future__ import annotations
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.models import Candidate
from solver.models import SolverConfig
from solver.search import solve


def cand(i, fee):
    return Candidate(
        poi_id=f"node/{i}", name=f"Place {i}", category="heritage",
        lat=7.29 + 0.002 * i, lon=80.63, district="Kandy", open_min=480, close_min=1020,
        entry_fee_lkr=fee, typical_dwell_min=60, popularity_index=0.2, prominence=0.6,
        rail_access_score=0.5, hours_estimated=True, fee_estimated=True,
        distance_from_start_km=5.0)


# Famous, dear places score highest; free ones score lowest, so a top-K cap by
# utility keeps only the dear ones.
DEAR = [cand(i, 5000.0) for i in range(6)]
FREE = [cand(10 + i, 0.0) for i in range(4)]
POOL = DEAR + FREE
PREF = {c.poi_id: (0.9 if c.entry_fee_lkr else 0.3) for c in POOL}
SUST = {c.poi_id: 0.5 for c in POOL}


def run(budget, **changes):
    cfg = replace(SolverConfig(max_slots_per_day=3, min_stops_per_day=2, time_budget_s=3.0,
                               max_domain_per_slot=4, travel_cost_per_hour=0.0), **changes)
    return solve("b", POOL, 1, budget, PREF, SUST, 0.5, {}, cfg)


def test_cap_alone_refuses_a_request_that_fits_the_budget():
    """The failure being fixed: two free stops fit LKR 3,000, but the cap cut them."""
    res = run(3000.0, keep_cheapest_when_budget_binds=False)
    assert not res.feasible


def test_cheapest_places_are_kept_when_the_budget_binds():
    res = run(3000.0)
    assert res.feasible
    assert res.diagnostics.budget_keep_added > 0
    assert res.itinerary.total_cost_lkr <= 3000.0


def test_nothing_is_added_when_the_budget_does_not_bind():
    res = run(1e6)
    assert res.feasible and res.diagnostics.budget_keep_added == 0


def test_genuinely_unaffordable_request_is_still_refused():
    """Even the full pool cannot pay: nothing is added and the answer stays no."""
    only_dear = replace(SolverConfig(max_slots_per_day=3, min_stops_per_day=2,
                                     time_budget_s=3.0, max_domain_per_slot=4,
                                     travel_cost_per_hour=0.0))
    res = solve("b", DEAR, 1, 3000.0, PREF, SUST, 0.5, {}, only_dear)
    assert not res.feasible
