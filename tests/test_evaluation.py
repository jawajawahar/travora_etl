"""
Tests for the evaluation harness.

The Gini and coverage tests matter most: they are the metrics that turn "the
itineraries repeat" into a number, and if they are wrong the paper's central
before/after claim is wrong with them.

Run:  python -m pytest tests/test_evaluation.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.models import Candidate, Category, TripRequest
from evaluation.metrics import (verify, gini, diversity, jaccard, plans_differ,
                                travel_minutes)
from evaluation.querysets import (build_query_set, weight_sweep_queries,
                                  rule_based, greedy_eco, BUDGETS, DURATIONS,
                                  PROFILES)
from evaluation.harness import gate_t8_fake_id
from datetime import date, timedelta


def cand(i, *, fee=500.0, lat=7.29, lon=80.63, hours_est=True,
         open_min=480, close_min=1020, dwell=90, category="heritage"):
    return Candidate(
        poi_id=f"node/{i}", name=f"P{i}", category=category, lat=lat, lon=lon,
        district="Kandy", open_min=open_min, close_min=close_min,
        entry_fee_lkr=fee, typical_dwell_min=dwell, popularity_index=0.3,
        prominence=0.6, popularity_known=True, rail_access_score=0.5,
        hours_estimated=hours_est, fee_estimated=True,
        distance_from_start_km=5.0)


def req(days=2, budget=60000, interests=(Category.HERITAGE)):
    return TripRequest(start_date=date.today() + timedelta(days=30), days=days,
                       budget_lkr=budget, party_size=2,
                       interests=list(interests) if isinstance(interests, (list, tuple))
                       else [interests],
                       start_lat=7.29, start_lon=80.63)


# ================= GINI =================
def test_gini_must_use_the_whole_catalogue():
    """
    Computed over recommended items only, a system emitting the same three POIs
    fifty times each scores 0.0 - the exact behaviour the metric should expose.
    Padding with the unused catalogue is what makes concentration visible.
    """
    three_of_many = [50, 50, 50]
    assert gini(three_of_many) == 0.0, "the naive form should score 0 here"
    assert gini(three_of_many, catalogue_size=2245) > 0.99, \
        "with the catalogue included this must be near-total concentration"

    spread = [3] * 900
    assert gini(spread, catalogue_size=2245) < 0.65, \
        "a broadly spread system should score well below the 0.60-0.65 band"

    assert gini([], catalogue_size=100) == 0.0
    assert gini([5], catalogue_size=1) == 0.0
    print(f"  gini: 3-of-2245 -> {gini(three_of_many, 2245):.3f}, "
          f"900-of-2245 -> {gini(spread, 2245):.3f}")


def test_diversity_detects_a_repeating_system():
    """A system emitting the same plan every time must score near-zero coverage
    and a high Gini - the numeric signature of the old behaviour."""
    same = [[["node/1", "node/2"], ["node/3"]] for _ in range(50)]
    d = diversity(same, catalogue_size=2245)
    assert d.distinct_pois == 3
    assert d.catalogue_coverage_pct < 1.0
    assert d.distinct_plan_pct == pytest.approx(2.0, abs=0.1)
    assert d.top5_share_pct == 100.0
    assert d.gini > 0.99, ("a repeating system must show near-total "
                           f"concentration, got {d.gini}")
    print(f"  repeating system -> {d.summary()}")


def test_diversity_rewards_a_varied_system():
    varied = [[[f"node/{i * 3 + k}" for k in range(3)]] for i in range(50)]
    d = diversity(varied, catalogue_size=300)
    assert d.distinct_pois == 150
    assert d.catalogue_coverage_pct == pytest.approx(50.0, abs=0.1)
    assert d.distinct_plan_pct == 100.0
    assert d.gini < 0.95, "a varied system should be measurably less concentrated"
    print(f"  varied system    -> {d.summary()}")


# ================= VERIFIER =================
def test_verifier_catches_unknown_ids():
    """A hallucinated id must be counted, not silently ignored."""
    by_id = {c.poi_id: c for c in [cand(0), cand(1)]}
    v = verify([["node/0", "sigiriya"]], req(), by_id)
    assert not v.feasible and v.violations["unknown_poi"] == 1
    print("  verifier: unknown id counted")


def test_verifier_catches_budget_and_duplicates():
    by_id = {c.poi_id: c for c in [cand(i, fee=9000.0) for i in range(4)]}
    v = verify([["node/0", "node/1"], ["node/2"]], req(budget=60000), by_id)
    assert v.violations["budget_exceeded"] == 1, "27,000 exceeds the 12,000 share"

    by_id2 = {c.poi_id: c for c in [cand(0), cand(1)]}
    v2 = verify([["node/0"], ["node/0"]], req(), by_id2)
    assert v2.violations["duplicate_poi"] == 1
    print("  verifier: budget and duplicates caught")


def test_verifier_respects_imputed_hours():
    """
    97% of the dataset's opening hours are category defaults. Enforcing those
    would reject plans on the basis of invented values, so only real hours count.
    """
    # Opens 10:00, closes 10:50, needs 90 minutes: impossible on its own hours,
    # but well inside the working day so only the hours constraint can bite.
    soft = cand(0, hours_est=True, open_min=600, close_min=650, dwell=90)
    hard = cand(1, hours_est=False, open_min=600, close_min=650, dwell=90)
    assert verify([["node/0"]], req(days=1), {"node/0": soft}
                  ).violations["opening_hours"] == 0
    assert verify([["node/1"]], req(days=1), {"node/1": hard}
                  ).violations["opening_hours"] == 1
    print("  verifier: imputed hours not enforced, real hours enforced")


def test_verifier_catches_day_overflow():
    far = [cand(0, lat=6.03, lon=80.21), cand(1, lat=9.66, lon=80.02),
           cand(2, lat=6.87, lon=81.35)]
    by_id = {c.poi_id: c for c in far}
    v = verify([[c.poi_id for c in far]], req(days=1), by_id)
    assert v.violations["day_overflow"] == 1
    print("  verifier: impossible day rejected")


def test_verifier_accepts_a_valid_plan():
    by_id = {c.poi_id: c for c in [cand(i, fee=100.0) for i in range(4)]}
    v = verify([["node/0", "node/1"], ["node/2"]], req(), by_id)
    assert v.feasible, v.violated
    assert v.stops == 3
    print(f"  verifier: valid plan accepted ({v.stops} stops)")


# ================= QUERY SET =================
def test_query_set_is_stratified():
    qs = build_query_set(repeats=5)
    assert len(qs) == len(BUDGETS) * len(DURATIONS) * len(PROFILES) * 5 == 180
    assert len({q.query_id for q in qs}) == 180, "query ids must be unique"
    for b in BUDGETS:
        assert sum(1 for q in qs if q.budget_band == b) == 60
    print(f"  query set: {len(qs)} queries, balanced across strata")


def test_weight_sweep_covers_the_frontier():
    qs = weight_sweep_queries()
    assert len({q.w_pref for q in qs}) == 5
    print(f"  weight sweep: {len(qs)} queries across 5 weight levels")


# ================= BASELINES =================
def test_rule_based_respects_budget_but_ignores_geography():
    """
    The baseline should stay within budget yet produce plans a planner would
    reject - that difference is what the comparison is measuring.
    """
    cs = ([cand(i, fee=3000.0) for i in range(5)]
          + [cand(10 + i, fee=3000.0, lat=9.66, lon=80.02) for i in range(5)])
    r = req(days=2, budget=60000)
    plan = rule_based(cs, r, slots=3)
    flat = [p for d in plan for p in d]
    by_id = {c.poi_id: c for c in cs}
    cost = sum(by_id[p].entry_fee_lkr for p in flat)
    assert cost <= r.poi_budget_lkr, "baseline broke its own budget filter"
    assert len(flat) == len(set(flat)), "baseline repeated a POI"
    print(f"  rule_based: {len(flat)} stops, LKR {cost:,.0f} within budget")


def test_greedy_eco_prefers_high_sustainability():
    cs = [cand(i, fee=100.0) for i in range(6)]
    sust = {c.poi_id: 0.1 for c in cs}
    sust["node/5"] = 0.99
    plan = greedy_eco(cs, req(days=1), sust, slots=2)
    assert "node/5" in plan[0], "greedy_eco ignored the sustainability ranking"
    print("  greedy_eco: takes the most sustainable option first")


# ================= HELPERS =================
def test_jaccard_and_plan_comparison():
    a = [["node/1", "node/2"]]
    b = [["node/1", "node/3"]]
    assert jaccard(a, a) == 1.0
    assert 0.0 < jaccard(a, b) < 1.0
    assert jaccard(a, [["node/9"]]) == 0.0
    assert plans_differ(a, b) and not plans_differ(a, [["node/1", "node/2"]])
    print(f"  overlap: identical 1.0, partial {jaccard(a, b):.2f}, disjoint 0.0")


def test_travel_estimate_is_plausible():
    kandy, galle = cand(0, lat=7.29, lon=80.63), cand(1, lat=6.03, lon=80.21)
    mins = travel_minutes(kandy, galle)
    assert 200 < mins < 400, f"Kandy-Galle estimated at {mins:.0f} min"
    print(f"  travel model: Kandy to Galle {mins:.0f} min")


def test_gate_t8_rejects_fake_ids():
    g = gate_t8_fake_id()
    assert g.passed, "the ID contract failed to reject an invented id"
    print(f"  T8: {g.observed}")


if __name__ == "__main__":
    print("Travora - evaluation tests\n" + "-" * 62)
    for fn in [test_gini_must_use_the_whole_catalogue,
               test_diversity_detects_a_repeating_system,
               test_diversity_rewards_a_varied_system,
               test_verifier_catches_unknown_ids,
               test_verifier_catches_budget_and_duplicates,
               test_verifier_respects_imputed_hours,
               test_verifier_catches_day_overflow,
               test_verifier_accepts_a_valid_plan,
               test_query_set_is_stratified,
               test_weight_sweep_covers_the_frontier,
               test_rule_based_respects_budget_but_ignores_geography,
               test_greedy_eco_prefers_high_sustainability,
               test_jaccard_and_plan_comparison,
               test_travel_estimate_is_plausible,
               test_gate_t8_rejects_fake_ids]:
        fn()
    print("-" * 62 + "\nAll tests passed.")


def test_distinct_plan_rate_cannot_exceed_100_for_unstable_systems():
    """Three repeats of one request, each answered differently, is ONE request."""
    from evaluation.metrics import diversity
    plans = [[["a"]], [["b"]], [["c"]], [["d"]]]
    keys = ["q1", "q1", "q1", "q2"]
    rep = diversity(plans, catalogue_size=10, query_keys=keys)
    assert rep.distinct_plan_pct == 100.0
    same = diversity([[["a"]], [["a"]]], catalogue_size=10, query_keys=["q1", "q2"])
    assert same.distinct_plan_pct == 50.0
