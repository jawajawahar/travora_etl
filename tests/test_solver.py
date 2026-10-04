"""
Tests for the CSP solver.

The critical properties verified here:

  * AC-3 is SOUND - it never removes a value that appears in a solution. This is
    checked by brute force on a small instance rather than asserted.
  * Feasibility is real - an independent verifier re-derives every constraint
    from the produced itinerary rather than trusting the solver.
  * Imputed data is not enforced as a hard constraint. 97% of opening hours in
    the dataset are category defaults; enforcing those would reject feasible
    itineraries on the basis of invented values.
  * Infeasibility is reported, never fabricated into a plan.

Run:  python -m pytest tests/test_solver.py -v
"""
from __future__ import annotations
import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.models import Candidate
from solver.csp import ItineraryCSP, TravelModel, Slot, NULL
from solver.models import SolverConfig, ConflictType
from solver.search import solve, BranchAndBound


# ---------------------------------------------------------------------------
def cand(i, *, lat=7.29, lon=80.63, fee=1000.0, dwell=90,
         open_min=480, close_min=1020, hours_est=True, fee_est=True,
         district="Kandy", category="heritage"):
    return Candidate(
        poi_id=f"node/{i}", name=f"Place {i}", category=category,
        lat=lat, lon=lon, district=district,
        open_min=open_min, close_min=close_min, entry_fee_lkr=fee,
        typical_dwell_min=dwell, popularity_index=0.2, prominence=0.6,
        rail_access_score=0.5, hours_estimated=hours_est,
        fee_estimated=fee_est, distance_from_start_km=5.0)


def cluster(n, *, spread=0.01, **kw):
    """n candidates close enough that travel between them is short."""
    return [cand(i, lat=7.29 + i * spread, lon=80.63 + i * spread, **kw)
            for i in range(n)]


def flat_scores(cs, v=0.7):
    return {c.poi_id: v for c in cs}


# ------------------------------------------------------- AC-3 soundness
def test_ac3_is_sound_against_brute_force():
    """
    AC-3 must never remove a value that participates in a real solution.

    Verified by enumerating every assignment on a deliberately tiny instance and
    checking that each value appearing in some valid solution survives
    propagation. This is the property the whole approach rests on: if AC-3 were
    unsound it could silently discard the best itinerary.
    """
    cs = cluster(4)
    cfg = SolverConfig(max_slots_per_day=2, min_stops_per_day=1,
                       day_start_min=480, day_end_min=1020)
    travel = TravelModel(cs, {})
    csp = ItineraryCSP(cs, travel, days=1, poi_budget=100000, cfg=cfg)
    csp.build()

    before = {s: set(v) for s, v in csp.domains.items()}
    assert csp.ac3(), "this instance should be arc-consistent"

    slots = csp.slots
    surviving_in_solution: dict[Slot, set[str]] = {s: set() for s in slots}
    for combo in itertools.product(*[sorted(before[s]) for s in slots]):
        assignment = dict(zip(slots, combo))
        ok = all(csp._consistent(xi, assignment[xi], xj, assignment[xj])
                 for xi in slots for xj in slots if xi != xj)
        if ok:
            for s, v in assignment.items():
                surviving_in_solution[s].add(v)

    for s in slots:
        lost = surviving_in_solution[s] - csp.domains[s]
        assert not lost, (
            f"AC-3 removed value(s) {sorted(lost)} from {s} that appear in a "
            f"valid solution - the algorithm is unsound")
    print("  AC-3 soundness: no solution-bearing value removed (brute force)")


def test_ac3_prunes_unreachable_candidates():
    """A POI too far from every other must be removed from adjacent slots."""
    near = cluster(3)
    far = cand(99, lat=9.66, lon=80.02)         # Jaffna, hours from Kandy
    cs = near + [far]
    cfg = SolverConfig(max_slots_per_day=2, max_leg_minutes=60.0)
    csp = ItineraryCSP(cs, TravelModel(cs, {}), days=1,
                       poi_budget=100000, cfg=cfg)
    csp.build()
    assert csp.ac3()

    slot1 = Slot(1, 1)
    assert "node/99" not in csp.domains[slot1], \
        "unreachable candidate should be pruned from the follow-on slot"
    assert csp.diagnostics.domain_reduction_pct > 0
    print(f"  AC-3 pruning: {csp.diagnostics.domain_reduction_pct:.1f}% domain "
          f"reduction, {csp.diagnostics.arc_revisions} revisions")


def test_ac3_wipeout_reports_infeasible():
    """An impossible budget must produce a conflict, not an itinerary."""
    cs = cluster(3, fee=50000.0, fee_est=False)
    res = solve("r", cs, days=1, poi_budget=100.0,
                pref=flat_scores(cs), sust=flat_scores(cs), w_pref=0.5,
                travel_edges={})
    assert res.feasible is False
    assert res.itinerary is None, "an infeasible request must not return a plan"
    assert res.conflict.conflict in (ConflictType.DOMAIN_WIPEOUT,
                                     ConflictType.NO_FEASIBLE_ASSIGNMENT)
    print(f"  infeasibility: reported as {res.conflict.conflict.value}")


# ------------------------------------------------- constraint verification
def verify(itin, cs, cfg: SolverConfig, poi_budget: float) -> list[str]:
    """Independent verifier: re-derives constraints from the produced plan."""
    by_id = {c.poi_id: c for c in cs}
    errs: list[str] = []

    ids = itin.poi_ids
    if len(ids) != len(set(ids)):
        errs.append("duplicate POI in itinerary")
    if itin.total_cost_lkr > poi_budget + 1e-6:
        errs.append(f"budget exceeded: {itin.total_cost_lkr} > {poi_budget}")

    for day in itin.days:
        for s in day.stops:
            c = by_id[s.poi_id]
            if s.depart_min - s.arrive_min != c.typical_dwell_min:
                errs.append(f"{s.poi_id}: dwell mismatch")
            if s.arrive_min < cfg.day_start_min or s.depart_min > cfg.day_end_min:
                errs.append(f"{s.poi_id}: outside working day")
            # opening hours only where the source supplied them
            if c.hours_are_hard and (s.arrive_min < c.open_min
                                     or s.depart_min > c.close_min):
                errs.append(f"{s.poi_id}: violates real opening hours")
        for a, b in zip(day.stops, day.stops[1:]):
            if b.arrive_min < a.depart_min:
                errs.append(f"{b.poi_id}: starts before previous stop ends")
    return errs


def test_solution_passes_independent_verification():
    cs = cluster(8, fee=500.0)
    cfg = SolverConfig(max_slots_per_day=2, time_budget_s=5.0)
    res = solve("r", cs, days=2, poi_budget=20000.0,
                pref=flat_scores(cs, 0.8), sust=flat_scores(cs, 0.6),
                w_pref=0.5, travel_edges={}, cfg=cfg)

    assert res.feasible, res.conflict
    errs = verify(res.itinerary, cs, cfg, 20000.0)
    assert not errs, f"verifier found violations: {errs}"
    assert len(res.itinerary.stops) >= 2
    print(f"  verification: {len(res.itinerary.stops)} stops, "
          f"LKR {res.itinerary.total_cost_lkr:.0f}, 0 violations")


def test_budget_is_respected_exactly():
    cs = cluster(6, fee=3000.0, fee_est=False)
    res = solve("r", cs, days=2, poi_budget=7000.0,
                pref=flat_scores(cs), sust=flat_scores(cs), w_pref=0.5,
                travel_edges={},
                cfg=SolverConfig(max_slots_per_day=3, min_stops_per_day=1))
    assert res.feasible
    assert res.itinerary.total_cost_lkr <= 7000.0, \
        f"budget breached: {res.itinerary.total_cost_lkr}"
    print(f"  budget: LKR {res.itinerary.total_cost_lkr:.0f} of 7,000 cap")


def test_imputed_hours_are_not_enforced():
    """
    A POI whose recorded hours are impossible for the working day must still be
    usable when those hours were imputed, and must be excluded when they are
    real. 97% of the dataset's hours are category defaults.
    """
    impossible = dict(open_min=1200, close_min=1380)     # 20:00-23:00

    soft = [cand(i, hours_est=True, **impossible) for i in range(4)]
    r1 = solve("r", soft, days=1, poi_budget=50000,
               pref=flat_scores(soft), sust=flat_scores(soft), w_pref=0.5,
               travel_edges={})
    assert r1.feasible, "imputed hours must not make a POI unusable"
    assert r1.diagnostics.soft_violations.get("hours_imputed", 0) > 0, \
        "soft violation should be recorded even though it is not enforced"

    hard = [cand(i, hours_est=False, **impossible) for i in range(4)]
    r2 = solve("r", hard, days=1, poi_budget=50000,
               pref=flat_scores(hard), sust=flat_scores(hard), w_pref=0.5,
               travel_edges={})
    assert not r2.feasible, "real opening hours outside the day must be enforced"
    print("  hardness: imputed hours soft (recorded), real hours enforced")


# --------------------------------------------------------- objective
def test_higher_utility_candidates_are_preferred():
    cs = cluster(6, fee=100.0)
    pref = {c.poi_id: 0.1 for c in cs}
    for c in cs[:2]:
        pref[c.poi_id] = 0.95                      # two clearly better options
    res = solve("r", cs, days=1, poi_budget=50000, pref=pref,
                sust={c.poi_id: 0.5 for c in cs}, w_pref=1.0,
                travel_edges={}, cfg=SolverConfig(max_slots_per_day=2))
    assert res.feasible
    chosen = set(res.itinerary.poi_ids)
    assert chosen & {cs[0].poi_id, cs[1].poi_id}, \
        "search should select the high-utility candidates"
    print(f"  objective: selected {sorted(chosen)}")


def test_weights_shift_the_selection():
    """w_pref must actually change what is chosen, or the weighting is decorative."""
    cs = cluster(6, fee=100.0)
    pref = {c.poi_id: 0.0 for c in cs}
    sust = {c.poi_id: 0.0 for c in cs}
    pref[cs[0].poi_id] = 1.0                       # best on preference only
    sust[cs[5].poi_id] = 1.0                       # best on sustainability only
    cfg = SolverConfig(max_slots_per_day=1, min_stops_per_day=1)

    a = solve("r", cs, 1, 50000, pref, sust, w_pref=1.0, travel_edges={}, cfg=cfg)
    b = solve("r", cs, 1, 50000, pref, sust, w_pref=0.0, travel_edges={}, cfg=cfg)
    assert a.feasible and b.feasible
    assert a.itinerary.poi_ids == [cs[0].poi_id]
    assert b.itinerary.poi_ids == [cs[5].poi_id]
    print("  weights: w_pref=1 and w_pref=0 select different POIs")


# --------------------------------------------------------- travel model
def test_travel_provenance_measured_over_itinerary_legs():
    """
    The stored-vs-estimated counts must describe the legs in the FINAL plan.

    They were originally taken from TravelModel's lookup counters, but AC-3's
    support check short-circuits on the first supporting value, so those counters
    depend on arbitrary set-iteration order. Reporting them would have put a
    meaningless number in the evaluation table.
    """
    cs = cluster(4)
    stored = {("node/0", "node/1"): 12.0, ("node/1", "node/2"): 9.0,
              ("node/0", "node/2"): 8.0, ("node/2", "node/3"): 7.0,
              ("node/1", "node/3"): 6.0, ("node/0", "node/3"): 5.0}
    res = solve("r", cs, days=1, poi_budget=50000,
                pref=flat_scores(cs), sust=flat_scores(cs), w_pref=0.5,
                travel_edges=stored, cfg=SolverConfig(max_slots_per_day=2))
    assert res.feasible
    d = res.diagnostics
    legs = [s for s in res.itinerary.stops if s.slot > 0]
    assert d.travel_edges_stored + d.travel_edges_estimated == len(legs), (
        "provenance counts must total the number of legs travelled, not the "
        "number of lookups performed during propagation")
    assert d.travel_edges_stored == len(legs), \
        "every leg here has a stored edge and should be counted as such"
    assert all(not s.travel_estimated for s in legs)
    print(f"  travel provenance: {d.travel_edges_stored}/{len(legs)} legs from "
          f"stored edges (lookups: {d.travel_lookups_stored} stored, "
          f"{d.travel_lookups_estimated} estimated)")


def test_missing_travel_edges_are_flagged_not_invented():
    """With no stored edges at all, every leg must be flagged as estimated."""
    cs = cluster(4)
    res = solve("r", cs, days=1, poi_budget=50000,
                pref=flat_scores(cs), sust=flat_scores(cs), w_pref=0.5,
                travel_edges={}, cfg=SolverConfig(max_slots_per_day=2))
    assert res.feasible
    legs = [s for s in res.itinerary.stops if s.slot > 0]
    assert legs, "expected at least one leg"
    assert all(s.travel_estimated for s in legs), \
        "legs with no stored edge must be flagged, never silently invented"
    assert res.diagnostics.travel_edges_stored == 0
    print(f"  travel fallback: all {len(legs)} leg(s) flagged as estimated")


def test_diagnostics_report_search_bounds():
    cs = cluster(10, fee=100.0)
    cfg = SolverConfig(max_slots_per_day=3, time_budget_s=0.05)
    res = solve("r", cs, days=3, poi_budget=50000,
                pref=flat_scores(cs), sust=flat_scores(cs), w_pref=0.5,
                travel_edges={}, cfg=cfg)
    d = res.diagnostics
    assert d.search_nodes > 0
    assert d.variables == 9
    assert d.domain_size_before >= d.domain_size_after
    if res.feasible and (d.hit_time_budget or d.hit_node_cap):
        assert res.itinerary.optimal is False, \
            "a bounded search must not claim optimality"
    print(f"  diagnostics: {d.variables} vars, {d.search_nodes} nodes, "
          f"{d.domain_reduction_pct:.1f}% reduction, "
          f"optimal={res.itinerary.optimal if res.feasible else 'n/a'}")


def test_too_few_candidates_reports_conflict():
    cs = cluster(1)
    res = solve("r", cs, days=5, poi_budget=50000,
                pref=flat_scores(cs), sust=flat_scores(cs), w_pref=0.5,
                travel_edges={})
    assert not res.feasible
    assert res.conflict.conflict == ConflictType.TOO_FEW_CANDIDATES
    print("  guard: too few candidates reported before search")


# ---------------------------------------------------------------------------
# AC-3 ablation: propagation must never change the optimum
# ---------------------------------------------------------------------------
def test_ac3_never_changes_the_optimum():
    """
    Solving with and without arc consistency must yield the same objective.

    This caught a real defect. `max_leg_minutes` was enforced inside the CSP's
    reachability constraint but not inside the day scheduler, so the propagator
    and the search disagreed about feasibility: enabling AC-3 pruned itineraries
    the search would happily have built, and returned a WORSE objective than
    disabling it (3.285 against 3.342 on a 40-candidate instance). Arc
    consistency is only sound relative to the constraints the search actually
    enforces, so the two must agree exactly.

    This test is also the M3 ablation in the evaluation protocol.
    """
    import numpy as np
    rng = np.random.default_rng(7)
    cs = [cand(i, lat=float(rng.uniform(6.0, 9.5)), lon=float(rng.uniform(79.9, 81.6)),
               fee=float(rng.choice([0, 1000, 3000])),
               dwell=int(rng.choice([60, 90, 120]))) for i in range(40)]
    pref = {c.poi_id: float(rng.random()) for c in cs}
    sust = {c.poi_id: float(rng.random()) for c in cs}

    results = {}
    for use_ac3 in (False, True):
        cfg = SolverConfig(max_slots_per_day=2, min_stops_per_day=2,
                           max_leg_minutes=90.0, time_budget_s=30.0,
                           max_nodes=5_000_000, use_ac3=use_ac3)
        r = solve("s", cs, 2, 20000.0, pref, sust, 0.5, {}, cfg)
        assert r.feasible, f"instance should be feasible (ac3={use_ac3})"
        assert r.itinerary.optimal, "search must complete for a valid ablation"
        results[use_ac3] = r

    u_off = results[False].itinerary.utility
    u_on = results[True].itinerary.utility
    assert abs(u_on - u_off) < 1e-6, (
        f"AC-3 changed the optimum ({u_on} vs {u_off}). Arc consistency is only "
        f"sound relative to the constraints the search enforces; the propagator "
        f"and the scheduler have diverged.")

    assert results[True].diagnostics.domain_reduction_pct > 0, \
        "AC-3 should prune something on this instance"
    assert results[True].diagnostics.search_nodes <= results[False].diagnostics.search_nodes, \
        "AC-3 should not increase the number of search nodes"
    print(f"  AC-3 ablation: identical optimum {u_on:.3f}, "
          f"{results[True].diagnostics.domain_reduction_pct:.1f}% pruned, "
          f"{results[False].diagnostics.search_nodes:,} -> "
          f"{results[True].diagnostics.search_nodes:,} nodes")


def test_max_leg_enforced_by_scheduler():
    """The scheduler must reject a leg longer than the configured limit."""
    near = cluster(2)
    far = cand(50, lat=9.66, lon=80.02)          # Jaffna
    cs = near + [far]
    cfg = SolverConfig(max_slots_per_day=2, min_stops_per_day=1,
                       max_leg_minutes=30.0, use_ac3=False)
    res = solve("r", cs, days=1, poi_budget=50000,
                pref=flat_scores(cs), sust=flat_scores(cs), w_pref=0.5,
                travel_edges={}, cfg=cfg)
    if res.feasible:
        for day in res.itinerary.days:
            for s in day.stops[1:]:
                assert s.travel_from_prev_min <= 30.0 + 1e-6, \
                    f"scheduler admitted a {s.travel_from_prev_min:.0f} min leg"
    print("  scheduler: no leg exceeds max_leg_minutes")


def test_day_coherence_limits_intra_day_travel():
    """
    Every pair of stops on a day must be mutually close, not just consecutive
    ones. Without this the solver produced Galle -> Kalutara -> Colombo: each
    leg individually legal, but 3.5 hours of driving across three districts.
    """
    near = cluster(4, spread=0.01)
    far = cand(70, lat=6.03, lon=80.21)          # Galle, hours from Kandy
    cs = near + [far]
    cfg = SolverConfig(max_slots_per_day=3, min_stops_per_day=2,
                       max_leg_minutes=300.0, max_intra_day_minutes=45.0,
                       use_ac3=False)
    res = solve("r", cs, days=1, poi_budget=50000,
                pref=flat_scores(cs), sust=flat_scores(cs), w_pref=0.5,
                travel_edges={}, cfg=cfg)
    assert res.feasible, res.conflict

    from solver.csp import TravelModel
    tm = TravelModel(cs, {})
    for day in res.itinerary.days:
        ids = [s.poi_id for s in day.stops]
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                mins, _ = tm.minutes(a, b)
                assert mins <= 45.0 + 1e-6, (
                    f"day {day.day} pairs {a}/{b} are {mins:.0f} min apart, "
                    f"above the 45 min day-coherence limit")
    print("  day coherence: all same-day pairs within the configured limit")


def test_ac3_and_cap_reductions_are_reported_separately():
    """
    Domain capping is a search heuristic, not arc consistency. Folding its
    effect into domain_reduction_pct reported 73-83% as "AC-3 pruning" when the
    true figure was 0% - a number that would have been wrong in the evaluation
    table.
    """
    cs = cluster(60, spread=0.05)
    pref = {c.poi_id: (i % 7) / 7 for i, c in enumerate(cs)}
    cfg = SolverConfig(max_slots_per_day=2, min_stops_per_day=2,
                       max_domain_per_slot=10, time_budget_s=3.0)
    res = solve("r", cs, days=2, poi_budget=50000, pref=pref,
                sust=flat_scores(cs), w_pref=0.5, travel_edges={}, cfg=cfg)
    d = res.diagnostics
    assert d.domain_capped_to == 10
    assert d.domain_size_capped > 0
    assert d.domain_size_capped < d.domain_size_after, "the cap should shrink domains"
    assert d.cap_reduction_pct > 0, "cap reduction must be reported"
    # AC-3's figure must describe AC-3 alone
    expected = round(100.0 * (d.domain_size_before - d.domain_size_after)
                     / d.domain_size_before, 2)
    assert abs(d.domain_reduction_pct - expected) < 1e-6, \
        "domain_reduction_pct must measure arc consistency only"
    print(f"  reporting: AC-3 {d.domain_reduction_pct:.1f}%, "
          f"cap a further {d.cap_reduction_pct:.1f}% (kept separate)")


def test_mmr_reduces_utility_without_widening_selection():
    """
    Records a measured negative result.

    MMR re-ranking penalises choosing a place similar to one already selected.
    It was added to widen catalogue coverage and does not achieve that, because
    coverage is governed by how broad the retrieved candidate pool is rather
    than by the rule used to select within it. The parameter is kept so the
    ablation is reproducible, and defaults to 0.0.
    """
    import numpy as np
    from evaluation.metrics import diversity

    rng = np.random.default_rng(3)
    districts = ["Kandy", "Galle", "Matale", "Colombo"]
    cats = ["heritage", "wildlife", "nature", "beach"]
    pool = [cand(i, lat=float(rng.uniform(6.0, 9.5)),
                 lon=float(rng.uniform(79.9, 81.6)), fee=float(rng.choice([0, 1000])),
                 district=str(rng.choice(districts)),
                 category=str(rng.choice(cats))) for i in range(80)]
    pref = {c.poi_id: float(rng.random()) for c in pool}
    sust = {c.poi_id: 0.5 for c in pool}

    results = {}
    for lam in (0.0, 0.4):
        plans, utils = [], []
        for q in range(10):
            # The recorded finding was measured with the place-only objective;
            # travel and journey costs are switched off to reproduce it. Whether
            # it still holds under the journey objective must be re-measured on
            # real data before the paper text relies on it.
            cfg = SolverConfig(max_slots_per_day=2, min_stops_per_day=2,
                               max_domain_per_slot=30, time_budget_s=1.5,
                               mmr_lambda=lam, travel_cost_per_hour=0.0,
                               journey_in_objective=False)
            r = solve(f"q{q}", pool, 2 + q % 3, 15000.0, pref, sust, 0.5, {}, cfg)
            if r.feasible:
                plans.append([[s.poi_id for s in d.stops] for d in r.itinerary.days])
                utils.append(r.itinerary.utility)
        results[lam] = (diversity(plans, 80).catalogue_coverage_pct,
                        sum(utils) / max(len(utils), 1))

    cov0, u0 = results[0.0]
    cov4, u4 = results[0.4]
    assert cov4 <= cov0 + 5.0, (
        "MMR appears to widen coverage substantially; if so this recorded "
        "finding is out of date and the paper text must be revised")
    assert u4 <= u0 + 1e-6, "MMR must not increase utility; it only subtracts"
    print(f"  MMR ablation: coverage {cov0:.1f}% -> {cov4:.1f}%, "
          f"utility {u0:.2f} -> {u4:.2f} (no coverage gain, utility cost)")


if __name__ == "__main__":
    print("Travora - solver tests\n" + "-" * 62)
    for fn in [test_ac3_is_sound_against_brute_force,
               test_ac3_prunes_unreachable_candidates,
               test_ac3_wipeout_reports_infeasible,
               test_solution_passes_independent_verification,
               test_budget_is_respected_exactly,
               test_imputed_hours_are_not_enforced,
               test_higher_utility_candidates_are_preferred,
               test_weights_shift_the_selection,
               test_travel_provenance_measured_over_itinerary_legs,
               test_missing_travel_edges_are_flagged_not_invented,
               test_diagnostics_report_search_bounds,
               test_too_few_candidates_reports_conflict,
               test_ac3_never_changes_the_optimum,
               test_max_leg_enforced_by_scheduler,
               test_day_coherence_limits_intra_day_travel,
               test_ac3_and_cap_reductions_are_reported_separately,
               test_mmr_reduces_utility_without_widening_selection]:
        fn()
    print("-" * 62 + "\nAll tests passed.")


# ------------------------------------------------------- alternatives
def _varied(cs):
    return {c.poi_id: round(0.95 - 0.03 * i, 3) for i, c in enumerate(cs)}


def test_alternatives_respect_overlap_and_quality():
    cs = cluster(14)
    pref, sust = _varied(cs), flat_scores(cs, 0.5)
    cfg = SolverConfig(max_slots_per_day=3, min_stops_per_day=2, time_budget_s=5.0)
    res = solve("alt", cs, 2, 1_000_000, pref, sust, 0.7, {}, cfg,
                alternatives=2, max_shared_frac=0.5, min_quality=0.8)
    assert res.feasible
    assert res.alternatives, "a 14-place pool should admit a distinct near-optimal plan"

    plans = [{s.poi_id for s in res.itinerary.stops}]
    for alt in res.alternatives:
        stops = {s.poi_id for s in alt.stops}
        for prev in plans:
            assert len(stops & prev) <= int(0.5 * len(prev)), "diversity cut violated"
        assert alt.utility >= 0.8 * res.itinerary.utility, "alternative below quality floor"
        assert alt.utility <= res.itinerary.utility + 1e-9, "alternative beat the optimum"
        plans.append(stops)


def test_no_alternatives_by_default_or_above_quality_floor():
    cs = cluster(14)
    pref, sust = _varied(cs), flat_scores(cs, 0.5)
    cfg = SolverConfig(max_slots_per_day=3, min_stops_per_day=2, time_budget_s=5.0)
    assert solve("a", cs, 2, 1_000_000, pref, sust, 0.7, {}, cfg).alternatives == []
    strict = solve("b", cs, 2, 1_000_000, pref, sust, 0.7, {}, cfg,
                   alternatives=2, min_quality=1.01)
    assert strict.feasible and strict.alternatives == []
