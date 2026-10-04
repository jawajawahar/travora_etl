"""
Evaluation harness (Phase 8) and anti-regression gates (Phase 6).

Both read from one execution path: run a query set through a system, verify
every plan independently, and aggregate. Splitting them would mean two
implementations of "run the queries" that could drift apart.

Anti-regression gates
---------------------
  T1  provenance             every recommended id exists in the graph
  T2  budget perturbation    +/-20% budget changes the itinerary
  T3  interest sensitivity   different interests give different itineraries
  T4  catalogue coverage     distinct POIs used / POIs available
  T5  Gini index             concentration of recommendations
  T6  distinct-plan rate     unique outputs across distinct queries
  T8  fake-id injection      an invented id is rejected by the contract
  T9  seed reproducibility   same query and seed give the same plan
  T10 district reach         districts appearing across the query set

T4 and T5 are also paper results: they are the quantitative form of the
supervisor's finding that the old system repeated itself.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from retrieval.models import Candidate, TripRequest, Category, ScoreVector
from retrieval.service import CandidateRetriever
from solver.models import SolverConfig
from solver.search import solve
from agents.preference import PreferenceAgent
from agents.sustainability import SustainabilityAgent
from .querysets import Query, Plan, rule_based, greedy_eco, single_agent_llm
from .metrics import (verify, Verdict, diversity, DiversityReport, jaccard,
                      plans_differ)

log = logging.getLogger(__name__)

SYSTEMS = ("rule_based", "single_agent_llm", "greedy_eco", "travora")


@dataclass
class RunRecord:
    query_id: str
    system: str
    plan: Plan
    verdict: Verdict
    seconds: float
    claimed_feasible: bool = True
    s_pref: float = 0.0
    s_sust: float = 0.0
    utility: float = 0.0
    ac3_reduction_pct: float = 0.0
    search_nodes: int = 0
    optimal: bool = False


@dataclass
class SystemSummary:
    system: str
    runs: int = 0
    csr_pct: float = 0.0
    false_feasible_pct: float = 0.0
    mean_stops: float = 0.0
    mean_cost: float = 0.0
    mean_pref: float = 0.0
    mean_sust: float = 0.0
    mean_utility: float = 0.0
    p50_s: float = 0.0
    p95_s: float = 0.0
    ac3_reduction_pct: float = 0.0
    mean_nodes: float = 0.0
    optimal_pct: float = 0.0
    violations: dict[str, float] = field(default_factory=dict)
    diversity: Optional[DiversityReport] = None


# ---------------------------------------------------------------------------
class Harness:
    """Runs query sets through each system against one shared candidate cache."""

    def __init__(self, session, completion: Optional[Callable] = None,
                 solver_config: Optional[SolverConfig] = None,
                 catalogue_size: int = 0):
        self.retriever = CandidateRetriever(session)
        self.completion = completion
        self.cfg = solver_config or SolverConfig()
        self.sust_agent = SustainabilityAgent()
        self.catalogue_size = catalogue_size or self._count_pois(session)
        self._cache: dict[tuple, tuple] = {}
        # Preference scores are cached per (candidate pool, interests). The
        # query set has 5 repeats per stratum but only 36 distinct pools, so
        # without this the model is called five times for identical input -
        # a 5x waste of a limited daily token quota.
        self._pref_cache: dict[tuple, dict[str, float]] = {}
        self.llm_calls = 0
        self.llm_cache_hits = 0

    @staticmethod
    def _count_pois(session) -> int:
        try:
            rec = session.run("MATCH (p:POI) RETURN count(p) AS n").single()
            return int(rec["n"])
        except Exception:                               # noqa: BLE001
            return 0

    def candidates_for(self, req: TripRequest):
        """Retrieval is cached: it is identical across systems for a query, and
        re-running it would make timing comparisons meaningless."""
        key = (req.days, req.budget_lkr, tuple(i.value for i in req.interests))
        if key not in self._cache:
            cset = self.retriever.retrieve(req)
            edges = self.retriever.travel_times(cset)
            self._cache[key] = (cset.candidates, edges)
        return self._cache[key]

    def preferences_for(self, req: TripRequest,
                        candidates: list[Candidate]) -> dict[str, float]:
        """Score once per distinct (pool, interests) pair, then reuse."""
        key = (tuple(sorted(c.poi_id for c in candidates)),
               tuple(sorted(i.value for i in req.interests)))
        if key in self._pref_cache:
            self.llm_cache_hits += 1
            return self._pref_cache[key]

        agent = PreferenceAgent(complete=self.completion)
        scores = agent.score(candidates, req.interests, days=req.days,
                             budget_lkr=req.budget_lkr,
                             party_size=req.party_size).scores
        if getattr(agent, "used_llm", False):
            self.llm_calls += 1
        self._pref_cache[key] = scores
        return scores

    # -- one run -----------------------------------------------------------
    def run_one(self, q: Query, system: str, seed: int = 42) -> RunRecord:
        req = q.to_request(seed)
        candidates, edges = self.candidates_for(req)
        by_id = {c.poi_id: c for c in candidates}
        pref: dict[str, float] = {}
        sust = self.sust_agent.score(candidates).scores if candidates else {}

        t0 = time.perf_counter()
        claimed = True
        extra = {}

        if not candidates:
            plan: Plan = [[] for _ in range(req.days)]
        elif system == "rule_based":
            plan = rule_based(candidates, req, self.cfg.max_slots_per_day)
        elif system == "greedy_eco":
            plan = greedy_eco(candidates, req, sust, self.cfg.max_slots_per_day)
        elif system == "single_agent_llm":
            plan = single_agent_llm(candidates, req, self.completion,
                                    self.cfg.max_slots_per_day)
        elif system == "travora":
            pref = self.preferences_for(req, candidates)
            res = solve(q.query_id, candidates, req.days, req.poi_budget_lkr,
                        pref, sust, req.w_pref, edges, self.cfg)
            claimed = res.feasible
            if res.feasible:
                plan = [[s.poi_id for s in d.stops] for d in res.itinerary.days]
                extra = {"s_pref": res.itinerary.s_pref,
                         "s_sust": res.itinerary.s_sust,
                         "utility": res.itinerary.utility,
                         "optimal": res.itinerary.optimal}
            else:
                plan = [[] for _ in range(req.days)]
            extra["ac3_reduction_pct"] = res.diagnostics.domain_reduction_pct
            extra["search_nodes"] = res.diagnostics.search_nodes
        else:
            raise ValueError(f"unknown system {system}")

        seconds = time.perf_counter() - t0
        verdict = verify(plan, req, by_id, self.cfg.max_slots_per_day, travel=edges)

        flat = [p for d in plan for p in d]
        if not extra.get("s_pref") and flat and pref:
            extra["s_pref"] = round(
                sum(pref.get(p, 0.0) for p in flat) / len(flat), 4)
        if not extra.get("s_sust") and flat and sust:
            extra["s_sust"] = round(
                sum(sust.get(p, 0.0) for p in flat) / len(flat), 4)

        return RunRecord(q.query_id, system, plan, verdict, seconds,
                         claimed_feasible=claimed, **extra)

    # -- many runs ---------------------------------------------------------
    def run_system(self, queries: list[Query], system: str) -> list[RunRecord]:
        out = []
        for i, q in enumerate(queries, 1):
            try:
                out.append(self.run_one(q, system))
            except Exception as e:                      # noqa: BLE001
                log.error("%s failed on %s: %s", system, q.query_id, str(e)[:150])
            if i % 20 == 0:
                log.info("  %s: %d/%d", system, i, len(queries))
        return out

    @staticmethod
    def summarise(system: str, records: list[RunRecord],
                  catalogue_size: int,
                  district_of: dict[str, str] | None = None) -> SystemSummary:
        if not records:
            return SystemSummary(system=system)

        n = len(records)
        feasible = [r for r in records if r.verdict.feasible]
        times = sorted(r.seconds for r in records)

        def pct(p):
            return times[min(int(round(p / 100 * (len(times) - 1))), len(times) - 1)]

        # A system that CLAIMED feasibility but failed independent verification.
        # This is what separates fluent generation from a guarantee.
        false_claims = sum(1 for r in records
                           if r.claimed_feasible and not r.verdict.feasible)

        viol: dict[str, float] = {}
        for r in records:
            for k, v in r.verdict.violations.items():
                if v:
                    viol[k] = viol.get(k, 0.0) + 1
        viol = {k: round(100.0 * v / n, 1) for k, v in sorted(
            viol.items(), key=lambda kv: -kv[1])}

        with_stops = [r for r in records if r.verdict.stops]
        return SystemSummary(
            system=system, runs=n,
            csr_pct=round(100.0 * len(feasible) / n, 1),
            false_feasible_pct=round(100.0 * false_claims / n, 1),
            mean_stops=round(sum(r.verdict.stops for r in records) / n, 2),
            mean_cost=round(sum(r.verdict.total_cost for r in records) / n, 0),
            mean_pref=round(sum(r.s_pref for r in with_stops) / max(len(with_stops), 1), 4),
            mean_sust=round(sum(r.s_sust for r in with_stops) / max(len(with_stops), 1), 4),
            mean_utility=round(sum(r.utility for r in records) / n, 4),
            p50_s=round(pct(50), 3), p95_s=round(pct(95), 3),
            ac3_reduction_pct=round(
                sum(r.ac3_reduction_pct for r in records) / n, 2),
            mean_nodes=round(sum(r.search_nodes for r in records) / n, 0),
            optimal_pct=round(100.0 * sum(1 for r in records if r.optimal) / n, 1),
            violations=viol,
            diversity=diversity([r.plan for r in records], catalogue_size,
                                district_of),
        )


# ---------------------------------------------------------------------------
# Anti-regression gates
# ---------------------------------------------------------------------------
@dataclass
class Gate:
    gate_id: str
    name: str
    observed: str
    threshold: str
    passed: bool


def run_gates(h: Harness, queries: list[Query], system: str = "travora",
              district_of: dict[str, str] | None = None,
              coverage_min: float = 40.0, gini_max: float = 0.60,
              distinct_min: float = 90.0, perturb_min: float = 80.0,
              overlap_max: float = 30.0) -> tuple[list[Gate], SystemSummary]:
    records = h.run_system(queries, system)
    summ = Harness.summarise(system, records, h.catalogue_size, district_of)
    d = summ.diversity
    gates: list[Gate] = []

    def add(gid, name, obs, thr, ok):
        gates.append(Gate(gid, name, obs, thr, ok))

    # T1 provenance
    unknown = sum(r.verdict.violations.get("unknown_poi", 0) for r in records)
    add("T1", "Provenance: every id exists in the graph",
        f"{unknown} unknown ids", "0", unknown == 0)

    # T4/T5/T6 diversity
    add("T4", "Catalogue coverage", f"{d.catalogue_coverage_pct:.1f}%",
        f">= {coverage_min}%", d.catalogue_coverage_pct >= coverage_min)
    add("T5", "Gini index of recommendation frequency", f"{d.gini:.3f}",
        f"<= {gini_max}", d.gini <= gini_max)
    add("T6", "Distinct-plan rate", f"{d.distinct_plan_pct:.1f}%",
        f">= {distinct_min}%", d.distinct_plan_pct >= distinct_min)

    # T2 budget perturbation
    sample = queries[:min(20, len(queries))]
    changed = 0
    for q in sample:
        base = h.run_one(q, system)
        hi = Query(q.query_id + "-hi", "high" if q.budget_band != "high" else "low",
                   q.duration_band, q.profile, q.repeat, q.w_pref)
        alt = h.run_one(hi, system)
        if plans_differ(base.plan, alt.plan):
            changed += 1
    pct_changed = 100.0 * changed / max(len(sample), 1)
    add("T2", "Budget perturbation changes the itinerary",
        f"{pct_changed:.0f}%", f">= {perturb_min}%", pct_changed >= perturb_min)

    # T3 interest sensitivity
    a = h.run_one(Query("t3-culture", "medium", "medium", "culture", 0), system)
    b = h.run_one(Query("t3-beach", "medium", "medium", "beach", 0), system)
    ov = 100.0 * jaccard(a.plan, b.plan)
    add("T3", "Interest sensitivity (culture vs beach overlap)",
        f"{ov:.0f}%", f"<= {overlap_max}%", ov <= overlap_max)

    # T9 seed reproducibility
    q0 = queries[0]
    r1 = h.run_one(q0, system, seed=7)
    r2 = h.run_one(q0, system, seed=7)
    same = not plans_differ(r1.plan, r2.plan)
    add("T9", "Seed reproducibility", "identical" if same else "differs",
        "identical", same)

    # T10 district reach
    add("T10", "Districts reached across the query set",
        f"{d.districts_covered}", ">= 10", d.districts_covered >= 10)

    return gates, summ


def gate_t8_fake_id() -> Gate:
    """
    T8: an agent returning an id that was never retrieved must be rejected.
    Runs without a database because it tests the contract, not the data.
    """
    ok = False
    try:
        ScoreVector(agent="preference", candidate_ids=["node/1"],
                    scores={"node/1": 0.8, "sigiriya": 0.99})
    except Exception:                                   # noqa: BLE001
        ok = True
    return Gate("T8", "Fake-id injection is rejected by the contract",
                "rejected" if ok else "ACCEPTED", "rejected", ok)
