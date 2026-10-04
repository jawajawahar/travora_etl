"""
The orchestrated pipeline: retrieval -> agents -> solver.

Flow
----
    Retrieval Agent      graph query          -> CandidateSet
    Preference Agent     LLM, ID-validated    -> ScoreVector
    Sustainability Agent pure arithmetic      -> ScoreVector
    Decision Agent       AC-3 + branch/bound  -> Itinerary or Conflict

Only the Preference Agent calls a language model. It scores; it never selects.
Every place in the output traces to a row retrieved from Neo4j, and the typed
blackboard is what carries values between the agents instead of natural
language, so ambiguity cannot propagate.

The audit trail records what each agent contributed and how the decision was
reached, which is what the transparency view renders and what the evaluation
protocol reads.
"""
from __future__ import annotations
import logging
import time
from typing import Optional

from pydantic import BaseModel, Field

from retrieval.models import TripRequest, CandidateSet, ScoreVector
from retrieval.service import CandidateRetriever
from solver.models import SolverConfig, SolverResult
from solver.search import solve
from .preference import PreferenceAgent, PROMPT_VERSION
from .sustainability import SustainabilityAgent

log = logging.getLogger(__name__)


class AuditTrail(BaseModel):
    """What each agent did. Rendered by the transparency view."""
    candidates_retrieved: int = 0
    districts_covered: int = 0
    query_widened: bool = False
    widening_reason: Optional[str] = None

    preference_used_llm: bool = False
    preference_prompt_version: str = PROMPT_VERSION
    preference_failures: list[str] = Field(default_factory=list)
    preference_mean: float = 0.0
    sustainability_mean: float = 0.0

    ac3_domain_reduction_pct: float = 0.0
    ac3_arc_revisions: int = 0
    domain_capped_to: int = 0
    search_nodes: int = 0
    optimal: bool = False

    retrieval_ms: float = 0.0
    scoring_ms: float = 0.0
    solve_ms: float = 0.0
    total_ms: float = 0.0

    w_pref: float = 0.5
    w_sust: float = 0.5


class PipelineResult(BaseModel):
    request_id: str
    solver: SolverResult
    audit: AuditTrail
    preference_scores: dict[str, float] = Field(default_factory=dict)
    sustainability_scores: dict[str, float] = Field(default_factory=dict)


class TravoraPipeline:
    def __init__(self, session, preference: Optional[PreferenceAgent] = None,
                 sustainability: Optional[SustainabilityAgent] = None,
                 solver_config: Optional[SolverConfig] = None):
        self.retriever = CandidateRetriever(session)
        self.preference = preference or PreferenceAgent()
        self.sustainability = sustainability or SustainabilityAgent()
        self.solver_config = solver_config or SolverConfig()

    def run(self, req: TripRequest, alternatives: int = 0) -> PipelineResult:
        t0 = time.perf_counter()

        # 1. Retrieval
        cset: CandidateSet = self.retriever.retrieve(req)
        travel_edges = self.retriever.travel_times(cset)
        t_retrieved = time.perf_counter()

        audit = AuditTrail(
            candidates_retrieved=len(cset),
            districts_covered=cset.districts_covered,
            query_widened=cset.query_widened,
            widening_reason=cset.widening_reason,
            retrieval_ms=round((t_retrieved - t0) * 1000, 1),
            w_pref=req.w_pref, w_sust=req.w_sust,
        )

        if not cset.candidates:
            from solver.models import Conflict, ConflictType
            return PipelineResult(
                request_id=cset.request_id, audit=audit,
                solver=SolverResult(
                    feasible=False,
                    conflict=Conflict(
                        conflict=ConflictType.TOO_FEW_CANDIDATES,
                        detail="Retrieval returned no candidates.",
                        suggestion="Widen the interests or the search radius.")))

        # 2. Scoring. Both agents receive the SAME candidate list and may only
        #    return scores for exactly those ids.
        pref_vec: ScoreVector = self.preference.score(
            cset.candidates, req.interests, days=req.days,
            budget_lkr=req.budget_lkr, party_size=req.party_size)
        sust_vec: ScoreVector = self.sustainability.score(cset.candidates)

        # Belt and braces: the ScoreVector validator already enforces this, but
        # a mismatch here would mean an agent scored a different pool than the
        # solver will choose from, which must never pass silently.
        assert set(pref_vec.scores) == cset.ids, "preference scored a different pool"
        assert set(sust_vec.scores) == cset.ids, "sustainability scored a different pool"

        t_scored = time.perf_counter()
        audit.scoring_ms = round((t_scored - t_retrieved) * 1000, 1)
        audit.preference_used_llm = getattr(self.preference, "used_llm", False)
        audit.preference_failures = list(getattr(self.preference, "failures", []))
        audit.preference_mean = round(
            sum(pref_vec.scores.values()) / len(pref_vec.scores), 4)
        audit.sustainability_mean = round(
            sum(sust_vec.scores.values()) / len(sust_vec.scores), 4)

        # 3. Decision. Deterministic; no model involved.
        result = solve(cset.request_id, cset.candidates, req.days,
                       req.poi_budget_lkr, pref_vec.scores, sust_vec.scores,
                       req.w_pref, travel_edges, self.solver_config,
                       alternatives=alternatives,
                       start=(req.start_lat, req.start_lon),
                       party_size=req.party_size)

        d = result.diagnostics
        audit.ac3_domain_reduction_pct = d.domain_reduction_pct
        audit.ac3_arc_revisions = d.arc_revisions
        audit.domain_capped_to = d.domain_capped_to
        audit.search_nodes = d.search_nodes
        audit.optimal = result.itinerary.optimal if result.itinerary else False
        audit.solve_ms = round((time.perf_counter() - t_scored) * 1000, 1)
        audit.total_ms = round((time.perf_counter() - t0) * 1000, 1)

        log.info("Pipeline: %d candidates -> %s in %.0f ms (llm=%s)",
                 len(cset),
                 f"{len(result.itinerary.stops)} stops" if result.feasible
                 else "no feasible itinerary",
                 audit.total_ms, audit.preference_used_llm)

        return PipelineResult(request_id=cset.request_id, solver=result,
                              audit=audit,
                              preference_scores=pref_vec.scores,
                              sustainability_scores=sust_vec.scores)
