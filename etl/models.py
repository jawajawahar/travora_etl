"""
Solver models: the itinerary, the constraint specification, and the result.

Design note on constraint hardness
----------------------------------
Only 2.9% of opening hours and 0.4% of entry fees in the dataset come from the
source; the rest are category defaults produced by the ETL. A constraint
enforced against an imputed value rejects itineraries on the basis of fiction,
and would inflate the reported constraint-satisfaction rate while guaranteeing
nothing. Every constraint therefore declares whether it is HARD (checked against
real data) or SOFT (checked against an estimate, recorded but not enforced).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Hardness(str, Enum):
    HARD = "hard"      # violation makes the itinerary infeasible
    SOFT = "soft"      # violation is recorded and penalised, not rejected


class ConflictType(str, Enum):
    BUDGET = "budget_exceeded"
    NO_FEASIBLE_ASSIGNMENT = "no_feasible_assignment"
    DOMAIN_WIPEOUT = "domain_wipeout"
    TOO_FEW_CANDIDATES = "too_few_candidates"
    TIME_EXHAUSTED = "day_time_exhausted"


# ---------------------------------------------------------------------------
@dataclass
class SolverConfig:
    """All tunables in one place so they can be swept during evaluation."""
    day_start_min: int = 8 * 60          # 08:00
    day_end_min: int = 18 * 60           # 18:00
    max_slots_per_day: int = 3
    # Two stops minimum. Besides being a more realistic day, this is what lets
    # AC-3 work at all: NULL supports every value in the reachability
    # constraint, so while any slot may be NULL no value can ever lose all
    # support and propagation prunes nothing.
    min_stops_per_day: int = 2
    # A single hop longer than this is rejected. 180 min was permissive enough
    # that arc consistency pruned nothing on real data; a 2-hour leg between two
    # stops in one day is already at the limit of what a traveller will accept,
    # and tightening it makes the reachability constraint actually bite.
    max_leg_minutes: float = 120.0

    # Every pair of stops on the SAME DAY must be within this travel time of
    # each other, not merely consecutive ones. Without it the solver produced
    # days like Galle -> Kalutara -> Colombo: each leg individually legal, but
    # 3.5 hours of driving and three districts in one day. This is also the
    # constraint that gives arc consistency real work on dense candidate sets.
    max_intra_day_minutes: float = 150.0
    buffer_minutes: int = 15             # slack between stops

    # Objective
    district_repeat_penalty: float = 0.05
    category_repeat_penalty: float = 0.03

    # Maximal Marginal Relevance: subtract a penalty for choosing a place
    # similar to one already selected.
    #
    # Implemented and measured, and it does NOT improve catalogue coverage.
    # Coverage turns out to be governed by retrieval breadth rather than by the
    # selection rule: holding the candidate pool fixed, raising the penalty from
    # 0.0 to 0.6 left coverage flat (18.0% -> 16.0%) while mean utility fell
    # 26%. Varying retrieval instead moved coverage from 4.5% (one pool) to
    # 15.0% (eight pools), and HALVING the pool size doubled it (15.0% -> 30.0%)
    # because the planner only ever draws from the top of whatever pool it is
    # given. Default 0.0 on that evidence; the parameter is retained so the
    # ablation can be reproduced.
    mmr_lambda: float = 0.0

    # Search
    time_budget_s: float = 10.0
    max_nodes: int = 200_000

    # Ablation switch. Metric M3 in the evaluation protocol compares search
    # nodes explored with and without arc consistency; set False for the
    # control condition.
    use_ac3: bool = True

    # Cap each slot's domain to the top-K candidates by utility. Exact search
    # over 150 candidates and 15 slots is intractable, so this is a stated
    # approximation rather than a silent one: it is reported in diagnostics and
    # must be disclosed alongside any optimality claim. 0 = uncapped.
    max_domain_per_slot: int = 40

    @property
    def day_active_minutes(self) -> int:
        return self.day_end_min - self.day_start_min


# ---------------------------------------------------------------------------
class ScheduledStop(BaseModel):
    poi_id: str
    name: str
    # Coordinates travel with the stop because later steps need them: lodging is
    # searched near the last stop of each day, and transport mode depends on the
    # distance between consecutive stops. Without these the lodging search ran
    # at (0, 0) and returned nothing for every night.
    lat: float = 0.0
    lon: float = 0.0
    day: int
    slot: int
    arrive_min: int
    depart_min: int
    dwell_min: int
    travel_from_prev_min: float = 0.0
    travel_estimated: bool = True        # True when no stored graph edge existed
    cost_lkr: float
    district: str
    category: str
    utility: float = 0.0
    hours_enforced: bool = False         # False when opening hours were imputed
    fee_enforced: bool = False

    @property
    def arrive_hhmm(self) -> str:
        return f"{self.arrive_min // 60:02d}:{self.arrive_min % 60:02d}"

    @property
    def depart_hhmm(self) -> str:
        return f"{self.depart_min // 60:02d}:{self.depart_min % 60:02d}"


class DayPlan(BaseModel):
    day: int
    stops: list[ScheduledStop] = Field(default_factory=list)

    @property
    def cost_lkr(self) -> float:
        return sum(s.cost_lkr for s in self.stops)

    @property
    def active_minutes(self) -> int:
        if not self.stops:
            return 0
        return self.stops[-1].depart_min - self.stops[0].arrive_min


class Itinerary(BaseModel):
    request_id: str
    days: list[DayPlan] = Field(default_factory=list)
    total_cost_lkr: float = 0.0
    utility: float = 0.0
    s_pref: float = 0.0
    s_sust: float = 0.0
    optimal: bool = False                # False when the search hit its budget

    @property
    def stops(self) -> list[ScheduledStop]:
        return [s for d in self.days for s in d.stops]

    @property
    def poi_ids(self) -> list[str]:
        return [s.poi_id for s in self.stops]

    @property
    def districts(self) -> set[str]:
        return {s.district for s in self.stops}

    def signature(self) -> str:
        """Stable identity of an itinerary, for the distinct-output metric."""
        return "|".join(f"{s.day}:{s.poi_id}" for s in self.stops)


class Conflict(BaseModel):
    """Why no feasible itinerary exists. Returned instead of a fabricated plan."""
    conflict: ConflictType
    detail: str
    binding_constraint: Optional[str] = None
    suggestion: Optional[str] = None


class SolverDiagnostics(BaseModel):
    """Feeds the audit trail, the paper's Table II, and the regression tests."""
    variables: int = 0
    domain_size_before: int = 0
    domain_size_after: int = 0          # after AC-3 only
    domain_size_capped: int = 0         # after the top-K cap, if any
    arc_revisions: int = 0
    wipeout: bool = False
    search_nodes: int = 0
    time_ms: float = 0.0
    hit_node_cap: bool = False
    hit_time_budget: bool = False
    ac3_applied: bool = True
    domain_capped_to: int = 0
    mmr_lambda: float = 0.0
    mmr_penalty_total: float = 0.0
    # Counted over the legs in the FINAL itinerary, not over lookups performed
    # during propagation. AC-3's support check short-circuits, so propagation
    # lookup counts depend on arbitrary set-iteration order and would be a
    # meaningless statistic to report.
    travel_edges_stored: int = 0
    travel_edges_estimated: int = 0
    # Raw propagation lookups, kept for performance profiling only.
    travel_lookups_stored: int = 0
    travel_lookups_estimated: int = 0
    soft_violations: dict[str, int] = Field(default_factory=dict)

    @property
    def domain_reduction_pct(self) -> float:
        """
        Reduction attributable to ARC CONSISTENCY alone.

        Kept strictly separate from the top-K cap. The cap is a search
        heuristic, and folding its effect into this figure would report a
        heuristic's work as a property of AC-3 - a number that would then be
        wrong in the evaluation table.
        """
        if not self.domain_size_before:
            return 0.0
        return round(100.0 * (self.domain_size_before - self.domain_size_after)
                     / self.domain_size_before, 2)

    @property
    def cap_reduction_pct(self) -> float:
        """Further reduction from the top-K cap, reported separately."""
        if not self.domain_size_capped or not self.domain_size_after:
            return 0.0
        return round(100.0 * (self.domain_size_after - self.domain_size_capped)
                     / self.domain_size_after, 2)


class SolverResult(BaseModel):
    feasible: bool
    itinerary: Optional[Itinerary] = None
    conflict: Optional[Conflict] = None
    diagnostics: SolverDiagnostics = Field(default_factory=SolverDiagnostics)
