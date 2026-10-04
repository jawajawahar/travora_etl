"""
Query set and baseline planners.

The query set is stratified rather than sampled at random, so performance can be
reported by condition and any interaction between trip difficulty and system
becomes visible instead of averaging away.

Baselines
---------
  rule_based        sequential filters then rank by prominence. Represents
                    current practice and is the user-study control.
  greedy_eco        always takes the most sustainable option. Upper bound on
                    sustainability; shows the cost of ignoring the traveller.
  single_agent_llm  one language model prompted once for a whole itinerary, no
                    CSP layer. THE most important baseline: without it a
                    reviewer can object that Travora's performance comes from
                    the language model rather than the architecture, and there
                    is no evidence to answer with.

Every planner returns the same shape - a list of days, each a list of poi_ids -
so the independent verifier can assess them identically.
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import product
from typing import Callable, Optional

from retrieval.models import Candidate, Category, TripRequest

log = logging.getLogger(__name__)

Plan = list[list[str]]          # [day][poi_id]


# ---------------------------------------------------------------------------
# Query set
# ---------------------------------------------------------------------------
BUDGETS = {"low": 40_000.0, "medium": 100_000.0, "high": 250_000.0}
DURATIONS = {"short": 3, "medium": 7, "long": 12}
PROFILES = {
    "culture": [Category.HERITAGE, Category.CULTURAL],
    "wildlife": [Category.WILDLIFE, Category.NATURE],
    "beach": [Category.BEACH],
    "mixed": [Category.HERITAGE, Category.WILDLIFE, Category.BEACH],
}
START = (7.2906, 80.6337)       # Kandy, roughly central


@dataclass(frozen=True)
class Query:
    query_id: str
    budget_band: str
    duration_band: str
    profile: str
    repeat: int
    w_pref: float = 0.5

    def to_request(self, seed: int = 42) -> TripRequest:
        return TripRequest(
            start_date=date.today() + timedelta(days=30),
            days=DURATIONS[self.duration_band],
            budget_lkr=BUDGETS[self.budget_band],
            party_size=2,
            interests=PROFILES[self.profile],
            start_lat=START[0], start_lon=START[1],
            w_pref=self.w_pref, seed=seed)


def build_query_set(repeats: int = 5) -> list[Query]:
    """
    3 budgets x 3 durations x 4 profiles x `repeats` = 180 queries by default.

    Repeats exist because language model output is not deterministic even at low
    temperature; a single run per cell would confound model variance with
    system differences.
    """
    out = []
    for b, d, p in product(BUDGETS, DURATIONS, PROFILES):
        for r in range(repeats):
            out.append(Query(f"{b}-{d}-{p}-{r}", b, d, p, r))
    return out


def weight_sweep_queries(levels=(0.0, 0.25, 0.5, 0.75, 1.0)) -> list[Query]:
    """One query per cell at each weight level, for the trade-off frontier."""
    out = []
    for w in levels:
        for b, d, p in product(BUDGETS, DURATIONS, PROFILES):
            # "-0" is the repeat suffix summarise() strips to count distinct queries.
            out.append(Query(f"w{w}-{b}-{d}-{p}-0", b, d, p, 0, w_pref=w))
    return out


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def _greedy_fill(ordered: list[Candidate], days: int, slots: int,
                 poi_budget: float) -> Plan:
    """
    Take candidates in the given order, filling days while the budget lasts.

    No travel-time or opening-hour reasoning: that is precisely what
    distinguishes a ranking baseline from a planner, and the independent
    verifier will record the resulting violations.
    """
    plan: Plan = [[] for _ in range(days)]
    spent = 0.0
    idx = 0
    for d in range(days):
        for _ in range(slots):
            while idx < len(ordered):
                c = ordered[idx]
                idx += 1
                if spent + c.entry_fee_lkr <= poi_budget:
                    plan[d].append(c.poi_id)
                    spent += c.entry_fee_lkr
                    break
            else:
                break
    return plan


def rule_based(candidates: list[Candidate], req: TripRequest,
               slots: int = 3, **_) -> Plan:
    """Filter by interest, rank by prominence. Current practice."""
    wanted = {i.value for i in req.interests}
    pool = [c for c in candidates if c.category.value in wanted] or list(candidates)
    pool = sorted(pool, key=lambda c: -c.prominence)
    return _greedy_fill(pool, req.days, slots, req.poi_budget_lkr)


def greedy_eco(candidates: list[Candidate], req: TripRequest,
               sust: dict[str, float], slots: int = 3, **_) -> Plan:
    """Always take the most sustainable option, ignoring preference."""
    pool = sorted(candidates, key=lambda c: -sust.get(c.poi_id, 0.0))
    return _greedy_fill(pool, req.days, slots, req.poi_budget_lkr)


SINGLE_AGENT_SYSTEM = """You are a travel planner for Sri Lanka.

You will receive a trip request and a JSON list of candidate destinations, each
with an opaque "id".

Produce a day-by-day itinerary. Respond with JSON only, no prose:
  {"days": [["<id>", "<id>"], ["<id>"]]}

Rules:
  1. Use only ids from the candidate list.
  2. Respect the total budget across all entry fees.
  3. Keep each day's stops geographically sensible.
  4. Return exactly the requested number of days."""


def single_agent_llm(candidates: list[Candidate], req: TripRequest,
                     complete: Optional[Callable[[str, str], str]] = None,
                     slots: int = 3, **_) -> Plan:
    """
    One model call for the whole itinerary, with no constraint layer.

    This is the condition TravelPlanner reports frontier models failing at a
    0.6% success rate. Its output is deliberately NOT repaired: invented ids and
    constraint violations are preserved so the verifier can measure them, which
    is the entire point of including it.
    """
    if complete is None:
        return [[] for _ in range(req.days)]

    payload = [{"id": c.poi_id, "category": c.category.value,
                "entry_fee_lkr": round(c.entry_fee_lkr),
                "visit_minutes": c.typical_dwell_min,
                "lat": round(c.lat, 3), "lon": round(c.lon, 3)}
               for c in candidates[:60]]
    user = (f"Trip: {req.days} days, budget LKR {round(req.budget_lkr)}, "
            f"party {req.party_size}, interests "
            f"{', '.join(i.value for i in req.interests)}. "
            f"Up to {slots} stops per day.\n\n"
            f"Candidates:\n{json.dumps(payload, separators=(',', ':'))}")

    try:
        raw = complete(SINGLE_AGENT_SYSTEM, user)
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end + 1])
        days = data.get("days", [])
        plan = [[str(x) for x in day] for day in days][:req.days]
        while len(plan) < req.days:
            plan.append([])
        return plan
    except Exception as e:                              # noqa: BLE001
        log.warning("single_agent_llm produced no usable plan: %s", str(e)[:150])
        return [[] for _ in range(req.days)]
