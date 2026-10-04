"""
Independent verification and diversity metrics.

The verifier deliberately shares no code with the planner. A verifier that
reuses the planner's own constraint logic cannot detect a fault in that logic:
the system marks its own homework and passes unconditionally. Everything here
re-derives feasibility from the raw plan and the raw request.

Diversity metrics answer the supervisor's finding directly. "The itineraries
repeat" becomes catalogue coverage and a Gini index, and those become the
before/after evidence.
"""
from __future__ import annotations
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from retrieval.models import Candidate, TripRequest

Plan = list[list[str]]

EARTH_KM = 6371.0
AVG_KMH = 35.0
DETOUR = 1.35
DAY_START = 8 * 60
DAY_END = 18 * 60
BUFFER = 15
# Each day begins at the traveller's starting point (day 1) or where the
# previous day ended; the first hop of a day may take at most this long.
MAX_TRANSFER = 240


def travel_minutes(a, b) -> float:
    """Minutes between two things with .lat/.lon (candidates or a start point)."""
    from math import radians, sin, cos, asin, sqrt
    la1, lo1, la2, lo2 = map(radians, (a.lat, a.lon, b.lat, b.lon))
    h = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
    km = 2 * EARTH_KM * asin(sqrt(h))
    return (km * DETOUR / AVG_KMH) * 60.0


def _leg(a, b, travel: dict | None) -> tuple[float, bool]:
    """
    Minutes for one leg, and whether a measured time was used.

    Independence is about code, not data. The verifier already reads fees and
    opening hours from the same knowledge graph as the planner; it reads the
    measured travel times there too (1,271 are OSRM road times), and estimates
    only where none was measured. Estimating everything instead made the check
    disagree with a measured road time by up to 32 minutes and reject plans
    that a traveller could in fact complete - a disagreement between two travel
    models, not a planner fault. The lookup below is the verifier's own.
    """
    ida, idb = getattr(a, "poi_id", None), getattr(b, "poi_id", None)
    if travel and ida and idb:
        key = (ida, idb) if ida < idb else (idb, ida)
        if key in travel:
            return float(travel[key]), True
    return travel_minutes(a, b), False


VIOLATIONS = ("unknown_poi", "duplicate_poi", "budget_exceeded",
              "day_overflow", "opening_hours", "empty_plan", "too_many_days",
              "transfer_too_long")


class _Point:
    def __init__(self, lat, lon):
        self.lat, self.lon = lat, lon


@dataclass
class Verdict:
    feasible: bool
    violations: dict[str, int] = field(default_factory=dict)
    total_cost: float = 0.0
    stops: int = 0
    # Journey emissions, kg CO2e per traveller per day, under each factor
    # scenario (low / central / high). Empty for an empty plan.
    emissions: dict[str, float] = field(default_factory=dict)
    # Legs checked against a measured travel time, and against the estimate.
    legs_measured: int = 0
    legs_estimated: int = 0

    @property
    def violated(self) -> list[str]:
        return [k for k, v in self.violations.items() if v]


def verify(plan: Plan, req: TripRequest, by_id: dict[str, Candidate],
           max_slots: int = 3, travel: dict | None = None) -> Verdict:
    """
    Re-derive every constraint from scratch.

    `travel` holds measured travel times keyed by (id, id) with the smaller id
    first, as stored in the knowledge graph. Without it every leg is estimated.
    """
    v = {k: 0 for k in VIOLATIONS}
    flat = [p for day in plan for p in day]

    if not flat:
        v["empty_plan"] = 1
        return Verdict(False, v, 0.0, 0)
    if len(plan) > req.days:
        v["too_many_days"] = 1

    unknown = [p for p in flat if p not in by_id]
    v["unknown_poi"] = len(unknown)
    v["duplicate_poi"] = len(flat) - len(set(flat))

    known = [by_id[p] for p in flat if p in by_id]
    total = sum(c.entry_fee_lkr for c in known)
    if total > req.poi_budget_lkr + 1e-6:
        v["budget_exceeded"] = 1

    measured = estimated = 0

    def leg(a, b) -> float:
        nonlocal measured, estimated
        mins, was_measured = _leg(a, b, travel)
        if was_measured:
            measured += 1
        else:
            estimated += 1
        return mins

    start = getattr(req, "start_lat", None)
    anchor = _Point(req.start_lat, req.start_lon) if start is not None else None
    for day in plan:
        cs = [by_id[p] for p in day if p in by_id]
        if not cs:
            continue
        clock = DAY_START
        if anchor is not None:
            hop = leg(anchor, cs[0])
            if hop > MAX_TRANSFER:
                v["transfer_too_long"] += 1
            clock += hop + BUFFER
        for i, c in enumerate(cs):
            if i:
                clock += leg(cs[i - 1], c) + BUFFER
            arrive = clock
            # Opening hours are enforced only where the source supplied them:
            # 97% of the dataset's hours are category defaults, and rejecting a
            # plan against an invented value would measure nothing.
            if not c.hours_estimated:
                arrive = max(arrive, c.open_min)
                if arrive + c.typical_dwell_min > c.close_min:
                    v["opening_hours"] += 1
            clock = arrive + c.typical_dwell_min
        if clock > DAY_END:
            v["day_overflow"] += 1
        anchor = cs[-1]                 # the next day starts where this one ended

    return Verdict(all(x == 0 for x in v.values()), v, round(total, 2), len(flat),
                   emissions=trip_emissions(plan, req, by_id),
                   legs_measured=measured, legs_estimated=estimated)


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------
def gini(counts: Iterable[float], catalogue_size: int | None = None) -> float:
    """
    Gini index of recommendation frequency across the WHOLE catalogue.

    Computing it over recommended items only is a trap: a system that emits the
    same three POIs fifty times each scores 0.0, because those three are
    perfectly equal among themselves - the precise failure mode the metric
    exists to detect. Padding with the unused catalogue makes the concentration
    visible, so three POIs out of 2,245 scores close to 1.

    Pass catalogue_size to get the meaningful figure; omitting it falls back to
    the recommended-items-only form, which is reported only for comparison.
    """
    xs = [float(c) for c in counts if c > 0]
    if catalogue_size:
        xs = xs + [0.0] * max(0, catalogue_size - len(xs))
    xs.sort()
    n = len(xs)
    if n == 0:
        return 0.0
    total = sum(xs)
    if total == 0:
        return 0.0
    weighted = sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(xs))
    return round(weighted / (n * total), 4)


@dataclass
class DiversityReport:
    plans: int = 0
    total_recommendations: int = 0
    distinct_pois: int = 0
    catalogue_size: int = 0
    catalogue_coverage_pct: float = 0.0
    gini: float = 0.0                 # over the whole catalogue, zeros included
    distinct_plan_pct: float = 0.0
    districts_covered: int = 0
    top5_share_pct: float = 0.0

    def summary(self) -> str:
        return (f"coverage {self.catalogue_coverage_pct:.1f}%, "
                f"Gini {self.gini:.3f}, distinct plans "
                f"{self.distinct_plan_pct:.1f}%, top-5 share "
                f"{self.top5_share_pct:.1f}%")


def diversity(plans: list[Plan], catalogue_size: int,
              district_of: dict[str, str] | None = None,
              distinct_queries: int | None = None,
              query_keys: list[str] | None = None) -> DiversityReport:
    """
    distinct_queries is the number of genuinely different requests. The query
    set repeats each stratum, so measuring distinct plans against all runs caps
    the achievable rate and makes a deterministic system look repetitive when
    it is simply reproducible.

    With query_keys (the stratum of each plan), the distinct-plan rate uses ONE
    plan per distinct request - the first - and counts how many requests got a
    plan no other request got. Counting every run instead let a system that
    answers each repeat differently exceed 100% (the LLM baseline reached 283%),
    rewarding instability rather than diversity. The rate is at most 100%.
    """
    freq: Counter = Counter()
    signatures = set()
    first_by_key: dict[str, str] = {}
    for i, plan in enumerate(plans):
        flat = [p for day in plan for p in day]
        freq.update(flat)
        sig = "|".join(f"{d}:{p}" for d, day in enumerate(plan) for p in day)
        signatures.add(sig)
        if query_keys is not None:
            first_by_key.setdefault(query_keys[i], sig)
    if query_keys is not None:
        signatures = set(first_by_key.values())
        distinct_queries = len(first_by_key)

    total = sum(freq.values())
    top5 = sum(c for _, c in freq.most_common(5))
    districts = ({district_of.get(p) for p in freq if district_of}
                 if district_of else set())

    return DiversityReport(
        plans=len(plans),
        total_recommendations=total,
        distinct_pois=len(freq),
        catalogue_size=catalogue_size,
        catalogue_coverage_pct=round(100.0 * len(freq) / max(catalogue_size, 1), 2),
        gini=gini(freq.values(), catalogue_size),
        distinct_plan_pct=round(
            100.0 * len(signatures) / max(distinct_queries or len(plans), 1), 2),
        districts_covered=len({d for d in districts if d}),
        top5_share_pct=round(100.0 * top5 / max(total, 1), 2),
    )


def jaccard(a: Plan, b: Plan) -> float:
    """Overlap between two plans, for the interest-sensitivity gate."""
    sa = {p for day in a for p in day}
    sb = {p for day in b for p in day}
    if not sa and not sb:
        return 1.0
    return round(len(sa & sb) / max(len(sa | sb), 1), 4)


def plans_differ(a: Plan, b: Plan) -> bool:
    return [list(d) for d in a] != [list(d) for d in b]


# ---------------------------------------------------------------------------
# Journey emissions (independent of the planner)
#
# Reads the same factor TABLE as the planner (config/sustainability.yaml, which
# is data with cited sources) but none of its code. Every system's plan is
# judged by the same rule: each leg, including the transfer from the starting
# point and each morning's move, is a road leg whose mode follows its length.
# The verifier has no rail network, so no system is credited with a train; the
# comparison is like for like, and conservative for Travora, whose app legs may
# use the train.
# ---------------------------------------------------------------------------
_EF_CACHE: dict = {}


def _emission_table() -> dict:
    if not _EF_CACHE:
        import yaml
        from pathlib import Path
        cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "config"
                              / "sustainability.yaml").read_text(encoding="utf-8"))
        _EF_CACHE.update(factors=cfg["transport_emissions"], journey=cfg["journey"])
    return _EF_CACHE


def _mode_for(km: float) -> str:
    if km <= 1.5:
        return "walk"
    if km <= 6.0:
        return "tuktuk"
    if km < 60.0:
        return "bus"
    return "car_shared"


def _kg(mode: str, km: float, party: int, scenario: str) -> float:
    f = _emission_table()["factors"][mode]
    value = f[scenario]
    if f["basis"] == "pkm":
        return km * value
    if "occupancy" in f:
        return km * value / f["occupancy"]
    return km * value * math.ceil(party / f.get("capacity", 4)) / party


def trip_emissions(plan: Plan, req, by_id: dict) -> dict[str, float]:
    """kg CO2e per traveller per day, for each factor scenario."""
    party = max(int(getattr(req, "party_size", 1) or 1), 1)
    days = max(len([d for d in plan if d]), 1)
    anchor = (_Point(req.start_lat, req.start_lon)
              if getattr(req, "start_lat", None) is not None else None)
    kms = []
    for day in plan:
        cs = [by_id[p] for p in day if p in by_id]
        if not cs:
            continue
        prev = anchor
        for c in cs:
            if prev is not None:
                kms.append(_straight_km(prev, c) * DETOUR)
            prev = c
        anchor = cs[-1]
    if not kms:
        return {}
    return {sc: round(sum(_kg(_mode_for(k), k, party, sc) for k in kms) / days, 3)
            for sc in ("low", "central", "high")}


def journey_score(kg_per_traveller_day: float, scenario: str = "central") -> float:
    j = _emission_table()["journey"]
    ref = _kg(j["reference_mode"], j["reference_km_per_day"], 1, scenario)
    return round(1.0 - min(1.0, kg_per_traveller_day / ref), 4) if ref > 0 else 1.0


def trip_sustainability(place_mean: float, kg_per_traveller_day: float,
                        scenario: str = "central") -> float:
    lam = _emission_table()["journey"]["lambda"]
    return round((1 - lam) * place_mean + lam * journey_score(kg_per_traveller_day, scenario), 4)


def _straight_km(a, b) -> float:
    from math import radians, sin, cos, asin, sqrt
    la1, lo1, la2, lo2 = map(radians, (a.lat, a.lon, b.lat, b.lon))
    h = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_KM * asin(sqrt(h))
