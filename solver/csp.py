"""
Constraint satisfaction for itinerary construction.

Formulation
-----------
  Variables  X = { slot(d, s) : d in 1..days, s in 0..max_slots-1 }
  Domains    D(x) = retrieved candidates + NULL   (NULL = "no visit in this slot")
  Constraints
    C1 unary   fee <= per-POI cap
    C2 unary   the POI can be visited within the working day at all
    C3 binary  consecutive slots on a day must be mutually reachable
    C4 binary  no POI may appear twice in the whole trip
    C5 binary  NULL slots trail: if slot s is NULL then slot s+1 is NULL
    C6 global  total entry fees <= budget share   (enforced during search)

C1 and C2 are node consistency, applied first because they are cheap and shrink
the domains that AC-3 then has to work over. C3-C5 are binary and are what AC-3
propagates. C6 is global, so it cannot be expressed as an arc and is enforced by
bounding during branch and bound.

AC-3
----
Implemented as Mackworth (1977) specifies: a queue of arcs, a REVISE step that
removes any value in D(x_i) with no supporting value in D(x_j), and re-enqueuing
of the neighbouring arcs whenever a domain shrinks. The algorithm is sound (it
never removes a value that participates in a solution) but not complete (an
arc-consistent problem may still have no solution), so search follows.

Constraint hardness
-------------------
Opening hours are enforced ONLY where the source supplied them. 97% of the
dataset's opening hours are category defaults, and enforcing those as hard
constraints would reject feasible itineraries on the basis of invented data.
Imputed hours are recorded as soft violations instead.
"""
from __future__ import annotations
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

from retrieval.models import Candidate
from .models import SolverConfig, SolverDiagnostics

log = logging.getLogger(__name__)

NULL = "__NULL__"          # the "no visit" value present in every domain


@dataclass(frozen=True)
class Slot:
    day: int
    index: int

    def __repr__(self) -> str:      # pragma: no cover - debug aid
        return f"d{self.day}s{self.index}"


# Pseudo-id for the traveller's starting point in the travel model.
START = "__start__"


class TravelModel:
    """
    Travel durations between candidates.

    The knowledge graph stores a k-nearest-neighbour subset, so most candidate
    pairs have no stored edge. Missing pairs fall back to a corrected
    straight-line estimate and are FLAGGED, because a solver that silently
    invents durations produces itineraries that cannot be executed.
    """

    def __init__(self, candidates: list[Candidate],
                 stored: dict[tuple[str, str], float],
                 avg_kmh: float = 35.0, detour: float = 1.35,
                 start: Optional[tuple[float, float]] = None):
        self.pos = {c.poi_id: (c.lat, c.lon) for c in candidates}
        # The starting point has no stored edges, so hops from it are always
        # the flagged straight-line estimate.
        if start is not None:
            self.pos[START] = (float(start[0]), float(start[1]))
        self.stored = stored
        self.avg_kmh = avg_kmh
        self.detour = detour
        self.hits = 0
        self.misses = 0

    def _haversine_km(self, a: str, b: str) -> float:
        from math import radians, sin, cos, asin, sqrt
        la1, lo1 = self.pos[a]
        la2, lo2 = self.pos[b]
        la1, lo1, la2, lo2 = map(radians, (la1, lo1, la2, lo2))
        h = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
        return 2 * 6371.0 * asin(sqrt(h))

    def road_km(self, a: str, b: str) -> float:
        """Estimated road distance: straight line x detour, as the time estimate uses."""
        return 0.0 if a == b else self._haversine_km(a, b) * self.detour

    def minutes(self, a: str, b: str) -> tuple[float, bool]:
        """Returns (minutes, estimated). `estimated` is True when no edge existed."""
        if a == b:
            return 0.0, False
        key = (a, b) if a < b else (b, a)
        if key in self.stored:
            self.hits += 1
            return self.stored[key], False
        self.misses += 1
        km = self._haversine_km(a, b)
        return (km * self.detour / self.avg_kmh) * 60.0, True


# ---------------------------------------------------------------------------
class ItineraryCSP:
    """Builds the CSP, applies node consistency, then AC-3."""

    def __init__(self, candidates: list[Candidate], travel: TravelModel,
                 days: int, poi_budget: float, cfg: SolverConfig):
        self.cfg = cfg
        self.days = days
        self.poi_budget = poi_budget
        self.travel = travel
        self.by_id: dict[str, Candidate] = {c.poi_id: c for c in candidates}

        self.slots: list[Slot] = [Slot(d, s)
                                  for d in range(1, days + 1)
                                  for s in range(cfg.max_slots_per_day)]
        self.domains: dict[Slot, set[str]] = {}
        self.diagnostics = SolverDiagnostics()
        self.soft_violations: dict[str, int] = {}

    # -- node consistency (C1, C2) ----------------------------------------
    def _unary_ok(self, c: Candidate) -> bool:
        # C1: a single entry fee must not consume the whole POI budget.
        if c.fee_is_hard and c.entry_fee_lkr > self.poi_budget:
            return False

        # C2: the visit must fit inside the working day. Enforced only where
        # the opening hours are real; imputed hours are recorded, not enforced.
        if c.hours_are_hard:
            earliest = max(self.cfg.day_start_min, c.open_min)
            if earliest + c.typical_dwell_min > min(self.cfg.day_end_min, c.close_min):
                return False
        else:
            self.soft_violations["hours_imputed"] = \
                self.soft_violations.get("hours_imputed", 0) + 1
        return True

    def build(self) -> None:
        viable = {pid for pid, c in self.by_id.items() if self._unary_ok(c)}
        dropped = len(self.by_id) - len(viable)
        if dropped:
            log.info("Node consistency removed %d candidate(s)", dropped)

        base = viable | {NULL}
        self.domains = {s: set(base) for s in self.slots}

        # The first `min_stops_per_day` slots of each day may not be NULL, so the
        # solver cannot satisfy the problem by planning an empty trip.
        #
        # This previously cleared NULL from slot 0 only, so min_stops_per_day
        # above 1 had no effect on the domains at all - and because NULL
        # supports every value in the reachability constraint, AC-3 could then
        # never prune anything: every POI always had NULL as a fallback support.
        # Clearing NULL across the required slots is what lets arc consistency
        # actually bite on the reachability constraint.
        required = max(0, min(self.cfg.min_stops_per_day, self.cfg.max_slots_per_day))
        for d in range(1, self.days + 1):
            for idx in range(required):
                self.domains[Slot(d, idx)].discard(NULL)

        self.diagnostics.variables = len(self.slots)
        self.diagnostics.domain_size_before = sum(len(v) for v in self.domains.values())

    def cap_domains(self, utility: dict[str, float],
                    keep: Optional[set[str]] = None) -> None:
        """
        Reduce each slot's domain to a manageable size for search.

        Taking simply the top-K by utility was the real cause of the narrow
        catalogue coverage: the same K candidates were the only ones ever
        considered, so no amount of re-ranking inside the search could widen
        the result. The cap is therefore STRATIFIED - it fills round-robin
        across (district, category) groups, taking the best remaining
        candidate from each group in turn.

        Applied after AC-3 so propagation still runs over full domains, and
        recorded in diagnostics so no optimality claim is made without
        disclosing it.
        """
        k = self.cfg.max_domain_per_slot
        if k <= 0:
            return

        groups: dict[tuple, list[str]] = {}
        for pid, c in self.by_id.items():
            groups.setdefault((c.district, c.category.value), []).append(pid)
        for g in groups.values():
            g.sort(key=lambda v: -utility.get(v, 0.0))

        # Round-robin across groups, best-first within each.
        order: list[str] = []
        depth = 0
        while len(order) < len(self.by_id):
            added = False
            for g in sorted(groups,
                            key=lambda kk: -utility.get(groups[kk][0], 0.0)):
                if depth < len(groups[g]):
                    order.append(groups[g][depth])
                    added = True
            if not added:
                break
            depth += 1
        rank = {pid: i for i, pid in enumerate(order)}

        for slot, dom in self.domains.items():
            reals = sorted((v for v in dom if v != NULL),
                           key=lambda v: rank.get(v, len(rank)))[:k]
            if keep:
                reals = reals + [v for v in keep if v in dom and v not in reals]
            self.domains[slot] = set(reals) | ({NULL} if NULL in dom else set())

        self.diagnostics.domain_capped_to = k
        self.diagnostics.domain_size_capped = sum(
            len(v) for v in self.domains.values())

    # -- binary constraints ------------------------------------------------
    def _reachable(self, a: str, b: str) -> bool:
        """C3: b can follow a on the same day within the time available."""
        if a == NULL or b == NULL:
            return True
        if a == b:
            return False
        ca, cb = self.by_id[a], self.by_id[b]
        mins, _ = self.travel.minutes(a, b)
        if mins > self.cfg.max_leg_minutes:
            return False
        need = (ca.typical_dwell_min + mins + self.cfg.buffer_minutes
                + cb.typical_dwell_min)
        return need <= self.cfg.day_active_minutes

    def _consistent(self, xi: Slot, a: str, xj: Slot, b: str) -> bool:
        # C4: no POI twice anywhere in the trip.
        if a != NULL and b != NULL and a == b:
            return False

        same_day = xi.day == xj.day

        # Day coherence: any two stops on the same day must be mutually close,
        # not just consecutive ones.
        if same_day and a != NULL and b != NULL:
            mins, _ = self.travel.minutes(a, b)
            if mins > self.cfg.max_intra_day_minutes:
                return False

        if same_day and abs(xi.index - xj.index) == 1:
            first, second = (a, b) if xi.index < xj.index else (b, a)
            # C5: NULL slots trail, removing symmetric duplicate solutions.
            if first == NULL and second != NULL:
                return False
            if not self._reachable(first, second):
                return False
        return True

    def _neighbours(self, x: Slot) -> list[Slot]:
        """Every other slot constrains x through C4; adjacent slots also via C3/C5."""
        return [s for s in self.slots if s != x]

    # -- AC-3 --------------------------------------------------------------
    def ac3(self) -> bool:
        """
        Enforce arc consistency. Returns False if a domain is emptied, which
        proves the problem has no solution under the stated constraints.
        """
        queue: deque[tuple[Slot, Slot]] = deque(
            (xi, xj) for xi in self.slots for xj in self._neighbours(xi))
        revisions = 0

        while queue:
            xi, xj = queue.popleft()
            revisions += 1
            if self._revise(xi, xj):
                if not self.domains[xi]:
                    self.diagnostics.wipeout = True
                    self.diagnostics.arc_revisions = revisions
                    self.diagnostics.domain_size_after = sum(
                        len(v) for v in self.domains.values())
                    log.warning("AC-3 wipeout at %s: no feasible assignment", xi)
                    return False
                for xk in self._neighbours(xi):
                    if xk != xj:
                        queue.append((xk, xi))

        self.diagnostics.arc_revisions = revisions
        self.diagnostics.domain_size_after = sum(len(v) for v in self.domains.values())
        log.info("AC-3: %d revisions, domain %d -> %d (%.1f%% reduction)",
                 revisions, self.diagnostics.domain_size_before,
                 self.diagnostics.domain_size_after,
                 self.diagnostics.domain_reduction_pct)
        return True

    def _revise(self, xi: Slot, xj: Slot) -> bool:
        """Remove values in D(xi) with no support in D(xj). Returns True if changed."""
        removed = {a for a in self.domains[xi]
                   if not any(self._consistent(xi, a, xj, b)
                              for b in self.domains[xj])}
        if removed:
            self.domains[xi] -= removed
        return bool(removed)

    # -- reporting ---------------------------------------------------------
    def finalise_diagnostics(self) -> SolverDiagnostics:
        # Propagation lookups only. The itinerary-level counts are filled in by
        # solve() once a plan exists, because those are the legs a traveller
        # actually makes and therefore the only ones worth reporting.
        self.diagnostics.travel_lookups_stored = self.travel.hits
        self.diagnostics.travel_lookups_estimated = self.travel.misses
        self.diagnostics.soft_violations = dict(self.soft_violations)
        return self.diagnostics
