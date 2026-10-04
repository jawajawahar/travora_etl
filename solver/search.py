"""
Branch-and-bound search over the arc-consistent CSP.

AC-3 is sound but not complete: it removes values that cannot appear in any
solution, but an arc-consistent problem may still have none. Search therefore
follows propagation, and this module performs it.

Search strategy
---------------
Slots are assigned day by day, in order. At each node the partial utility is
extended by an OPTIMISTIC bound on the unassigned slots - the best remaining
per-slot utility multiplied by the slots left. Because the bound can never
understate what remains achievable, pruning a branch whose bound falls below the
incumbent cannot discard the optimum. That is what makes the pruning safe rather
than merely fast.

The search is bounded by wall-clock time and node count. When either bound is
hit the best itinerary found so far is returned with `optimal=False`, which is
reported honestly rather than presented as an optimum.
"""
from __future__ import annotations
import logging
import math
import time
from typing import Optional

from retrieval.models import Candidate
from .csp import ItineraryCSP, Slot, NULL, TravelModel, START
from .emissions import (kg_per_traveller, estimated_mode, reference_kg_per_day,
                        journey_params)
from .models import (SolverConfig, ScheduledStop, DayPlan, Itinerary,
                     Conflict, ConflictType, SolverResult)

log = logging.getLogger(__name__)


class BranchAndBound:
    def __init__(self, csp: ItineraryCSP, utility: dict[str, float],
                 pref: dict[str, float], sust: dict[str, float],
                 avoid: Optional[list[set[str]]] = None,
                 max_shared_frac: float = 0.5,
                 emission_weight: float = 0.0, party_size: int = 1):
        self.csp = csp
        self.cfg = csp.cfg
        self.has_start = START in csp.travel.pos
        # Journey part of the objective: each leg costs
        #   w_sust * lambda * slots_per_day * (kg per traveller / reference kg per day),
        # which is the per-leg form of lambda * w_sust * S_journey summed over days.
        self.emission_weight = emission_weight
        self.party_size = party_size
        self.ref_kg = reference_kg_per_day()
        self.utility = utility
        self.pref = pref
        self.sust = sust

        # Diversity cuts: a plan may share at most floor(frac * |prev|) stops
        # with each earlier plan. Overlap only grows as stops are added, so a
        # partial assignment over the limit can be pruned, and because the cut
        # only removes solutions the utility bound stays admissible.
        self._cuts = [(s, int(max_shared_frac * len(s))) for s in (avoid or []) if s]

        self.best_assignment: Optional[dict[Slot, str]] = None
        self.best_value = float("-inf")
        self._seed: Optional[dict[Slot, str]] = None
        self.nodes = 0
        self.deadline = 0.0
        self.hit_time = False
        self.hit_nodes = False

        # Admissible bound.
        #
        # The first version used `remaining * max_utility`, which is valid but
        # so loose that search exhausted its node cap on realistic instances and
        # never proved optimality. Since no POI may repeat, the true ceiling on
        # `remaining` further slots is the sum of the `remaining` LARGEST
        # utilities in the pool. Prefix sums make that an O(1) lookup and it is
        # still admissible - it can never understate what remains achievable, so
        # pruning against it cannot discard the optimum.
        # Similarity for MMR. Two places are most alike when they share both
        # district and category, which is exactly the repetition that produced
        # five museums in one city.
        self.mmr_lambda = csp.cfg.mmr_lambda
        self._meta = {pid: (c.district, c.category.value)
                      for pid, c in csp.by_id.items()}
        self.mmr_penalty_total = 0.0

        vals = sorted(utility.values(), reverse=True)
        self._prefix: list[float] = [0.0]
        for v in vals:
            self._prefix.append(self._prefix[-1] + max(v, 0.0))
        self.max_unit = vals[0] if vals else 0.0

        # Slots of each day in index order, built once: the day checks run at
        # every search node and rebuilding these lists there was measurable.
        self._by_day: dict[int, list[Slot]] = {}
        for s in csp.slots:
            self._by_day.setdefault(s.day, []).append(s)
        for slots in self._by_day.values():
            slots.sort(key=lambda s: s.index)

    # -- day-level feasibility --------------------------------------------
    def _anchor(self, assignment: dict[Slot, str], day: int) -> Optional[str]:
        """
        Where a day begins: the last stop of the most recent earlier day with
        stops (where the traveller slept), else the starting point. Days are
        searched in order, so every earlier day is already fully assigned.
        """
        if not self.cfg.chain_days:
            return None
        for d in range(day - 1, 0, -1):
            real = [assignment[s] for s in sorted(self._day_slots(d), key=lambda s: s.index)
                    if s in assignment and assignment[s] != NULL]
            if real:
                return real[-1]
        return START if self.has_start else None

    def _schedule_day(self, ordered: list[str], anchor: Optional[str] = None
                      ) -> Optional[list[tuple[str, int, int, float, bool]]]:
        """
        Lay out one day's stops in time, starting from `anchor` when given.

        Returns [(poi_id, arrive, depart, travel_from_prev, travel_estimated)]
        or None if the day cannot be scheduled. Opening hours are enforced only
        where the source supplied them.
        """
        cfg = self.cfg
        out: list[tuple[str, int, int, float, bool]] = []
        clock = cfg.day_start_min
        prev: Optional[str] = anchor

        # Day coherence, mirroring the CSP constraint exactly. The propagator and
        # the scheduler must enforce the same rules or AC-3 can prune plans the
        # search would have accepted, and the ablation stops being meaningful.
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                mins, _ = self.csp.travel.minutes(a, b)
                if mins > cfg.max_intra_day_minutes:
                    return None

        for i, pid in enumerate(ordered):
            c = self.csp.by_id[pid]
            travel, estimated = (0.0, False) if prev is None else \
                self.csp.travel.minutes(prev, pid)

            # The same leg limit AC-3 propagates. Without this the scheduler
            # accepted itineraries the constraint model rejects, so enabling
            # AC-3 could return a WORSE objective than disabling it - the
            # propagator and the search must agree on what feasible means.
            # The day's first hop, from the anchor, has its own transfer limit;
            # AC-3 does not propagate it, which keeps propagation sound (the
            # scheduler is only ever stricter than the propagator).
            limit = cfg.max_transfer_minutes if (i == 0 and anchor is not None) \
                else cfg.max_leg_minutes
            if prev is not None and travel > limit:
                return None

            # Rounded UP: rounding to nearest let a day the solver saw ending at
            # 17:59 end after 18:00 in the independent verifier, which keeps
            # fractions. A conservative schedule can never claim false feasibility.
            arrive = clock + math.ceil(travel) + (0 if prev is None else cfg.buffer_minutes)

            if c.hours_are_hard:
                arrive = max(arrive, c.open_min)
                if arrive + c.typical_dwell_min > c.close_min:
                    return None

            depart = arrive + c.typical_dwell_min
            if depart > cfg.day_end_min:
                return None

            out.append((pid, arrive, depart, travel, estimated))
            clock = depart
            prev = pid
        return out

    # -- search ------------------------------------------------------------
    def solve(self, poi_budget: float) -> Optional[dict[Slot, str]]:
        self.deadline = time.perf_counter() + self.cfg.time_budget_s
        self._seed = None
        if self.cfg.warm_start:
            for prefer in ("gain", "near"):
                found = self._greedy(prefer)
                if found and found[1] > self.best_value:
                    self.best_assignment, self.best_value = found
                    self._seed = self.best_assignment
        order = self.csp.slots                       # already day-major, slot-minor
        self._search(order, 0, {}, 0.0, 0.0, set())
        return self.best_assignment

    @property
    def returned_seed(self) -> bool:
        """True when search stopped without improving on the greedy seed."""
        return self._seed is not None and self.best_assignment is self._seed

    # -- greedy seed ---------------------------------------------------------
    def _prev_stop(self, assignment: dict[Slot, str], slot: Slot) -> Optional[str]:
        if slot.index > 0:
            prev = assignment.get(Slot(slot.day, slot.index - 1))
        else:
            prev = self._anchor(assignment, slot.day)
        return None if prev in (None, NULL) else prev

    def _greedy(self, prefer: str) -> Optional[tuple[dict[Slot, str], float]]:
        """
        One quick feasible itinerary to start the search from, or None.

        Days are filled in order, each from where it starts. At every slot the
        choice is limited to values the search itself would accept: the slot's
        domain, no repeats, the diversity cuts, the budget, and a day that still
        schedules. The result is therefore a real solution, valued with the
        search's own gain, so pruning against it cannot discard a better plan.

        prefer="gain" takes the highest utility net of travel; prefer="near"
        the shortest hop from the previous stop, which keeps days compact.
        Enough budget is held back to buy the stops the remaining days still
        need at the lowest fees still available, so early days cannot spend it
        all. (Reserving "stops needed x cheapest fee" reserved nothing whenever
        a single free place existed, and the seed ran out of money by day 2-4.)
        """
        need_min = self.cfg.min_stops_per_day
        pool_fees = {v: self.csp.by_id[v].entry_fee_lkr
                     for s in self.csp.slots for v in self.csp.domains[s] if v != NULL}
        assignment: dict[Slot, str] = {}
        used: set[str] = set()
        cost = value = 0.0

        for day in range(1, self.csp.days + 1):
            closed = False
            for slot in self._day_slots(day):
                domain = self.csp.domains[slot]
                if closed:
                    if NULL not in domain:
                        return None
                    assignment[slot] = NULL
                    continue
                today = sum(1 for s in self._day_slots(day) if assignment.get(s, NULL) != NULL)
                later = (self.csp.days - day) * need_min + max(0, need_min - today - 1)
                spare = sorted(f for v, f in pool_fees.items() if v not in used)
                # Cheapest `later` fees still available, and one more, so that a
                # candidate which is itself among the cheapest can be swapped out
                # of the reserve for the next cheapest.
                cheap_k = sum(spare[:later])
                cheap_k1 = sum(spare[:later + 1]) if len(spare) > later else float("inf")
                kth = spare[later - 1] if 0 < later <= len(spare) else float("inf")

                best_key, best_v, best_gain = None, None, 0.0
                for v in domain:
                    if v == NULL or v in used:
                        continue
                    if self._cuts and any(v in s and len(used & s) + 1 > limit
                                          for s, limit in self._cuts):
                        continue
                    fee = self.csp.by_id[v].entry_fee_lkr
                    reserve = cheap_k1 - fee if later and fee <= kth else cheap_k
                    if cost + fee + reserve > self.csp.poi_budget:
                        continue
                    assignment[slot] = v
                    if self._day_prefix_ok(assignment, slot):
                        gain = (self._effective_utility(v, used)
                                - self._travel_cost(assignment, slot, v))
                        if prefer == "near":
                            prev = self._prev_stop(assignment, slot)
                            mins = self.csp.travel.minutes(prev, v)[0] if prev else 0.0
                            key = (-mins, gain)
                        else:
                            key = (gain,)
                        if best_key is None or key > best_key:
                            best_key, best_v, best_gain = key, v, gain
                    del assignment[slot]

                if best_v is None:
                    if NULL not in domain:
                        return None
                    assignment[slot] = NULL
                    closed = True
                    continue
                assignment[slot] = best_v
                used.add(best_v)
                cost += self.csp.by_id[best_v].entry_fee_lkr
                value += best_gain

        if cost > self.csp.poi_budget or not self._days_valid(assignment):
            return None
        return assignment, value

    def _similarity(self, a: str, b: str) -> float:
        da, ca = self._meta.get(a, ("", ""))
        db, cb = self._meta.get(b, ("", ""))
        if da == db and ca == cb:
            return 1.0
        if da == db:
            return 0.6
        if ca == cb:
            return 0.4
        return 0.0

    def _effective_utility(self, v: str, chosen: set[str]) -> float:
        """
        Utility less an MMR penalty for resembling what is already selected.

        The penalty only ever REDUCES a value, so the admissible bound built
        from raw utilities remains an upper bound and pruning stays safe.
        """
        u = self.utility.get(v, 0.0)
        if not chosen or self.mmr_lambda <= 0.0 or v == NULL:
            return u
        worst = max(self._similarity(v, c) for c in chosen)
        return u - self.mmr_lambda * worst * u

    def _travel_cost(self, assignment: dict[Slot, str], slot: Slot, v: str) -> float:
        """
        Cost of travelling into v: from the previous stop that day, or from
        where the day starts. It only ever lowers a value, so the utility
        bound stays admissible and pruning stays safe.
        """
        prev = None
        if slot.index > 0:
            prev = assignment.get(Slot(slot.day, slot.index - 1))
        else:
            prev = self._anchor(assignment, slot.day)
        if prev is None or prev == NULL:
            return 0.0
        mins, _ = self.csp.travel.minutes(prev, v)
        cost = self.cfg.travel_cost_per_hour * mins / 60.0
        if self.emission_weight > 0:
            km = self.csp.travel.road_km(prev, v)
            kg = kg_per_traveller(estimated_mode(km), km, self.party_size)
            cost += self.emission_weight * kg / self.ref_kg
        return cost

    def _bound(self, remaining: int) -> float:
        """Upper bound on the utility obtainable from `remaining` further slots."""
        idx = min(remaining, len(self._prefix) - 1)
        return self._prefix[idx]

    def _search(self, order: list[Slot], i: int, assignment: dict[Slot, str],
                value: float, cost: float, used: set[str]) -> None:
        if time.perf_counter() > self.deadline:
            self.hit_time = True
            return
        if self.nodes >= self.cfg.max_nodes:
            self.hit_nodes = True
            return
        self.nodes += 1

        if i == len(order):
            if value > self.best_value and self._days_valid(assignment):
                self.best_value = value
                self.best_assignment = dict(assignment)
            return

        # Safe pruning: the bound never understates what the rest can add.
        if value + self._bound(len(order) - i) <= self.best_value:
            return

        slot = order[i]
        domain = self.csp.domains[slot]

        # Try the most promising values first so the incumbent rises quickly,
        # ordering by the MMR-adjusted value so diverse options surface earlier.
        values = sorted((v for v in domain if v != NULL),
                        key=lambda v: -self._effective_utility(v, used))
        if NULL in domain:
            values.append(NULL)

        for v in values:
            if v != NULL and v in used:
                continue
            if v != NULL and self._cuts and any(
                    v in s and len(used & s) + 1 > limit for s, limit in self._cuts):
                continue
            if v != NULL:
                c = self.csp.by_id[v]
                new_cost = cost + c.entry_fee_lkr
                if new_cost > self.csp.poi_budget:
                    continue
            else:
                new_cost = cost

            assignment[slot] = v
            if v != NULL:
                used.add(v)

            if self._day_prefix_ok(assignment, slot):
                gain = (self._effective_utility(v, used - {v}) if v != NULL
                        else 0.0)
                if v != NULL and (self.cfg.travel_cost_per_hour > 0 or self.emission_weight > 0):
                    gain -= self._travel_cost(assignment, slot, v)
                self._search(order, i + 1, assignment,
                             value + gain, new_cost, used)

            del assignment[slot]
            if v != NULL:
                used.discard(v)

    def _day_slots(self, day: int) -> list[Slot]:
        return self._by_day.get(day, [])

    def _day_prefix_ok(self, assignment: dict[Slot, str], just_set: Slot) -> bool:
        """Check the day containing `just_set` is still schedulable so far."""
        slots = [s for s in self._day_slots(just_set.day) if s in assignment]
        ordered = [assignment[s] for s in sorted(slots, key=lambda s: s.index)]

        # NULL slots must trail, so a gap is invalid.
        seen_null = False
        for v in ordered:
            if v == NULL:
                seen_null = True
            elif seen_null:
                return False

        real = [v for v in ordered if v != NULL]
        if not real:
            return True
        return self._schedule_day(real, self._anchor(assignment, just_set.day)) is not None

    def _days_valid(self, assignment: dict[Slot, str]) -> bool:
        for d in range(1, self.csp.days + 1):
            slots = sorted(self._day_slots(d), key=lambda s: s.index)
            real = [assignment[s] for s in slots
                    if s in assignment and assignment[s] != NULL]
            if len(real) < self.cfg.min_stops_per_day:
                return False
            if self._schedule_day(real, self._anchor(assignment, d)) is None:
                return False
        return True

    # -- assembly ----------------------------------------------------------
    def build_itinerary(self, request_id: str,
                        assignment: dict[Slot, str]) -> Itinerary:
        days: list[DayPlan] = []
        total_cost = 0.0
        prefs: list[float] = []
        susts: list[float] = []

        for d in range(1, self.csp.days + 1):
            slots = sorted(self._day_slots(d), key=lambda s: s.index)
            real = [assignment[s] for s in slots
                    if s in assignment and assignment[s] != NULL]
            anchor = self._anchor(assignment, d)
            laid = self._schedule_day(real, anchor) or []

            stops: list[ScheduledStop] = []
            for idx, (pid, arrive, depart, travel, estimated) in enumerate(laid):
                c = self.csp.by_id[pid]
                stops.append(ScheduledStop(
                    poi_id=pid, name=c.name, lat=c.lat, lon=c.lon,
                    day=d, slot=idx,
                    arrive_min=arrive, depart_min=depart,
                    dwell_min=c.typical_dwell_min,
                    travel_from_prev_min=round(travel, 1),
                    travel_estimated=estimated,
                    cost_lkr=c.entry_fee_lkr, district=c.district,
                    category=c.category.value,
                    utility=round(self.utility.get(pid, 0.0), 4),
                    hours_enforced=c.hours_are_hard,
                    fee_enforced=c.fee_is_hard,
                    from_anchor=(idx == 0 and anchor is not None),
                ))
                total_cost += c.entry_fee_lkr
                prefs.append(self.pref.get(pid, 0.0))
                susts.append(self.sust.get(pid, 0.0))
            days.append(DayPlan(day=d, stops=stops))

        n = max(len(prefs), 1)
        return Itinerary(
            request_id=request_id, days=days,
            total_cost_lkr=round(total_cost, 2),
            utility=round(self.best_value, 4),
            s_pref=round(sum(prefs) / n, 4),
            s_sust=round(sum(susts) / n, 4),
            optimal=not (self.hit_time or self.hit_nodes),
        )


# ---------------------------------------------------------------------------
def _keep_affordable(csp: ItineraryCSP, uncapped: dict[Slot, set[str]],
                     utility: dict[str, float], keep: Optional[set[str]],
                     days: int, poi_budget: float, cfg: SolverConfig) -> int:
    """
    Undo a cap that priced the request out of its own budget.

    The minimum stops (days x min_stops_per_day) are priced at the lowest
    fees available, first in the capped pool, then in the full pool. Only
    when the capped pool cannot pay but the full pool can is the cap redone,
    keeping the cheapest places as well. Adding values never removes a
    solution, so every guarantee of the search is unchanged. Returns how many
    places were added: 0 when the budget did not bind, and 0 when even the
    full pool is over budget, since that request is genuinely infeasible.
    """
    if not cfg.keep_cheapest_when_budget_binds or cfg.max_domain_per_slot <= 0:
        return 0
    need = days * cfg.min_stops_per_day

    def cheapest_total(ids: set[str]) -> float:
        fees = sorted(csp.by_id[v].entry_fee_lkr for v in ids)
        return sum(fees[:need]) if len(fees) >= need else float("inf")

    capped = {v for dom in csp.domains.values() for v in dom if v != NULL}
    if cheapest_total(capped) <= poi_budget:
        return 0
    full = {v for dom in uncapped.values() for v in dom if v != NULL}
    if cheapest_total(full) > poi_budget:
        return 0

    cheap = sorted(full, key=lambda v: (csp.by_id[v].entry_fee_lkr, -utility.get(v, 0.0)))[:need]
    added = [v for v in cheap if v not in capped]
    csp.domains = {slot: set(dom) for slot, dom in uncapped.items()}
    csp.cap_domains(utility, (keep or set()) | set(cheap))
    log.info("Budget binds after the cap: kept %d cheaper places", len(added))
    return len(added)


def solve(request_id: str, candidates: list[Candidate], days: int,
          poi_budget: float, pref: dict[str, float], sust: dict[str, float],
          w_pref: float, travel_edges: dict[tuple[str, str], float],
          cfg: Optional[SolverConfig] = None, alternatives: int = 0,
          max_shared_frac: float = 0.5, min_quality: float = 0.9,
          start: Optional[tuple[float, float]] = None,
          party_size: int = 1) -> SolverResult:
    """
    Build the CSP, propagate, search, and return an itinerary or a conflict.

    No language model is involved anywhere in this function. Feasibility is a
    structural property of the search, not an emergent property of generation.

    With `alternatives` > 0, the search is repeated with diversity cuts against
    every plan found so far. An alternative is kept only if its utility is at
    least `min_quality` of the best plan's, so "another option" never means a
    materially worse one; the gap is the measured price of diversity.
    """
    cfg = cfg or SolverConfig()
    t0 = time.perf_counter()

    if len(candidates) < days * cfg.min_stops_per_day:
        return SolverResult(
            feasible=False,
            conflict=Conflict(
                conflict=ConflictType.TOO_FEW_CANDIDATES,
                detail=f"{len(candidates)} candidates for {days} days",
                suggestion="Widen the interests or the search radius."))

    # Trip sustainability = (1 - lambda) * places + lambda * journey. The place
    # share enters per stop; the journey share is charged per leg in the search.
    w_sust = 1.0 - w_pref
    lam = journey_params()["lambda"]
    utility = {c.poi_id: w_pref * pref.get(c.poi_id, 0.0)
                        + w_sust * (1.0 - lam) * sust.get(c.poi_id, 0.0)
               for c in candidates}
    emission_weight = (w_sust * lam * cfg.max_slots_per_day
                       if cfg.journey_in_objective else 0.0)

    travel = TravelModel(candidates, travel_edges, start=start)
    csp = ItineraryCSP(candidates, travel, days, poi_budget, cfg)
    csp.build()

    csp.diagnostics.ac3_applied = cfg.use_ac3
    if cfg.use_ac3 and not csp.ac3():
        diag = csp.finalise_diagnostics()
        diag.time_ms = round((time.perf_counter() - t0) * 1000, 1)
        return SolverResult(
            feasible=False, diagnostics=diag,
            conflict=Conflict(
                conflict=ConflictType.DOMAIN_WIPEOUT,
                detail="Arc consistency emptied a slot domain: no assignment "
                       "satisfies the stated constraints.",
                binding_constraint="reachability or budget",
                suggestion="Increase the budget, extend the trip, or widen the "
                           "search radius."))

    if not cfg.use_ac3:
        csp.diagnostics.domain_size_after = csp.diagnostics.domain_size_before

    keep = None
    if start is not None and cfg.keep_nearest_to_start > 0:
        reach = sorted((travel.minutes(START, c.poi_id)[0], c.poi_id) for c in candidates)
        keep = {pid for mins, pid in reach[:cfg.keep_nearest_to_start]
                if mins <= cfg.max_transfer_minutes}
    uncapped = {slot: set(dom) for slot, dom in csp.domains.items()}
    csp.cap_domains(utility, keep)
    budget_keep_added = _keep_affordable(csp, uncapped, utility, keep, days, poi_budget, cfg)

    bnb = BranchAndBound(csp, utility, pref, sust,
                         emission_weight=emission_weight, party_size=party_size)
    assignment = bnb.solve(poi_budget)

    diag = csp.finalise_diagnostics()
    diag.search_nodes = bnb.nodes
    diag.mmr_lambda = cfg.mmr_lambda
    diag.hit_time_budget = bnb.hit_time
    diag.hit_node_cap = bnb.hit_nodes
    diag.budget_keep_added = budget_keep_added
    diag.warm_start_found = bnb._seed is not None
    diag.returned_warm_start = bnb.returned_seed
    diag.time_ms = round((time.perf_counter() - t0) * 1000, 1)

    if assignment is None:
        return SolverResult(
            feasible=False, diagnostics=diag,
            conflict=Conflict(
                conflict=ConflictType.NO_FEASIBLE_ASSIGNMENT,
                detail="Arc consistency left non-empty domains but no complete "
                       "assignment satisfies every constraint.",
                suggestion="Relax the daily schedule or increase the budget."))

    itin = bnb.build_itinerary(request_id, assignment)

    # Travel provenance over the legs actually travelled.
    legs = [s for s in itin.stops if s.slot > 0]
    diag.travel_edges_stored = sum(1 for s in legs if not s.travel_estimated)
    diag.travel_edges_estimated = sum(1 for s in legs if s.travel_estimated)

    log.info("Solved: %d stops, LKR %.0f, U=%.3f, %d nodes, %.0f ms%s",
             len(itin.stops), itin.total_cost_lkr, itin.utility,
             bnb.nodes, diag.time_ms, "" if itin.optimal else " (search bounded)")

    alts: list[Itinerary] = []
    seen = [{s.poi_id for s in itin.stops}]
    for n in range(1, alternatives + 1):
        alt_bnb = BranchAndBound(csp, utility, pref, sust,
                                 avoid=seen, max_shared_frac=max_shared_frac,
                                 emission_weight=emission_weight, party_size=party_size)
        alt_assignment = alt_bnb.solve(poi_budget)
        if alt_assignment is None or alt_bnb.best_value < min_quality * bnb.best_value:
            log.info("Alternative %d: none within %.0f%% of best", n, min_quality * 100)
            break
        alt = alt_bnb.build_itinerary(f"{request_id}-alt{n}", alt_assignment)
        alts.append(alt)
        seen.append({s.poi_id for s in alt.stops})

    return SolverResult(feasible=True, itinerary=itin, diagnostics=diag,
                        alternatives=alts)
