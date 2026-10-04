"""
Candidate retrieval over the Neo4j knowledge graph.

This is the component that makes hardcoded itineraries impossible. Every place
the system can ever propose comes from this query, against real data, filtered
by the traveller's actual request. Nothing downstream may add a place.

Ranking
-------
Candidates are ordered by popularity ASCENDING within a prominence floor, so
under-visited destinations surface first. That is the sustainability objective
expressed directly in retrieval rather than bolted on afterwards: a recommender
that ranks by popularity concentrates demand on already-congested sites, which
is the mechanism by which recommendation technology intensifies overtourism.

The prominence floor matters too. Without it, ranking purely by low popularity
returns unvisitable minor features. Prominence answers "is this a real
destination"; popularity answers "is this crowded". They are separate signals.
"""
from __future__ import annotations
import logging
import time
import uuid
from typing import Optional, Protocol

from .models import (TripRequest, Candidate, CandidateSet, Category,
                     RetrievalDiagnostics)

log = logging.getLogger(__name__)

MIN_CANDIDATES = 15          # below this the solver has no room to optimise
TARGET_CANDIDATES = 150      # above this the CSP domain grows without benefit
DEFAULT_PROMINENCE_FLOOR = 0.25

# How strongly known crowding is penalised, in units of prominence.
#
# An earlier version ordered by popularity ASCENDING. Because popularity is
# resolved for only 7.3% of POIs and defaults to 0.0, that sorted "has no
# Wikipedia article" to the top and made Sigiriya and Dambulla unreachable in a
# heritage itinerary. Ranking is therefore driven by prominence, with crowding
# subtracted only where it is actually KNOWN. A famous crowded site is pushed
# down but stays reachable; an unmeasured site is neither rewarded nor punished
# for the absence of data.
DEFAULT_CROWDING_PENALTY = 0.35


CANDIDATE_QUERY = """
MATCH (p:POI)
WHERE p.category IN $categories
  AND p.prominence >= $prominence_floor
  AND p.entry_fee_lkr <= $max_fee
  AND point.distance(p.location,
        point({latitude: $lat, longitude: $lon})) <= $radius_m
WITH p,
     (p.popularity_source = 'wikipedia') AS popularity_known,
     CASE WHEN p.popularity_source = 'wikipedia'
          THEN coalesce(p.popularity_index, 0.0) ELSE 0.0 END AS known_pop
WITH p, popularity_known, known_pop,
     p.prominence - ($crowding_penalty * known_pop) AS retrieval_score
RETURN p.poi_id            AS poi_id,
       p.name              AS name,
       p.category          AS category,
       p.lat               AS lat,
       p.lon               AS lon,
       p.district          AS district,
       p.open_min          AS open_min,
       p.close_min         AS close_min,
       p.entry_fee_lkr     AS entry_fee_lkr,
       p.typical_dwell_min AS typical_dwell_min,
       coalesce(p.popularity_index, 0.0) AS popularity_index,
       p.prominence        AS prominence,
       popularity_known    AS popularity_known,
       retrieval_score     AS retrieval_score,
       coalesce(p.rail_access_score, 0.0) AS rail_access_score,
       coalesce(p.hours_estimated, true)  AS hours_estimated,
       coalesce(p.fee_estimated, true)    AS fee_estimated,
       point.distance(p.location,
         point({latitude: $lat, longitude: $lon})) / 1000.0
                           AS distance_from_start_km
ORDER BY retrieval_score DESC, p.prominence DESC, p.poi_id ASC
LIMIT $limit
"""
# poi_id breaks ties. Many places share a score, and without it which of them
# fell inside LIMIT was up to the database, so the same request could get a
# different candidate set - and a different plan - from one run to the next
# (Methodology 3.10.5: the same request with the same seed must give the same plan).

TRAVEL_QUERY = """
MATCH (a:POI)-[t:TRAVEL]-(b:POI)
WHERE a.poi_id IN $ids AND b.poi_id IN $ids AND a.poi_id < b.poi_id
RETURN a.poi_id AS from_id, b.poi_id AS to_id,
       t.duration_min AS duration_min, t.distance_km AS distance_km,
       t.method AS method
"""


class GraphSession(Protocol):
    """Minimal surface of a neo4j Session, so tests can inject a fake."""
    def run(self, query: str, **params): ...


class CandidateRetriever:
    """
    Retrieves candidate POIs for a trip request.

    The session is injected rather than constructed here, which keeps the query
    logic testable without a live database and keeps credential handling in one
    place.
    """

    def __init__(self, session: GraphSession,
                 prominence_floor: float = DEFAULT_PROMINENCE_FLOOR,
                 crowding_penalty: float = DEFAULT_CROWDING_PENALTY):
        self.session = session
        self.prominence_floor = prominence_floor
        self.crowding_penalty = crowding_penalty
        self.diagnostics = RetrievalDiagnostics()
        # Kept so downstream steps (transport mode, lodging) can look up a
        # candidate's coordinates without querying the graph again.
        self._last_candidates: list[Candidate] = []

    # -- internals ---------------------------------------------------------
    def _fetch(self, req: TripRequest, *, floor: float, radius_km: float,
               categories: list[str], limit: int) -> list[dict]:
        params = {
            "categories": categories,
            "prominence_floor": floor,
            "max_fee": req.poi_budget_lkr,
            "lat": req.start_lat,
            "lon": req.start_lon,
            "radius_m": radius_km * 1000.0,
            "crowding_penalty": self.crowding_penalty,
            "limit": limit,
        }
        result = self.session.run(CANDIDATE_QUERY, **params)
        return [dict(r) for r in result]

    @staticmethod
    def _to_candidates(rows: list[dict]) -> list[Candidate]:
        out, rejected = [], 0
        for r in rows:
            try:
                out.append(Candidate(**r))
            except Exception as e:              # noqa: BLE001
                rejected += 1
                log.debug("rejected row %s: %s", r.get("poi_id"), e)
        if rejected:
            log.warning("%d graph rows failed validation and were skipped "
                        "(likely close_min <= open_min)", rejected)
        return out

    # -- public ------------------------------------------------------------
    def retrieve(self, req: TripRequest) -> CandidateSet:
        """
        Retrieve candidates, widening the query only if too few are found.

        Widening is stepwise and logged, because a silently widened query is
        indistinguishable from a query that matched: the traveller would be
        shown places outside what they asked for with no indication why.
        """
        t0 = time.perf_counter()
        categories = [c.value for c in req.interests]
        floor = self.prominence_floor
        radius = min(req.max_travel_radius_km, 250.0)
        steps: list[str] = []

        rows = self._fetch(req, floor=floor, radius_km=radius,
                           categories=categories, limit=TARGET_CANDIDATES)
        self.diagnostics.candidates_before_filter = len(rows)

        # Step 1: relax the prominence floor.
        if len(rows) < MIN_CANDIDATES and floor > 0.0:
            floor = 0.0
            steps.append("prominence floor lowered to 0.0")
            rows = self._fetch(req, floor=floor, radius_km=radius,
                               categories=categories, limit=TARGET_CANDIDATES)

        # Step 2: widen the search radius.
        if len(rows) < MIN_CANDIDATES and radius < 250.0:
            radius = 250.0
            steps.append("radius widened to 250 km")
            rows = self._fetch(req, floor=floor, radius_km=radius,
                               categories=categories, limit=TARGET_CANDIDATES)

        # Step 3: admit all categories. Last resort - this returns places the
        # traveller did not ask for, so it is always surfaced to the user.
        if len(rows) < MIN_CANDIDATES:
            all_cats = [c.value for c in Category]
            steps.append("all categories admitted (interest filter relaxed)")
            rows = self._fetch(req, floor=floor, radius_km=radius,
                               categories=all_cats, limit=TARGET_CANDIDATES)

        candidates = self._to_candidates(rows)

        self.diagnostics.candidates_after_filter = len(candidates)
        self.diagnostics.widening_steps = steps
        if candidates:
            self.diagnostics.category_counts = _counts(c.category.value for c in candidates)
            self.diagnostics.district_counts = _counts(c.district for c in candidates)
            known = [c for c in candidates if c.popularity_known]
            self.diagnostics.mean_popularity = round(
                sum(c.popularity_index for c in known) / len(known), 4
            ) if known else 0.0
            self.diagnostics.popularity_known_pct = round(
                100.0 * len(known) / len(candidates), 1)
            self.diagnostics.mean_prominence = round(
                sum(c.prominence for c in candidates) / len(candidates), 4)

        elapsed = (time.perf_counter() - t0) * 1000.0
        if steps:
            log.warning("Query widened: %s", "; ".join(steps))
        log.info("Retrieved %d candidates across %d districts in %.0f ms",
                 len(candidates), len(self.diagnostics.district_counts), elapsed)

        self._last_candidates = candidates
        return CandidateSet(
            request_id=str(uuid.uuid4()),
            candidates=candidates,
            query_widened=bool(steps),
            widening_reason="; ".join(steps) if steps else None,
            districts_covered=len(self.diagnostics.district_counts),
            retrieval_ms=round(elapsed, 1),
        )

    def travel_times(self, cset: CandidateSet) -> dict[tuple[str, str], float]:
        """
        Stored travel durations between candidates, keyed by an ordered id pair.

        The graph holds a k-nearest-neighbour subset, so not every pair has a
        stored edge. Missing pairs are the solver's responsibility to estimate;
        returning a partial map is correct and is preferable to silently
        fabricating durations here.
        """
        ids = sorted(cset.ids)
        if not ids:
            return {}
        rows = list(self.session.run(TRAVEL_QUERY, ids=ids))
        out: dict[tuple[str, str], float] = {}
        for r in rows:
            d = dict(r)
            a, b = d["from_id"], d["to_id"]
            out[(a, b) if a < b else (b, a)] = float(d["duration_min"])

        possible = len(ids) * (len(ids) - 1) // 2
        log.info("Travel edges: %d of %d candidate pairs stored (%.1f%%)",
                 len(out), possible, 100 * len(out) / max(possible, 1))
        return out


def _counts(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
