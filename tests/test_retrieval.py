"""
Tests for the candidate retrieval layer.

The most important test here is test_score_vector_rejects_invented_ids. That
validator is what makes the original failure - a language model emitting famous
destinations from its own priors - structurally impossible rather than merely
discouraged.

Run:  python -m pytest tests/test_retrieval.py -v
"""
from __future__ import annotations
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.models import (TripRequest, Candidate, CandidateSet, ScoreVector,
                              Category)
from retrieval.service import (CandidateRetriever, MIN_CANDIDATES,
                               TARGET_CANDIDATES, CANDIDATE_QUERY,
                               DEFAULT_CROWDING_PENALTY)


# ---------------------------------------------------------------------------
def make_row(i: int, *, category="heritage", pop=0.5, prom=0.6,
             fee=1000.0, district="Kandy", open_min=480, close_min=1020,
             known=True):
    return {
        "popularity_known": known,
        "retrieval_score": prom - (DEFAULT_CROWDING_PENALTY * pop if known else 0.0),
        "poi_id": f"node/{i}", "name": f"Place {i}", "category": category,
        "lat": 7.29 + i * 0.001, "lon": 80.63 + i * 0.001, "district": district,
        "open_min": open_min, "close_min": close_min, "entry_fee_lkr": fee,
        "typical_dwell_min": 90, "popularity_index": pop, "prominence": prom,
        "rail_access_score": 0.5, "hours_estimated": True,
        "fee_estimated": True, "distance_from_start_km": 5.0 + i,
    }


class FakeSession:
    """Records queries and returns scripted rows, so no database is needed."""
    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = []

    def run(self, query, **params):
        self.calls.append({"query": query, "params": params})
        if not self.batches:
            return []
        return self.batches.pop(0) if len(self.batches) > 1 else self.batches[0]


def request(**kw) -> TripRequest:
    base = dict(start_date=date(2026, 9, 1), days=7, budget_lkr=120000,
                party_size=2, interests=[Category.HERITAGE, Category.WILDLIFE],
                start_lat=7.29, start_lon=80.63)
    base.update(kw)
    return TripRequest(**base)


# ------------------------------------------------------------ TripRequest
def test_trip_request_weights_and_budget():
    r = request(w_pref=0.7, budget_lkr=100000)
    assert r.w_sust == 0.3
    assert r.poi_budget_lkr == 20000.0
    with pytest.raises(Exception):
        request(days=0)
    with pytest.raises(Exception):
        request(start_lat=13.08)          # Chennai, outside Sri Lanka
    with pytest.raises(Exception):
        request(interests=[])
    print("  TripRequest: weights, budget share and bounds validated")


# -------------------------------------------------------------- Candidate
def test_candidate_rejects_impossible_hours():
    with pytest.raises(Exception):
        Candidate(**make_row(1, open_min=1020, close_min=480))
    ok = Candidate(**make_row(1))
    assert ok.hours_are_hard is False, "imputed hours must not be a hard constraint"
    hard = Candidate(**{**make_row(2), "hours_estimated": False,
                        "fee_estimated": False})
    assert hard.hours_are_hard and hard.fee_is_hard
    print("  Candidate: impossible hours rejected, provenance flags exposed")


def test_candidate_set_rejects_duplicates():
    c = [Candidate(**make_row(1)), Candidate(**make_row(1))]
    with pytest.raises(Exception):
        CandidateSet(request_id="x", candidates=c)
    print("  CandidateSet: duplicate ids rejected")


# ----------------------------------------------------- THE ID CONTRACT
def test_score_vector_rejects_invented_ids():
    """An agent must not add a place that was never retrieved."""
    ids = ["node/1", "node/2"]
    with pytest.raises(ValueError, match="not in the candidate set"):
        ScoreVector(agent="preference", candidate_ids=ids,
                    scores={"node/1": 0.8, "node/2": 0.6, "node/sigiriya": 0.99})
    print("  ID contract: invented id rejected")


def test_score_vector_rejects_omitted_ids():
    ids = ["node/1", "node/2", "node/3"]
    with pytest.raises(ValueError, match="omitted"):
        ScoreVector(agent="preference", candidate_ids=ids,
                    scores={"node/1": 0.8, "node/2": 0.6})
    print("  ID contract: omitted candidate rejected")


def test_score_vector_rejects_out_of_range():
    ids = ["node/1"]
    with pytest.raises(ValueError, match="out-of-range"):
        ScoreVector(agent="sustainability", candidate_ids=ids,
                    scores={"node/1": 1.4})
    print("  ID contract: out-of-range score rejected")


def test_score_vector_accepts_exact_match():
    ids = ["node/1", "node/2"]
    sv = ScoreVector(agent="preference", candidate_ids=ids,
                     scores={"node/2": 0.6, "node/1": 0.8})   # order irrelevant
    assert sv.scores["node/1"] == 0.8
    print("  ID contract: exact match accepted regardless of ordering")


# ------------------------------------------------------------- retrieval
def test_ranking_penalises_known_crowding_only():
    """
    Ranking must not treat missing popularity data as evidence of low crowding.

    An earlier version ordered by popularity ASCENDING. Popularity resolves for
    only 7.3% of POIs and defaults to 0.0, so that sorted "has no Wikipedia
    article" to the top and made Sigiriya and Dambulla unreachable in a heritage
    itinerary - the opposite of a useful result.
    """
    assert "ORDER BY retrieval_score DESC" in CANDIDATE_QUERY
    assert "p.popularity_source = 'wikipedia'" in CANDIDATE_QUERY, \
        "crowding must be applied only where popularity is actually known"
    assert "ORDER BY p.popularity_index ASC" not in CANDIDATE_QUERY, \
        "the naive ascending sort must not return"
    print("  ranking: prominence-driven, crowding penalised only when known")


def test_famous_crowded_site_remains_reachable():
    """A crowded landmark is pushed down the ranking but not excluded."""
    w = DEFAULT_CROWDING_PENALTY
    sigiriya = 1.00 - w * 1.00          # maximally prominent, maximally crowded
    small_museum = 0.48 - 0.0           # modest, unmeasured
    assert sigiriya > small_museum, (
        f"crowding penalty {w} is too aggressive: a top landmark ({sigiriya:.2f}) "
        f"ranks below a minor unmeasured site ({small_museum:.2f})")
    print(f"  reachability: landmark scores {sigiriya:.2f} vs "
          f"minor site {small_museum:.2f}")


def test_unknown_popularity_is_not_rewarded_as_uncrowded():
    """Two equally prominent POIs: the one with unknown popularity must not win
    purely because its popularity defaults to zero."""
    w = DEFAULT_CROWDING_PENALTY
    known_quiet = 0.70 - w * 0.05       # measured, genuinely quiet
    unknown = 0.70 - 0.0                # unmeasured
    assert unknown - known_quiet < 0.05, (
        "an unmeasured POI gains too much advantage over a measured quiet one")
    print("  unknown popularity carries no large advantage over measured-quiet")


def test_retrieve_builds_correct_parameters():
    rows = [make_row(i) for i in range(40)]
    s = FakeSession([rows])
    out = CandidateRetriever(s).retrieve(request(budget_lkr=100000))

    p = s.calls[0]["params"]
    assert set(p["categories"]) == {"heritage", "wildlife"}
    assert p["max_fee"] == 20000.0                 # 20% of budget
    assert p["radius_m"] == 250000.0
    assert p["crowding_penalty"] == DEFAULT_CROWDING_PENALTY
    assert p["limit"] == TARGET_CANDIDATES
    assert len(out) == 40 and not out.query_widened
    assert out.retrieval_ms >= 0
    print(f"  retrieval: {len(out)} candidates, parameters correct")


def test_widening_is_stepwise_and_reported():
    """Too few results must widen the query AND say so."""
    thin = [make_row(i) for i in range(3)]
    wide = [make_row(i) for i in range(30)]
    s = FakeSession([thin, wide, wide, wide])
    out = CandidateRetriever(s).retrieve(request())

    assert out.query_widened is True
    assert "prominence floor lowered" in (out.widening_reason or "")
    assert len(s.calls) >= 2, "widening should issue a second query"
    assert s.calls[1]["params"]["prominence_floor"] == 0.0
    print(f"  widening: reported as '{out.widening_reason}'")


def test_no_widening_when_enough_results():
    rows = [make_row(i) for i in range(MIN_CANDIDATES + 5)]
    s = FakeSession([rows])
    out = CandidateRetriever(s).retrieve(request())
    assert out.query_widened is False
    assert len(s.calls) == 1, "must not widen when the first query suffices"
    print("  widening: not triggered when results are sufficient")


def test_invalid_graph_rows_are_skipped_not_fatal():
    rows = [make_row(1), make_row(2, open_min=1020, close_min=480), make_row(3)]
    out = CandidateRetriever(FakeSession([rows])).retrieve(request())
    assert len(out) == 2, "one malformed row should be skipped, not abort retrieval"
    print("  robustness: malformed graph row skipped, retrieval continues")


def test_diagnostics_populated():
    rows = ([make_row(i, district="Kandy") for i in range(10)]
            + [make_row(100 + i, district="Matale", pop=0.1) for i in range(10)])
    r = CandidateRetriever(FakeSession([rows]))
    out = r.retrieve(request())
    d = r.diagnostics
    assert d.district_counts == {"Kandy": 10, "Matale": 10}
    assert d.candidates_after_filter == 20
    assert 0 <= d.mean_popularity <= 1
    assert out.districts_covered == 2
    print(f"  diagnostics: {d.district_counts}, "
          f"mean popularity {d.mean_popularity:.2f}")


def test_travel_times_keyed_consistently():
    rows = [{"from_id": "node/2", "to_id": "node/1", "duration_min": 30.0,
             "distance_km": 12.0, "method": "osrm"}]
    cset = CandidateSet(request_id="x",
                        candidates=[Candidate(**make_row(1)), Candidate(**make_row(2))])
    tt = CandidateRetriever(FakeSession([rows])).travel_times(cset)
    assert ("node/1", "node/2") in tt, "pair keys must be ordered consistently"
    assert tt[("node/1", "node/2")] == 30.0
    print("  travel times: pair keys normalised to sorted order")


# ---------------------------------------------------------------------------
# Loader / retrieval contract
# ---------------------------------------------------------------------------
def test_loader_writes_every_property_retrieval_queries():
    """
    Regression guard.

    The Cypher loader was written before `prominence` and `rail_access_score`
    existed. The graph therefore held 2,245 POIs with no prominence property,
    and every candidate query returned zero rows while every unit test passed -
    because the tests used a fake session and never touched the real schema.

    This test compares the properties the loader writes against the properties
    the retrieval query reads, so the two can never drift apart again.
    """
    import re
    from etl.load_neo4j import LOAD_POIS
    from retrieval.service import CANDIDATE_QUERY

    written = (set(re.findall(r"p\.(\w+)\s*=", LOAD_POIS))
               | set(re.findall(r"MERGE \(p:POI \{(\w+):", LOAD_POIS)))
    read = set(re.findall(r"p\.(\w+)", CANDIDATE_QUERY))

    missing = read - written
    assert not missing, (
        f"CANDIDATE_QUERY reads {sorted(missing)} but LOAD_POIS never writes "
        f"them. Every retrieval would return zero rows.")
    print(f"  loader/retrieval contract: all {len(read)} queried properties are written")


if __name__ == "__main__":
    print("Travora - retrieval tests\n" + "-" * 58)
    for fn in [test_trip_request_weights_and_budget,
               test_candidate_rejects_impossible_hours,
               test_candidate_set_rejects_duplicates,
               test_score_vector_rejects_invented_ids,
               test_score_vector_rejects_omitted_ids,
               test_score_vector_rejects_out_of_range,
               test_score_vector_accepts_exact_match,
               test_ranking_penalises_known_crowding_only,
               test_famous_crowded_site_remains_reachable,
               test_unknown_popularity_is_not_rewarded_as_uncrowded,
               test_retrieve_builds_correct_parameters,
               test_widening_is_stepwise_and_reported,
               test_no_widening_when_enough_results,
               test_invalid_graph_rows_are_skipped_not_fatal,
               test_diagnostics_populated,
               test_travel_times_keyed_consistently,
               test_loader_writes_every_property_retrieval_queries]:
        fn()
    print("-" * 58 + "\nAll tests passed.")