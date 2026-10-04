"""
Tests for the HTTP API and the crew orchestration adapter.

Run:  python -m pytest tests/test_api.py -v
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

import api.main as apimain
from agents.crew import TravoraCrew, AGENT_SPECS, topology, crewai_available
from retrieval.models import TripRequest, Category
from solver.models import SolverConfig
from datetime import date, timedelta


# ---------------------------------------------------------------------------
def rows(n=30, category="heritage"):
    return [{
        "poi_id": f"node/{i}", "name": f"Place {i}", "category": category,
        "lat": 7.29 + i * 0.004, "lon": 80.63 + i * 0.004, "district": "Kandy",
        "open_min": 480, "close_min": 1020, "entry_fee_lkr": 500.0,
        "typical_dwell_min": 90, "popularity_index": 0.3, "prominence": 0.6,
        "popularity_known": i % 3 == 0, "retrieval_score": 0.5,
        "rail_access_score": 0.5, "hours_estimated": True,
        "fee_estimated": True, "distance_from_start_km": 5.0,
    } for i in range(n)]


class FakeSession:
    def __init__(self, poi_rows=None):
        self.poi_rows = poi_rows if poi_rows is not None else rows()

    def run(self, q, **p):
        # Order matters: several queries contain "MATCH (p:POI)", so the more
        # specific ones must be matched first.
        if "count(p) AS total" in q:
            return _Single({"total": len(self.poi_rows)})
        if "count(p)" in q:
            return _Single({"pois": len(self.poi_rows), "accommodation": 10})
        if "SKIP" in q:
            skip, lim = p.get("skip", 0), p.get("limit", 20)
            return [self._place(r) for r in self.poi_rows[skip:skip + lim]]
        if "MATCH (p:POI {poi_id:" in q:
            hit = [r for r in self.poi_rows if r["poi_id"] == p.get("id")]
            return _Single(self._place(hit[0])) if hit else _Single(None)
        if "District" in q and "RETURN d.name" in q:
            return [{"name": "Kandy"}, {"name": "Galle"}]
        if "MATCH (p:POI)" in q and "TRAVEL" not in q:
            return self.poi_rows
        return []

    @staticmethod
    def _place(r):
        return {"poi_id": r["poi_id"], "name": r["name"], "category": r["category"],
                "district": r["district"], "lat": r["lat"], "lon": r["lon"],
                "entry_fee_lkr": r["entry_fee_lkr"], "dwell_min": r["typical_dwell_min"],
                "prominence": r["prominence"], "popularity_index": r["popularity_index"],
                "popularity_known": r["popularity_known"],
                "rail_access_score": r["rail_access_score"],
                "nearest_station_km": 5.0, "open_min": r["open_min"],
                "close_min": r["close_min"],
                "hours_estimated": r["hours_estimated"],
                "fee_estimated": r["fee_estimated"]}

    def __enter__(self): return self
    def __exit__(self, *a): return False
    def close(self): pass


class _Single:
    def __init__(self, d): self._d = d
    def single(self): return self._d
    def __iter__(self): return iter([self._d] if self._d else [])


class FakeDriver:
    def __init__(self, session): self._s = session
    def session(self): return self._s
    def close(self): pass


@pytest.fixture
def client():
    session = FakeSession()
    apimain.STATE["driver"] = FakeDriver(session)
    apimain.STATE["completion"] = None          # deterministic fallback
    with TestClient(apimain.app) as c:
        yield c
    apimain.STATE["driver"] = None


# ================= API =================
def test_health_reports_graph_counts(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["pois"] == 30
    print(f"  health: {body['pois']} POIs, model configured={body['preference_model']}")


def test_meta_gives_the_client_everything_for_a_form(client):
    body = client.get("/api/v1/meta").json()
    assert len(body["categories"]) == 8
    assert "defaults" in body and "limits" in body
    print(f"  meta: {len(body['categories'])} categories, defaults supplied")


def test_itinerary_returns_a_plan(client):
    r = client.post("/api/v1/itinerary", json={
        "days": 2, "budget_lkr": 60000, "interests": ["heritage"],
        "party_size": 2, "max_slots_per_day": 2, "use_llm": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["feasible"] is True
    it = body["itinerary"]
    assert len(it["days"]) == 2
    assert it["total_cost_lkr"] <= 60000 * 0.20
    stop = it["days"][0]["stops"][0]
    for k in ("poi_id", "name", "arrive", "depart", "preference_score",
              "sustainability_score", "estimates"):
        assert k in stop, f"stop is missing {k}"
    assert set(stop["estimates"]) == {"opening_hours", "entry_fee", "travel_time"}
    print(f"  itinerary: {sum(len(d['stops']) for d in it['days'])} stops, "
          f"per-stop provenance present")


def test_infeasible_returns_422_with_a_stated_conflict(client):
    """
    A request that cannot be satisfied must say so, with a reason and a
    suggestion. Returning a plausible plan instead would be the failure this
    whole architecture exists to prevent.
    """
    r = client.post("/api/v1/itinerary", json={
        "days": 5, "budget_lkr": 1000, "interests": ["heritage"],
        "use_llm": False})
    assert r.status_code == 422, r.text
    body = r.json()
    assert body["feasible"] is False
    assert body["conflict"] and body["detail"]
    assert "itinerary" not in body, "an infeasible response must not carry a plan"
    print(f"  infeasible: 422 '{body['conflict']}' with suggestion")


def test_invalid_input_is_rejected(client):
    for payload, why in [
        ({"days": 0, "budget_lkr": 1000, "interests": ["heritage"]}, "days=0"),
        ({"days": 3, "budget_lkr": -5, "interests": ["heritage"]}, "negative budget"),
        ({"days": 3, "budget_lkr": 1000, "interests": []}, "no interests"),
        ({"days": 3, "budget_lkr": 1000, "interests": ["spelunking"]}, "bad category"),
        ({"days": 3, "budget_lkr": 1000, "interests": ["heritage"],
          "start_lat": 13.08}, "outside Sri Lanka"),
    ]:
        r = client.post("/api/v1/itinerary", json=payload)
        assert r.status_code == 422, f"{why} should be rejected, got {r.status_code}"
    print("  validation: 5 malformed requests rejected")


def test_audit_trail_is_returned(client):
    body = client.post("/api/v1/itinerary", json={
        "days": 2, "budget_lkr": 60000, "interests": ["heritage"],
        "max_slots_per_day": 2, "use_llm": False}).json()
    a = body["audit"]
    for k in ("candidates_retrieved", "preference_used_llm",
              "ac3_domain_reduction_pct", "search_nodes", "total_ms"):
        assert k in a, f"audit missing {k}"
    assert a["preference_used_llm"] is False, "fallback must be reported as such"
    assert "data_provenance" in body
    print(f"  audit: {a['candidates_retrieved']} candidates, "
          f"used_llm={a['preference_used_llm']}")


def test_graph_unavailable_returns_503(monkeypatch):
    # Without clearing the URI, startup builds a real driver from .env and the
    # result depends on whether the live database happens to be up.
    monkeypatch.setattr(apimain, "NEO4J_URI", "")
    apimain.STATE["driver"] = None
    with TestClient(apimain.app) as c:
        assert c.get("/health").status_code == 503
    print("  degraded: 503 when the graph is unavailable")


# ================= CREW =================
def test_topology_declares_one_llm_and_one_selector():
    """
    The architectural claim, asserted rather than described: exactly one agent
    calls a language model, and exactly one agent selects.
    """
    t = topology()
    assert len(t) == 4
    llm = [a["name"] for a in t if a["uses_llm"]]
    sel = [a["name"] for a in t if a["may_select"]]
    assert llm == ["preference"], f"expected only preference to use an LLM, got {llm}"
    assert sel == ["decision"], f"expected only decision to select, got {sel}"
    print(f"  topology: LLM={llm}, selector={sel}")


def test_crew_runs_and_reports_per_task():
    crew = TravoraCrew(FakeSession(), solver_config=SolverConfig(
        max_slots_per_day=2, time_budget_s=3.0))
    req = TripRequest(start_date=date.today() + timedelta(days=14), days=2,
                      budget_lkr=60000, party_size=2,
                      interests=[Category.HERITAGE],
                      start_lat=7.29, start_lon=80.63)
    result = crew.kickoff(req)
    report = crew.task_report()
    assert [t["agent"] for t in report] == \
        ["retrieval", "preference", "sustainability", "decision"]
    assert all(t["status"] in ("done", "infeasible") for t in report)
    assert result.audit.candidates_retrieved > 0
    print(f"  crew: 4 tasks reported, crewai_installed={crewai_available()}")


def test_crewai_flag_cannot_change_the_itinerary():
    """
    Enabling CrewAI must not alter results. Execution is delegated to the
    deterministic pipeline either way, which is what makes it safe to enable for
    a demonstration and disable for evaluation.
    """
    cfg = SolverConfig(max_slots_per_day=2, time_budget_s=3.0)
    req = TripRequest(start_date=date.today() + timedelta(days=14), days=2,
                      budget_lkr=60000, party_size=2,
                      interests=[Category.HERITAGE],
                      start_lat=7.29, start_lon=80.63)
    a = TravoraCrew(FakeSession(), solver_config=cfg, use_crewai=False).kickoff(req)
    b = TravoraCrew(FakeSession(), solver_config=cfg, use_crewai=True).kickoff(req)
    assert a.solver.feasible == b.solver.feasible
    if a.solver.feasible:
        assert a.solver.itinerary.poi_ids == b.solver.itinerary.poi_ids, \
            "the CrewAI flag changed the itinerary; it must be presentational only"
    print("  crew: identical itinerary with and without the CrewAI flag")


# ================= PAGINATION =================
def test_places_pagination(client):
    """Paging must be server-side, with honest page metadata."""
    r = client.get("/api/v1/places?page=1&size=10")
    assert r.status_code == 200, r.text
    b = r.json()
    for k in ("items", "page", "size", "total", "pages", "has_next", "has_prev"):
        assert k in b, f"missing {k}"
    assert b["page"] == 1 and b["size"] == 10
    assert b["has_prev"] is False
    item = b["items"][0]
    for k in ("poi_id", "name", "category", "district", "estimates"):
        assert k in item, f"item missing {k}"
    print(f"  places: page 1 of {b['pages']}, {b['total']} total, "
          f"{len(b['items'])} returned")


def test_places_size_is_capped(client):
    """One request must not be able to pull the whole graph."""
    b = client.get("/api/v1/places?size=5000").json()
    assert b["size"] <= 100, f"size cap not enforced, got {b['size']}"
    print(f"  places: size capped at {b['size']}")


def test_places_page_bounds_are_normalised(client):
    b = client.get("/api/v1/places?page=0&size=0").json()
    assert b["page"] == 1 and b["size"] >= 1
    print("  places: out-of-range page and size normalised")


def test_place_detail_and_404(client):
    b = client.get("/api/v1/places?size=1").json()
    poi_id = b["items"][0]["poi_id"]
    r = client.get(f"/api/v1/places/{poi_id}")
    assert r.status_code == 200, r.text
    assert r.json()["poi_id"] == poi_id
    assert "estimates" in r.json()
    assert client.get("/api/v1/places/node/does-not-exist").status_code == 404
    print(f"  places: detail for '{poi_id}' resolves; unknown id gives 404")


# ================= TRANSPORT AND LODGING =================
def test_transport_mode_chosen_per_leg():
    """
    Every leg must carry a mode and an emissions figure. The sustainability
    score contains a transport term, so an itinerary that does not say how the
    traveller moves is scoring an intention rather than a journey.
    """
    from solver.logistics import plan_leg
    from solver.emissions import kg_per_traveller
    from solver.routing import rail

    net = rail()
    assert plan_leg((7.2906, 80.6337), (7.2930, 80.6400), (0.8, 3.0), net)["mode"] == "walk"
    assert plan_leg((7.2906, 80.6337), (7.3100, 80.6500), (3.0, 8.0), net)["mode"] == "tuktuk"

    # Colombo Fort to Galle: both at stations on the Coastal Line.
    if net is not None:
        leg = plan_leg((6.9344, 79.8505), (6.0335, 80.2140), (130.0, 180.0), net)
        assert leg["mode"] == "train", "two stations joined by track should go by train"
        assert leg["basis"] == "track" and 100 < leg["km"] < 130

    # Matale to a point north of the end of the Matale line: stations are not
    # joined, so this must NOT be a train (the old rule said it was).
    leg = plan_leg((7.4675, 80.6234), (7.6600, 80.6200), (30.0, 45.0), net)
    assert leg["mode"] != "train"

    # Arugam Bay to Monaragala: no railway at all, long road journey.
    assert plan_leg((6.8400, 81.8360), (6.8720, 81.3500), (70.0, 100.0), net)["mode"] == "car_shared"

    # An unmeasured road leg is estimated and says so.
    est = plan_leg((7.2906, 80.6337), (7.4675, 80.6234), None, net)
    assert est["basis"] in ("estimate", "track") and (est["basis"] == "track" or "estimated" in est["why"])

    # Emissions must rise with the dirtier mode.
    one = lambda m: kg_per_traveller(m, 100.0, 1)
    assert one("train") < one("bus") < one("car_private")
    print("  transport: modes chosen from measured roads and the rail network")


def test_emissions_summary_compares_with_driving():
    from solver.logistics import Leg, emissions_summary
    legs = [Leg("a", "b", 100.0, 120.0, "train", 3500.0, ""),
            Leg("b", "c", 50.0, 60.0, "bus", 3400.0, "")]
    s = emissions_summary(legs, party_size=2)
    assert s["total_km"] == 150.0
    assert s["private_car_kg_co2"] > s["total_kg_co2"], \
        "train and bus must emit less than driving the same distance"
    assert 0 < s["saved_pct"] < 100
    print(f"  emissions: {s['total_kg_co2']} kg vs {s['private_car_kg_co2']} kg "
          f"driving ({s['saved_pct']}% lower)")


def test_lodging_prefers_sustainable_within_budget():
    """A home stay must beat a resort when both are affordable."""
    from solver.logistics import choose_lodging

    class Stop:
        poi_id, lat, lon = "node/1", 7.29, 80.63
    class Day:
        day, stops = 1, [Stop()]
    class Day2:
        day, stops = 2, [Stop()]

    rows = [
        {"acc_id": "a1", "name": "Family Home Stay", "type": "Home Stay Units",
         "district": "Kandy", "rooms": 3, "sustainability_score": 0.95,
         "rail_access_score": 0.5, "location_estimated": False, "distance_km": 2.0},
        {"acc_id": "a2", "name": "Grand Resort", "type": "Classified Hotels( 1-5 Star)",
         "district": "Kandy", "rooms": 200, "sustainability_score": 0.20,
         "rail_access_score": 0.5, "location_estimated": False, "distance_km": 1.0},
    ]

    class S:
        def run(self, q, **p): return rows

    stays, summary = choose_lodging(S(), [Day(), Day2()], budget_lkr=50000)
    assert len(stays) == 1, "one night is needed for a two-day trip"
    assert stays[0].type == "Home Stay Units", \
        "the more sustainable affordable option should be chosen"
    assert stays[0].rate_estimated is True, "nightly rates must be flagged as estimates"
    assert summary["nights_needed"] == 1 and summary["nights_assigned"] == 1
    print(f"  lodging: chose {stays[0].name} "
          f"(sustainability {stays[0].sustainability_score})")


def test_lodging_reports_unaffordable_nights():
    """When nothing affordable is nearby the night is reported, not filled."""
    from solver.logistics import choose_lodging

    class Stop:
        poi_id, lat, lon = "node/1", 7.29, 80.63
    class Day:
        day, stops = 1, [Stop()]
    class Day2:
        day, stops = 2, [Stop()]

    class S:
        def run(self, q, **p):
            return [{"acc_id": "a", "name": "Expensive", "type": "Classified Hotels( 1-5 Star)",
                     "district": "Kandy", "rooms": 200, "sustainability_score": 0.2,
                     "rail_access_score": 0.5, "location_estimated": False,
                     "distance_km": 1.0}]

    stays, summary = choose_lodging(S(), [Day(), Day2()], budget_lkr=100)
    assert stays == []
    assert summary["unassigned_nights"] == [1]
    print("  lodging: unaffordable night reported rather than filled")


if __name__ == "__main__":
    print("Travora - API and crew tests\n" + "-" * 60)
    session = FakeSession()
    apimain.STATE["driver"] = FakeDriver(session)
    apimain.STATE["completion"] = None
    with TestClient(apimain.app) as c:
        for fn in [test_health_reports_graph_counts,
                   test_meta_gives_the_client_everything_for_a_form,
                   test_itinerary_returns_a_plan,
                   test_infeasible_returns_422_with_a_stated_conflict,
                   test_invalid_input_is_rejected,
                   test_audit_trail_is_returned,
                   test_places_pagination,
                   test_places_size_is_capped,
                   test_places_page_bounds_are_normalised,
                   test_place_detail_and_404]:
            fn(c)
    test_graph_unavailable_returns_503()
    test_topology_declares_one_llm_and_one_selector()
    test_crew_runs_and_reports_per_task()
    test_crewai_flag_cannot_change_the_itinerary()
    test_transport_mode_chosen_per_leg()
    test_emissions_summary_compares_with_driving()
    test_lodging_prefers_sustainable_within_budget()
    test_lodging_reports_unaffordable_nights()
    print("-" * 60 + "\nAll tests passed.")
