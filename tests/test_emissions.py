"""Journey emissions: factor use, sharing, the trip score, and its effect on plans."""
from __future__ import annotations
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.models import Candidate
from solver.emissions import (kg_per_traveller, journey_score, trip_sustainability,
                              reference_kg_per_day, factors)
from solver.models import SolverConfig
from solver.search import solve
from evaluation.metrics import trip_emissions


def test_every_factor_has_a_bounded_range():
    for mode, f in factors().items():
        assert f["low"] <= f["central"] <= f["high"], mode
        assert f["basis"] in ("pkm", "vkm"), mode


def test_shared_vehicle_is_divided_not_multiplied():
    """Two people in one tuk-tuk each carry half of it; the old code doubled it."""
    one, two = kg_per_traveller("tuktuk", 10, 1), kg_per_traveller("tuktuk", 10, 2)
    assert abs(two - one / 2) < 1e-9
    # Four travellers need two tuk-tuks (capacity 3), so per person rises again.
    assert kg_per_traveller("tuktuk", 10, 4) > kg_per_traveller("tuktuk", 10, 3)


def test_public_transport_is_per_passenger():
    assert kg_per_traveller("bus", 50, 1) == kg_per_traveller("bus", 50, 4)


def test_journey_score_is_bounded_and_monotonic():
    assert journey_score(0.0) == 1.0
    assert journey_score(reference_kg_per_day() * 2) == 0.0
    assert journey_score(2.0) > journey_score(10.0)


def test_trip_score_weights_places_and_journey():
    assert trip_sustainability(1.0, 0.0) == 1.0
    assert abs(trip_sustainability(1.0, reference_kg_per_day()) - 0.6) < 1e-6


def cand(i, lat, lon, district):
    return Candidate(
        poi_id=f"node/{i}", name=f"Place {i}", category="heritage", lat=lat, lon=lon,
        district=district, open_min=480, close_min=1020, entry_fee_lkr=0.0,
        typical_dwell_min=60, popularity_index=0.2, prominence=0.6,
        rail_access_score=0.5, hours_estimated=True, fee_estimated=True,
        distance_from_start_km=5.0)


def test_sustainability_weight_pulls_plans_closer():
    """
    With the journey in the objective, a sustainability-first request stays near
    the start while an interest-only request goes further for better places.
    """
    near = [cand(i, 7.30 + 0.004 * i, 80.64 + 0.004 * i, "Kandy") for i in range(4)]
    far = [cand(100 + i, 7.80 + 0.004 * i, 80.66 + 0.004 * i, "Matale") for i in range(4)]
    pref = {c.poi_id: 0.55 for c in near} | {c.poi_id: 0.95 for c in far}
    sust = {c.poi_id: 0.6 for c in near + far}
    cfg = replace(SolverConfig(max_slots_per_day=3, min_stops_per_day=2, time_budget_s=5.0),
                  travel_cost_per_hour=0.0)
    start = (7.2906, 80.6337)
    interest = solve("i", near + far, 1, 1e6, pref, sust, 1.0, {}, cfg, start=start)
    green = solve("g", near + far, 1, 1e6, pref, sust, 0.0, {}, cfg, start=start)
    assert {s.district for s in interest.itinerary.stops} == {"Matale"}
    assert {s.district for s in green.itinerary.stops} == {"Kandy"}


def test_verifier_measures_every_leg_including_the_start():
    class Req:
        party_size, start_lat, start_lon = 2, 7.2906, 80.6337
    a, b = cand(1, 7.80, 80.66, "Matale"), cand(2, 7.81, 80.67, "Matale")
    e = trip_emissions([[a.poi_id, b.poi_id]], Req(), {a.poi_id: a, b.poi_id: b})
    assert set(e) == {"low", "central", "high"}
    assert e["low"] <= e["central"] <= e["high"]
    assert e["central"] > 0, "the transfer from the starting point must be counted"
