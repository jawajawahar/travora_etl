"""The verifier uses measured travel times where the graph has them, with its own code."""
from __future__ import annotations
import inspect
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.models import Candidate, Category, TripRequest
from evaluation import metrics
from evaluation.metrics import verify, travel_minutes


def cand(i, lat, lon, dwell=150):
    return Candidate(
        poi_id=f"node/{i}", name=f"P{i}", category="beach", lat=lat, lon=lon,
        district="Galle", open_min=0, close_min=1440, entry_fee_lkr=0.0,
        typical_dwell_min=dwell, popularity_index=0.3, prominence=0.6,
        popularity_known=True, rail_access_score=0.5, hours_estimated=True,
        fee_estimated=True, distance_from_start_km=5.0)


def req():
    return TripRequest(start_date=date.today() + timedelta(days=30), days=1,
                       budget_lkr=60000, party_size=2, interests=[Category.BEACH],
                       start_lat=6.03, start_lon=80.21)


# Three long visits, the last two about 60 km apart: by straight-line estimate
# (35 km/h, 1.35 detour) that hop takes about 140 minutes and the day overruns
# 18:00; on the measured road it takes 60 and the day fits.
A, B, C = cand(1, 6.03, 80.22), cand(2, 6.04, 80.23), cand(3, 6.50, 80.55)
BY_ID = {c.poi_id: c for c in (A, B, C)}
PLAN = [[A.poi_id, B.poi_id, C.poi_id]]
KEY = (B.poi_id, C.poi_id)


def test_estimate_alone_overruns_the_day():
    assert travel_minutes(B, C) > 120
    v = verify(PLAN, req(), BY_ID)
    assert v.violations["day_overflow"] == 1
    assert v.legs_measured == 0


def test_a_measured_faster_road_is_honoured():
    v = verify(PLAN, req(), BY_ID, travel={KEY: 60.0})
    assert v.feasible, v.violated
    assert v.legs_measured == 1


def test_a_measured_slower_road_is_still_caught():
    """Measured data can make the check stricter as well as looser."""
    v = verify(PLAN, req(), BY_ID, travel={KEY: 400.0})
    assert v.violations["day_overflow"] == 1


def test_verifier_does_not_use_the_planners_travel_model():
    """Shared data, separate code: no import of the solver's TravelModel."""
    src = inspect.getsource(metrics)
    assert "TravelModel" not in src and "from solver" not in src
