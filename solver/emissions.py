"""
Journey emissions and the journey part of the sustainability score.

Activity-based method: emissions = distance x mode factor. Public transport
factors are per passenger-km. Tuk-tuks and cars are per vehicle-km and are
divided among the people in the vehicle, so two travellers sharing a tuk-tuk
each carry half of it; multiplying a per-passenger figure by party size, as an
earlier version did, counted a shared vehicle once per passenger.

Factors and their sources live in config/sustainability.yaml; the documents are
in data/references/emission_factors/.
"""
from __future__ import annotations
import math
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parent.parent / "config" / "sustainability.yaml"
SCENARIOS = ("low", "central", "high")


@lru_cache(maxsize=1)
def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def factors() -> dict:
    return _config()["transport_emissions"]


def journey_params() -> dict:
    return _config()["journey"]


def kg_per_traveller(mode: str, km: float, party: int = 1,
                     scenario: str = "central") -> float:
    """Emissions one traveller is responsible for on one leg."""
    f = factors().get(mode)
    if f is None or km <= 0:
        return 0.0
    value = f[scenario]
    party = max(int(party), 1)
    if f["basis"] == "pkm":
        return km * value
    if "occupancy" in f:                       # shared with other travellers
        return km * value / f["occupancy"]
    vehicles = math.ceil(party / f.get("capacity", 4))
    return km * value * vehicles / party


def estimated_mode(km: float) -> str:
    """
    Mode assumed while searching, before rail and measured roads are known.
    It never assumes the train, so the search estimate is conservative.
    """
    if km <= 1.5:
        return "walk"
    if km <= 6.0:
        return "tuktuk"
    if km < 60.0:
        return "bus"
    return "car_shared"


def reference_kg_per_day(scenario: str = "central") -> float:
    p = journey_params()
    return kg_per_traveller(p["reference_mode"], p["reference_km_per_day"], 1, scenario)


def journey_score(kg_per_traveller_day: float, scenario: str = "central") -> float:
    """1 = no motorised travel; 0 = the most travel the planner's rules allow."""
    ref = reference_kg_per_day(scenario)
    return round(1.0 - min(1.0, kg_per_traveller_day / ref), 4) if ref > 0 else 1.0


def trip_sustainability(place_mean: float, kg_per_traveller_day: float,
                        scenario: str = "central") -> float:
    lam = journey_params()["lambda"]
    return round((1 - lam) * place_mean + lam * journey_score(kg_per_traveller_day, scenario), 4)
