"""
Tests for the SLTDA accommodation ingest and sustainability scoring.

Run:  python -m pytest tests/test_sltda.py -v   (or: python tests/test_sltda.py)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from etl.extract_sltda import (load_config, score_size, score_accommodation,
                               recover_coords, validate_coords, validate_ordering)
from etl.rail import rail_access_score, load_sustainability_config

CFG = load_config()


def _sample() -> pd.DataFrame:
    """Mirrors the real SLTDA schema, including a row with no coordinates."""
    return pd.DataFrame({
        "acc_id": [f"sltda/{i:05d}" for i in range(6)],
        "name": ["Family Homestay", "Village Bungalow", "Beach Guest House",
                 "Grand Resort", "Colonial Heritage Home", "Ungeocoded Homestay"],
        "address": ["a"] * 6,
        "sltda_type": ["Home Stay Units", "Bangalows", "Guest Houses",
                       "Classified Hotels( 1-5 Star)", "Heritage Homes",
                       "Home Stay Units"],
        "rooms": [3, 4, 9, 250, 4, 2],
        "grade": ["STANDARD", "DELUXE", "A", "FIVE", "SUPERIOR", "STANDARD"],
        "district": ["Kandy", "Kandy", "Galle", "Galle", "Kandy", "Kandy"],
        "aga_division": ["Kandy DS", "Kandy DS", "Galle DS", "Galle DS",
                         "Kandy DS", "Kandy DS"],
        "local_authority": ["x"] * 6,
        "lat": [7.29, 7.30, 6.03, 6.04, 7.31, np.nan],
        "lon": [80.63, 80.64, 80.21, 80.22, 80.65, np.nan],
    })


# ------------------------------------------------------------ size scoring
def test_size_score_monotonic():
    rooms = pd.Series([1, 3, 10, 50, 300])
    s = score_size(rooms, CFG)
    assert all(s.iloc[i] > s.iloc[i + 1] for i in range(len(s) - 1)), \
        "size score must decrease as room count rises"
    assert 0.0 <= s.min() and s.max() <= 1.0
    print(f"  size score monotonic: "
          f"{', '.join(f'{r}rm={v:.2f}' for r, v in zip(rooms, s))}")


# ------------------------------------------------ accommodation ordering
def test_homestay_beats_resort():
    df = score_accommodation(_sample().dropna(subset=["lat"]), CFG)
    home = df[df.name == "Family Homestay"].sustainability_score.iloc[0]
    resort = df[df.name == "Grand Resort"].sustainability_score.iloc[0]
    guest = df[df.name == "Beach Guest House"].sustainability_score.iloc[0]

    assert home > guest > resort, \
        f"expected homestay > guesthouse > resort, got {home:.3f}/{guest:.3f}/{resort:.3f}"
    assert 0.0 <= df.sustainability_score.min() <= df.sustainability_score.max() <= 1.0
    # acc_impact is the cost form and must be the complement
    assert np.allclose(df.sustainability_score + df.acc_impact, 1.0, atol=1e-6)
    print(f"  ordering: homestay {home:.3f} > guesthouse {guest:.3f} > resort {resort:.3f}")


def test_unknown_category_is_neutral_not_optimistic():
    df = _sample().dropna(subset=["lat"]).copy()
    df.loc[0, "sltda_type"] = "Some Unmapped Category"
    out = score_accommodation(df, CFG)
    assert out.category_score.iloc[0] == CFG["_default_category_score"]
    assert out.category_score.iloc[0] < 1.0, \
        "an unknown category must never score as well as a homestay"
    print(f"  unknown category scored at neutral default "
          f"{CFG['_default_category_score']}")


# --------------------------------------------------- coordinate recovery
def test_missing_coords_recovered_and_flagged():
    out = recover_coords(validate_coords(_sample()))
    assert len(out) == 6, "no record should be dropped when recovery is possible"

    rec = out[out.name == "Ungeocoded Homestay"].iloc[0]
    assert rec.location_estimated, "recovered location was not flagged"
    assert rec.location_method == "aga_division_centroid"
    assert 7.28 < rec.lat < 7.32, "centroid should sit among the Kandy records"

    real = out[out.name == "Family Homestay"].iloc[0]
    assert not real.location_estimated
    assert real.location_method == "sltda_published"
    print("  coordinate recovery: division centroid applied and flagged")


def test_coords_outside_sri_lanka_discarded():
    df = _sample()
    df.loc[0, ["lat", "lon"]] = [13.08, 80.27]      # Chennai
    out = validate_coords(df)
    assert pd.isna(out.loc[0, "lat"]), "coordinate outside Sri Lanka was kept"
    print("  out-of-country coordinate discarded")


def test_recovery_does_not_bias_against_small_properties():
    """
    The real register geocodes 100% of Classified Hotels but only ~56% of Home
    Stays. Dropping ungeocoded rows would delete the sustainable options, so
    recovery must retain them.
    """
    out = recover_coords(validate_coords(_sample()))
    homestays = (out.sltda_type == "Home Stay Units").sum()
    assert homestays == 2, f"expected both homestays retained, got {homestays}"
    print("  small-property retention preserved through recovery")


# --------------------------------------------------------- rail scoring
def test_rail_access_curve():
    cfg = load_sustainability_config()
    km = np.array([0.0, 2.0, 8.0, 25.0, 60.0])
    s = rail_access_score(km, cfg)
    assert s[0] == 1.0 and s[1] == 1.0, "walkable distance must score 1.0"
    assert s[4] == 0.0, "beyond the moderate threshold must score 0.0"
    assert all(s[i] >= s[i + 1] for i in range(len(s) - 1)), "must be non-increasing"
    print(f"  rail curve: {', '.join(f'{k:.0f}km={v:.2f}' for k, v in zip(km, s))}")


# ------------------------------------------------- ordering validation
def test_ordering_validation_reports_correlation():
    df = score_accommodation(_sample().dropna(subset=["lat"]), CFG)
    res = validate_ordering(df)
    assert "rho" in res and res["rho"] < 0, \
        "category score should correlate negatively with median room count"
    print(f"  ordering validation returns rho = {res['rho']:.3f}")


if __name__ == "__main__":
    print("Travora - SLTDA accommodation tests\n" + "-" * 52)
    for fn in [test_size_score_monotonic, test_homestay_beats_resort,
               test_unknown_category_is_neutral_not_optimistic,
               test_missing_coords_recovered_and_flagged,
               test_coords_outside_sri_lanka_discarded,
               test_recovery_does_not_bias_against_small_properties,
               test_rail_access_curve,
               test_ordering_validation_reports_correlation]:
        fn()
    print("-" * 52 + "\nAll tests passed.")
