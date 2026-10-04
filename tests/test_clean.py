"""
Tests for the Travora ETL cleaning stage.

These assert real behaviour on a fixture containing the edge cases that actually
occur in Sri Lankan OSM data: node/way duplicates, Sinhala names with an English
alternative, USD-denominated fees, multi-rule opening hours, unnamed features,
and records outside the country.

Run:  python -m pytest tests/ -v      (or: python tests/test_clean.py)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from etl.clean import (parse_opening_hours, parse_fee, categorise, pick_name,
                       nearest_district, normalise_elements, deduplicate, impute)
from etl.config import load_categories, FORBIDDEN_PERSIST_FIELDS

FIXTURE = Path(__file__).parent / "fixtures" / "sample_overpass.json"
CFG = load_categories()


def _elements():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["elements"]


# ---------------------------------------------------------------- opening hours
def test_opening_hours():
    assert parse_opening_hours("24/7") == (0, 1440)
    assert parse_opening_hours("Mo-Su 07:00-17:30") == (420, 1050)
    assert parse_opening_hours("06:00-18:00") == (360, 1080)
    # widest window across multiple rules
    assert parse_opening_hours("Mo-Fr 05:00-12:00; Sa-Su 05:00-21:00") == (300, 1260)
    # unparseable -> None so the caller imputes AND flags
    assert parse_opening_hours("sunrise-sunset") is None
    assert parse_opening_hours(None) is None
    assert parse_opening_hours("") is None
    print("  opening hours: 7/7")


# ------------------------------------------------------------------------- fees
def test_fees():
    assert parse_fee({"charge": "5000 LKR"}) == 5000.0
    assert parse_fee({"charge": "1,500 LKR"}) == 1500.0
    assert parse_fee({"fee": "no"}) == 0.0
    assert parse_fee({"charge": "20 USD"}) == 6000.0     # converted, flagged upstream
    assert parse_fee({}) is None                          # unknown != free
    print("  fees: 5/5")


# ------------------------------------------------------------------ categories
def test_categories():
    rules = CFG["rules"]
    assert categorise({"boundary": "protected_area"}, rules) == "wildlife"
    assert categorise({"natural": "beach"}, rules) == "beach"
    assert categorise({"historic": "archaeological_site"}, rules) == "heritage"
    assert categorise({"waterway": "waterfall"}, rules) == "nature"
    assert categorise({"highway": "bus_stop"}, rules) is None
    # ordering matters: protected_area must win over any nature interpretation
    assert categorise({"boundary": "protected_area", "natural": "wood"}, rules) == "wildlife"

    # --- corroboration gating -------------------------------------------
    # A bare village shrine is a real place but not a visitor destination.
    assert categorise({"amenity": "place_of_worship"}, rules) is None, \
        "bare place_of_worship must be rejected without corroboration"
    # With an external identifier or tourism tag it qualifies as religious.
    assert categorise({"amenity": "place_of_worship", "wikidata": "Q1"}, rules) == "religious"
    assert categorise({"amenity": "place_of_worship", "website": "x"}, rules) == "religious"
    # A heritage-listed temple is categorised as heritage: that rule runs first,
    # which is intended - heritage status is the more specific claim.
    assert categorise({"amenity": "place_of_worship", "heritage": "1"}, rules) == "heritage"
    assert categorise({"amenity": "place_of_worship", "tourism": "attraction"}, rules) == "religious"
    # Generic land cover is rejected; a named protected park is not.
    assert categorise({"natural": "wood"}, rules) is None, \
        "generic woodland must be rejected without corroboration"
    assert categorise({"leisure": "park", "tourism": "attraction"}, rules) == "nature"
    print("  categories: 11/11 (incl. corroboration gating)")


# ----------------------------------------------------------------------- names
def test_names():
    assert pick_name({"name:en": "Galle Fort", "name": "\u0dc0"}) == "Galle Fort"
    assert pick_name({"name": "Yala"}) == "Yala"
    assert pick_name({"name": "AB"}) is None       # too short
    assert pick_name({}) is None
    print("  names: 4/4")


# ------------------------------------------------------------------- districts
def test_districts():
    assert nearest_district(9.6615, 80.0255) == "Jaffna"
    assert nearest_district(6.0535, 80.2210) == "Galle"
    assert nearest_district(7.2906, 80.6337) == "Kandy"
    print("  districts: 3/3")


# --------------------------------------------------------------- normalisation
def test_normalise_drops_bad_records():
    df = normalise_elements(_elements(), CFG)
    names = set(df.name)

    assert "Chennai Museum" not in names, "record outside Sri Lanka was kept"
    assert "Bus stop 138" not in names, "non-touristic record was kept"
    assert "No Coordinates Place" not in names, "record without coordinates was kept"
    assert "AB" not in names, "name shorter than 3 characters was kept"
    assert "Galle Fort" in names, "name:en was not preferred over the local name"

    stats = df.attrs["drop_stats"]
    assert stats["out_of_bbox"] >= 1 and stats["no_coords"] >= 1 and stats["no_name"] >= 1
    print(f"  normalisation: {len(df)} kept, dropped {stats}")


# --------------------------------------------------------------- deduplication
def test_dedupe_merges_node_and_way():
    df = normalise_elements(_elements(), CFG)
    before = (df.name.str.contains("Sigiriya")).sum()
    out = deduplicate(df)
    after = (out.name.str.contains("Sigiriya")).sum()

    assert before == 2, f"fixture should contain 2 Sigiriya records, found {before}"
    assert after == 1, f"duplicate Sigiriya records not merged (found {after})"

    # the surviving record must be the richer one (parsed hours + fee + wikidata)
    sig = out[out.name.str.contains("Sigiriya")].iloc[0]
    assert sig.hours_parsed, "merge kept the poorer record"
    assert sig.entry_fee_lkr == 5000.0
    print(f"  deduplication: 2 -> 1 Sigiriya, richer record retained")


# ------------------------------------------------------------------- imputation
def test_bilingual_duplicates_merge():
    """
    Sri Lankan OSM carries bilingual names. "Fort Hammenhiel <tamil>" scored 71
    against "Fort Hammenhiel" - below the 85 threshold - so both survived and
    appeared twice in the output. Latin-script normalisation fixes this.
    """
    from etl.clean import normalise_name
    from rapidfuzz import fuzz
    a = "Fort Hammenhiel \u0b95\u0b9f\u0bb2\u0bcd \u0b95\u0bcb\u0b9f\u0bcd\u0b9f\u0bc8"
    b = "Fort Hammenhiel"
    assert fuzz.token_sort_ratio(a, b) < 85, "fixture no longer reproduces the bug"
    assert fuzz.token_sort_ratio(normalise_name(a), normalise_name(b)) == 100
    assert normalise_name(b) in normalise_name(a)

    els = [
        {"type": "node", "id": 7001, "lat": 9.6100, "lon": 79.8500,
         "tags": {"name": a, "historic": "fort"}},
        {"type": "node", "id": 7002, "lat": 9.6101, "lon": 79.8501,
         "tags": {"name": b, "historic": "fort", "wikidata": "Q5471681"}},
    ]
    out = deduplicate(normalise_elements(els, CFG))
    assert len(out) == 1, f"bilingual duplicate not merged (got {len(out)})"
    assert out.iloc[0]["name"].isascii(), "merge should retain the Latin-script name"
    print("  bilingual dedup: 2 -> 1, Latin-script name retained")


def test_imputation_is_flagged():
    df = impute(deduplicate(normalise_elements(_elements(), CFG)), CFG)

    # Marble Beach has no opening_hours in the fixture -> must be imputed AND flagged
    beach = df[df.name == "Marble Beach"].iloc[0]
    assert beach.hours_estimated, "imputed hours were not flagged"
    assert beach.open_min == 0 and beach.close_min == 1440

    # Sigiriya has real hours -> must NOT be flagged
    sig = df[df.name.str.contains("Sigiriya")].iloc[0]
    assert not sig.hours_estimated, "real parsed hours were incorrectly flagged"

    # absence of certification data must never be recorded as False
    assert set(df.eco_certified.unique()) == {"unknown"}
    assert (df.entry_fee_lkr >= 0).all()
    print(f"  imputation: flagged correctly "
          f"({100 * df.hours_estimated.mean():.0f}% hours imputed in fixture)")


# --------------------------------------------------------------- Tier B guard
def test_tier_b_fields_never_persisted():
    """Plan section 4.0: no commercial API content may reach the stored schema."""
    df = impute(deduplicate(normalise_elements(_elements(), CFG)), CFG)
    leaked = FORBIDDEN_PERSIST_FIELDS & set(df.columns)
    assert not leaked, f"Tier B content leaked into stored data: {leaked}"
    print(f"  tier B guard: no forbidden fields among {len(df.columns)} columns")


if __name__ == "__main__":
    print("Travora ETL - cleaning tests\n" + "-" * 46)
    for fn in [test_opening_hours, test_fees, test_categories, test_names,
               test_districts, test_normalise_drops_bad_records,
               test_dedupe_merges_node_and_way, test_bilingual_duplicates_merge,
               test_imputation_is_flagged,
               test_tier_b_fields_never_persisted]:
        fn()
    print("-" * 46 + "\nAll tests passed.")
