"""Free-text trip parsing: rule parser, validation bounds, and LLM fallback."""
import json

from agents.parse import parse_rules, parse_trip_text, validate


def test_rules_extracts_all_fields():
    r = parse_rules("5 days, budget 100k LKR, 2 people, love temples and safari")
    assert r["days"] == 5
    assert r["budget_lkr"] == 100_000
    assert r["party_size"] == 2
    assert set(r["interests"]) == {"religious", "wildlife"}


def test_rules_lakh_and_week():
    r = parse_rules("A week in the hills with my wife, Rs 1.5 lakh total")
    assert r["days"] == 7
    assert r["budget_lkr"] == 150_000
    assert r["party_size"] == 2
    assert "nature" in r["interests"]


def test_rules_leaves_unstated_fields_empty():
    r = parse_rules("somewhere with beaches")
    assert r["days"] is None and r["budget_lkr"] is None and r["party_size"] is None
    assert r["interests"] == ["beach"]


def test_validate_drops_out_of_range_and_unknown():
    fields, notes = validate({"days": 40, "budget_lkr": -5, "party_size": 3,
                              "interests": ["beach", "casinos", "beach"]})
    assert fields["days"] is None
    assert fields["budget_lkr"] is None
    assert fields["party_size"] == 3
    assert fields["interests"] == ["beach"]
    assert any("casinos" in n for n in notes)


def test_llm_output_is_validated_not_trusted():
    fake = lambda system, user: json.dumps(
        {"days": 3, "budget_lkr": 80000, "party_size": None,
         "interests": ["heritage", "Sigiriya Rock"]})
    out = parse_trip_text("3 days of ruins", complete=fake)
    assert out["source"] == "llm"
    assert out["fields"]["interests"] == ["heritage"]
    assert out["missing"] == ["party_size"]


def test_llm_failure_falls_back_to_rules():
    def broken(system, user):
        raise RuntimeError("rate limited")
    out = parse_trip_text("4 days beach trip for 2 people, budget 60k", complete=broken)
    assert out["source"] == "rules"
    assert out["fields"]["days"] == 4
    assert any("rule parser" in n for n in out["notes"])
