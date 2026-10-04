"""
Tests for the agent layer.

The decisive test is test_hallucinated_place_cannot_reach_the_solver. That is
the defect the supervisor found - a language model emitting destinations from
its own priors - and it must be structurally impossible, not merely unlikely.

Run:  python -m pytest tests/test_agents.py -v
"""
from __future__ import annotations
import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.models import Candidate, Category, TripRequest, ScoreVector
from agents.preference import (PreferenceAgent, HeuristicPreferenceAgent,
                               SYSTEM_PROMPT, build_user_prompt, parse_scores)
from agents.sustainability import SustainabilityAgent, UNKNOWN_CROWDING_PRIOR


def cand(i, *, category="heritage", pop=0.5, known=True, prom=0.6, rail=0.5):
    return Candidate(
        poi_id=f"node/{i}", name=f"Place {i}", category=category,
        lat=7.29, lon=80.63, district="Kandy", open_min=480, close_min=1020,
        entry_fee_lkr=1000.0, typical_dwell_min=90, popularity_index=pop,
        prominence=prom, popularity_known=known, rail_access_score=rail,
        distance_from_start_km=5.0)


CANDS = [cand(i) for i in range(4)]
INTERESTS = [Category.HERITAGE, Category.WILDLIFE]


# ================= THE CONTRACT =================
def test_hallucinated_place_cannot_reach_the_solver():
    """
    A model that invents a destination must be rejected outright.

    This is the exact failure mode of the previous system: asked to generate an
    itinerary, the model emitted the places in its own priors regardless of the
    data. Here the invented id fails validation on both attempts and the agent
    falls back to the deterministic scorer, so the fabricated place never enters
    the candidate pool the solver chooses from.
    """
    def hallucinating(system, user):
        return json.dumps({"scores": {
            "node/0": 0.8, "node/1": 0.7, "node/2": 0.6, "node/3": 0.5,
            "sigiriya": 0.99,            # never retrieved
        }})

    agent = PreferenceAgent(complete=hallucinating)
    vec = agent.score(CANDS, INTERESTS)

    assert "sigiriya" not in vec.scores, "an invented place reached the solver"
    assert set(vec.scores) == {c.poi_id for c in CANDS}
    assert agent.used_llm is False, "a failed run must not claim it used the model"
    assert agent.failures, "the failure must be recorded, not swallowed"
    print("  contract: invented place rejected, fallback used and flagged")


def test_omitted_candidate_is_rejected():
    def lazy(system, user):
        return json.dumps({"scores": {"node/0": 0.8, "node/1": 0.7}})
    agent = PreferenceAgent(complete=lazy)
    vec = agent.score(CANDS, INTERESTS)
    assert set(vec.scores) == {c.poi_id for c in CANDS}
    assert agent.used_llm is False
    print("  contract: omitted candidates rejected")


def test_valid_response_is_accepted_and_marked():
    def good(system, user):
        payload = json.loads(user[user.find("["):user.rfind("]") + 1])
        return json.dumps({"scores": {p["id"]: 0.6 for p in payload}})
    agent = PreferenceAgent(complete=good)
    vec = agent.score(CANDS, INTERESTS)
    assert agent.used_llm is True
    assert all(v == 0.6 for v in vec.scores.values())
    print("  contract: valid response accepted and marked used_llm=True")


def test_retry_then_success():
    calls = {"n": 0}

    def flaky(system, user):
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json at all"
        return json.dumps({"scores": {c.poi_id: 0.5 for c in CANDS}})

    agent = PreferenceAgent(complete=flaky)
    vec = agent.score(CANDS, INTERESTS)
    assert calls["n"] == 2, "should retry exactly once before succeeding"
    assert agent.used_llm is True
    assert len(vec.scores) == 4
    print("  resilience: malformed response retried, then accepted")


# ================= PROMPT POLICY =================
def test_prompt_contains_no_place_names():
    """
    Root cause RC2: few-shot examples and named places leak into output. The
    prompt must contain no Sri Lankan place name, and candidates must be
    identified only by opaque ids.
    """
    user = build_user_prompt(CANDS, INTERESTS, 7, 120000, 2)
    whole = (SYSTEM_PROMPT + user).lower()
    for name in ("sigiriya", "kandy", "ella", "galle", "dambulla",
                 "colombo", "anuradhapura", "yala"):
        assert name not in whole, f"prompt leaks the place name '{name}'"
    assert "Place 0".lower() not in whole, "candidate names must not be sent"
    assert "node/0" in user, "candidates must be identified by opaque id"
    print("  prompt policy: no place names, opaque ids only")


def test_parse_tolerates_fences_and_prose():
    assert parse_scores('```json\n{"scores":{"a":0.5}}\n```') == {"a": 0.5}
    assert parse_scores('Here you go: {"scores":{"a":1}} thanks') == {"a": 1.0}
    assert parse_scores('{"a":0.25}') == {"a": 0.25}     # bare object
    with pytest.raises(ValueError):
        parse_scores("no json here")
    print("  parsing: tolerates fences and surrounding prose")


def test_no_model_uses_flagged_fallback():
    agent = PreferenceAgent(complete=None)
    vec = agent.score(CANDS, INTERESTS)
    assert agent.used_llm is False
    assert len(vec.scores) == 4
    print("  fallback: no model configured -> deterministic scorer, flagged")


def test_heuristic_prefers_matching_categories():
    cs = [cand(0, category="heritage"), cand(1, category="urban")]
    vec = HeuristicPreferenceAgent().score(cs, [Category.HERITAGE])
    assert vec.scores["node/0"] > vec.scores["node/1"]
    print("  fallback: matching category scores higher")


# ================= SUSTAINABILITY =================
def test_sustainability_is_deterministic():
    """Same input, same output - required for the weight-sensitivity analysis."""
    a = SustainabilityAgent()
    r1 = a.score(CANDS).scores
    r2 = SustainabilityAgent().score(CANDS).scores
    assert r1 == r2, "sustainability scoring must be reproducible"
    print("  sustainability: deterministic across instances")


def test_rail_access_no_longer_counted_at_place_level():
    """
    Transport is measured on the journey now; counting station proximity at the
    place as well would count it twice. Two places differing only in rail
    access must score the same.
    """
    a = SustainabilityAgent()
    near, far = cand(0, rail=1.0), cand(1, rail=0.0)
    s = a.score([near, far]).scores
    assert s[near.poi_id] == s[far.poi_id]
    assert a.explain(near)["weights"]["transport"] == 0.0

def test_crowded_site_scores_lower():
    quiet = cand(0, pop=0.05, known=True)
    busy = cand(1, pop=0.95, known=True)
    a = SustainabilityAgent()
    assert a.score_one(quiet) > a.score_one(busy)
    print(f"  sustainability: quiet {a.score_one(quiet):.3f} > "
          f"crowded {a.score_one(busy):.3f}")


def test_unmeasured_crowding_uses_neutral_prior():
    """
    Absence of a Wikipedia article is not evidence a place is quiet. An
    unmeasured POI must not beat a measured-quiet one.
    """
    a = SustainabilityAgent()
    unmeasured = cand(0, pop=0.0, known=False)
    measured_quiet = cand(1, pop=0.02, known=True)
    assert a.crowding_cost(unmeasured) == UNKNOWN_CROWDING_PRIOR
    assert a.score_one(measured_quiet) > a.score_one(unmeasured), \
        "an unmeasured POI must not outrank a measured-quiet one"
    print(f"  sustainability: unmeasured uses prior {UNKNOWN_CROWDING_PRIOR}, "
          f"does not beat measured-quiet")


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        SustainabilityAgent(w_transport=0.5, w_crowd=0.5, w_eco=0.5)
    print("  sustainability: malformed weights rejected")


def test_explain_breaks_down_components():
    e = SustainabilityAgent().explain(cand(0))
    for k in ("transport_cost", "crowding_cost", "eco_cost", "weights", "score"):
        assert k in e
    assert e["crowding_measured"] is True
    print("  transparency: per-component breakdown available")


def test_both_agents_obey_the_same_contract():
    ids = [c.poi_id for c in CANDS]
    p = PreferenceAgent(complete=None).score(CANDS, INTERESTS)
    s = SustainabilityAgent().score(CANDS)
    for vec in (p, s):
        assert isinstance(vec, ScoreVector)
        assert set(vec.scores) == set(ids)
        assert all(0.0 <= v <= 1.0 for v in vec.scores.values())
    print("  both agents return validated ScoreVectors over the same ids")


# ================= KEY POOL =================
def test_keypool_loads_from_all_env_styles_and_dedupes():
    import os
    from agents.keypool import load_keys, mask
    saved = {k: v for k, v in os.environ.items() if k.startswith("GROQ_API_KEY")}
    for k in list(os.environ):
        if k.startswith("GROQ_API_KEY"):
            del os.environ[k]
    try:
        os.environ["GROQ_API_KEYS"] = "gsk_aaaaaaaaaaaa1111, gsk_bbbbbbbbbbbb2222"
        os.environ["GROQ_API_KEY_1"] = "gsk_cccccccccccc3333"
        os.environ["GROQ_API_KEY"] = "gsk_aaaaaaaaaaaa1111"      # duplicate
        keys = load_keys()
        assert len(keys) == 3, f"expected 3 unique keys, got {len(keys)}"
        assert mask(keys[0]).endswith("1111") and "..." in mask(keys[0])
        assert keys[0] not in mask(keys[0]), "mask must not reveal the whole key"
    finally:
        for k in list(os.environ):
            if k.startswith("GROQ_API_KEY"):
                del os.environ[k]
        os.environ.update(saved)
    print("  keypool: three env styles merged, duplicates removed, keys masked")


def test_keypool_rotates_and_recovers_after_cooldown():
    import time
    from agents.keypool import GroqKeyPool, KeyState
    pool = GroqKeyPool(keys=[KeyState(f"gsk_key{i:012d}") for i in range(3)],
                       cooldown_s=0.4)
    first = [pool.next_key().key for _ in range(3)]
    assert len(set(first)) == 3, "round-robin should visit every key"

    for _ in range(3):
        pool.penalise(pool.next_key(), 0.4)
    assert pool.available_count == 0
    assert pool.next_key() is None, "no key should be offered while all cool down"

    time.sleep(0.5)
    assert pool.available_count == 3, "keys must recover after cooldown"
    assert pool.next_key() is not None
    print("  keypool: rotates across keys, cools down, recovers")


def test_rate_limit_detection_covers_http_413():
    """
    Groq reports token-per-minute exhaustion as HTTP 413 with code
    `rate_limit_exceeded`, not only as 429. Matching on 429 alone meant the pool
    never rotated and the whole scoring pass failed on the first key.
    """
    from agents.keypool import is_rate_limit, retry_after_seconds
    assert is_rate_limit(Exception(
        "Error code: 413 - rate_limit_exceeded ... tokens per minute (TPM): Limit 8000"))
    assert is_rate_limit(Exception("Error code: 429 - Too Many Requests"))
    assert not is_rate_limit(Exception("Error code: 404 - model_not_found"))
    assert not is_rate_limit(Exception("json_validate_failed"))
    assert retry_after_seconds(Exception("Please try again in 7.5s")) == 8.5
    print("  keypool: 413 and 429 detected as rate limits, 404 is not")


if __name__ == "__main__":
    print("Travora - agent tests\n" + "-" * 60)
    for fn in [test_hallucinated_place_cannot_reach_the_solver,
               test_omitted_candidate_is_rejected,
               test_valid_response_is_accepted_and_marked,
               test_retry_then_success,
               test_prompt_contains_no_place_names,
               test_parse_tolerates_fences_and_prose,
               test_no_model_uses_flagged_fallback,
               test_heuristic_prefers_matching_categories,
               test_sustainability_is_deterministic,
               test_rail_access_no_longer_counted_at_place_level,
               test_crowded_site_scores_lower,
               test_unmeasured_crowding_uses_neutral_prior,
               test_weights_must_sum_to_one,
               test_explain_breaks_down_components,
               test_both_agents_obey_the_same_contract,
               test_keypool_loads_from_all_env_styles_and_dedupes,
               test_keypool_rotates_and_recovers_after_cooldown,
               test_rate_limit_detection_covers_http_413]:
        fn()
    print("-" * 60 + "\nAll tests passed.")
