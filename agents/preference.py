"""
Preference Agent - the only component in the system that calls a language model.

What it may and may not do
--------------------------
MAY   score a given list of candidates 0-1 against the traveller's stated
      interests.
MAY NOT add a place, remove a place, reorder anything, or decide the itinerary.

That restriction is the fix for the defect the supervisor identified. The old
system asked a model to "generate an itinerary", so it emitted the destinations
in its own priors - Sigiriya, Kandy, Ella, Galle - in roughly that order every
time, regardless of budget or interests. Output that looked agentic was a
template.

Here the model never sees a decision. It receives opaque candidate IDs with
attributes and returns a score per ID. ScoreVector rejects any response
containing an ID that was not supplied, so a hallucinated place cannot reach the
solver even in principle.

Prompt policy (plan section 6)
------------------------------
  * no place names anywhere in the prompt
  * no example itineraries - few-shot demonstrations leak into output
  * candidates injected as structured JSON with opaque ids
  * response schema enforced; one retry; then fail loud or fall back explicitly
"""
from __future__ import annotations
import json
import logging
import re
import time
from typing import Callable, Optional, Protocol

from retrieval.models import Candidate, ScoreVector, Category

log = logging.getLogger(__name__)

PROMPT_VERSION = "pref-v1"

SYSTEM_PROMPT = """You score travel destinations for a traveller.

You will receive:
  - the traveller's stated interests and trip context
  - a JSON list of candidate destinations, each with an opaque "id"

For EVERY candidate id, return a relevance score from 0.0 to 1.0 measuring how
well that candidate matches the stated interests.

Rules you must follow exactly:
  1. Return a score for every id you were given. Omit none.
  2. Never invent an id. Only ids from the input may appear in your output.
  3. Never suggest a destination that is not in the input list.
  4. Respond with JSON only, no prose, no markdown fences, in this shape:
     {"scores": {"<id>": 0.0, "<id>": 0.0}}

Score on match to the stated interests, not on fame. A well-known place is not
automatically a better match."""


class Completion(Protocol):
    """Any callable that takes (system, user) and returns the model's text."""
    def __call__(self, system: str, user: str) -> str: ...


# ---------------------------------------------------------------------------
def _candidate_payload(c: Candidate) -> dict:
    """
    What the model sees. Deliberately excludes name and district: scoring should
    follow the attributes, and a recognisable name invites the model to fall
    back on prior knowledge of the place instead of reading the data.
    """
    return {
        "id": c.poi_id,
        "category": c.category.value,
        "entry_fee_lkr": round(c.entry_fee_lkr),
        "typical_visit_minutes": c.typical_dwell_min,
        "reachable_by_rail": round(c.rail_access_score, 2),
        "distance_from_start_km": round(c.distance_from_start_km, 1),
    }


def build_user_prompt(candidates: list[Candidate], interests: list[Category],
                      days: int, budget_lkr: float, party_size: int) -> str:
    payload = [_candidate_payload(c) for c in candidates]
    return (
        f"Trip context:\n"
        f"  interests: {', '.join(i.value for i in interests)}\n"
        f"  duration_days: {days}\n"
        f"  total_budget_lkr: {round(budget_lkr)}\n"
        f"  party_size: {party_size}\n\n"
        f"Candidates ({len(payload)}):\n{json.dumps(payload, separators=(',', ':'))}\n\n"
        f"Return JSON with a score for all {len(payload)} ids."
    )


def parse_scores(raw: str) -> dict[str, float]:
    """Extract the scores object, tolerating fences and surrounding prose."""
    text = (raw or "").strip()
    if not text:
        raise ValueError(
            "model returned an empty response - usually the completion budget "
            "was consumed by reasoning tokens; lower reasoning_effort or raise "
            "max_completion_tokens")
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model response")
    data = json.loads(text[start:end + 1])
    scores = data.get("scores", data)
    if not isinstance(scores, dict):
        raise ValueError("'scores' is not an object")
    return {str(k): float(v) for k, v in scores.items()}


# ---------------------------------------------------------------------------
class HeuristicPreferenceAgent:
    """
    Deterministic fallback. Used when no model is configured, and when the model
    fails validation twice.

    Its outputs are marked `used_llm=False` so a run can never silently pass off
    heuristic scores as model reasoning.
    """

    agent_name = "preference"
    used_llm = False

    def score(self, candidates: list[Candidate], interests: list[Category],
              **_) -> ScoreVector:
        wanted = {i.value for i in interests}
        scores = {}
        for c in candidates:
            base = 0.75 if c.category.value in wanted else 0.25
            # prominence nudges within a category band, never across it
            scores[c.poi_id] = round(min(1.0, base + 0.25 * c.prominence), 4)
        return ScoreVector(agent=self.agent_name,
                           candidate_ids=[c.poi_id for c in candidates],
                           scores=scores)


class PreferenceAgent:
    """LLM-backed scorer with schema validation, one retry, then fallback."""

    agent_name = "preference"

    def __init__(self, complete: Optional[Completion] = None,
                 batch_size: int = 20, max_retries: int = 1):
        self.complete = complete
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.used_llm = False
        self.failures: list[str] = []
        self._fallback = HeuristicPreferenceAgent()

    def _score_batch(self, batch: list[Candidate], interests: list[Category],
                     days: int, budget_lkr: float, party_size: int) -> dict[str, float]:
        user = build_user_prompt(batch, interests, days, budget_lkr, party_size)
        ids = [c.poi_id for c in batch]
        last_error = ""

        for attempt in range(self.max_retries + 1):
            raw = self.complete(SYSTEM_PROMPT, user)
            try:
                scores = parse_scores(raw)
                # The contract. Rejects invented ids, omissions and bad ranges.
                ScoreVector(agent=self.agent_name, candidate_ids=ids, scores=scores)
                return scores
            except Exception as e:                      # noqa: BLE001
                last_error = str(e)
                log.warning("Preference batch attempt %d/%d rejected: %s",
                            attempt + 1, self.max_retries + 1, last_error[:200])
                user += ("\n\nYour previous response was rejected: "
                         f"{last_error[:200]}\nReturn JSON with exactly the "
                         f"{len(ids)} ids given, each scored 0.0-1.0.")

        raise ValueError(f"preference scoring failed validation: {last_error}")

    def score(self, candidates: list[Candidate], interests: list[Category],
              days: int = 7, budget_lkr: float = 0.0,
              party_size: int = 1) -> ScoreVector:
        if self.complete is None:
            log.info("No model configured; using the deterministic fallback")
            return self._fallback.score(candidates, interests)

        merged: dict[str, float] = {}
        try:
            for i in range(0, len(candidates), self.batch_size):
                batch = candidates[i:i + self.batch_size]
                merged.update(self._score_batch(batch, interests, days,
                                                budget_lkr, party_size))
            self.used_llm = True
        except Exception as e:                          # noqa: BLE001
            # Fail visibly, not silently. A run that fell back must say so.
            self.failures.append(str(e))
            log.error("Preference agent failed (%s); falling back to the "
                      "deterministic scorer. This run is marked used_llm=False.", e)
            self.used_llm = False
            return self._fallback.score(candidates, interests)

        return ScoreVector(agent=self.agent_name,
                           candidate_ids=[c.poi_id for c in candidates],
                           scores=merged)


# ---------------------------------------------------------------------------
# Groq retired llama-3.3-70b-versatile and llama-3.1-8b-instant on 16 August
# 2026 and recommends the gpt-oss family in their place. Hardcoding any single
# id will break again the next time a model is retired, so the preferred id is
# tried first and the live /models endpoint is consulted on failure.
GROQ_PREFERRED = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
]


def groq_list_models(api_key: str) -> list[str]:
    """Active model ids from GroqCloud, or [] if the call fails."""
    try:
        import requests
        r = requests.get("https://api.groq.com/openai/v1/models",
                         headers={"Authorization": f"Bearer {api_key}"}, timeout=20)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", []) if m.get("id")]
    except Exception as e:                              # noqa: BLE001
        log.warning("Could not list Groq models: %s", e)
        return []


def pick_groq_model(api_key: str, requested: Optional[str] = None) -> Optional[str]:
    """
    Choose a usable chat model.

    Order: the explicitly requested id, then the preferred list, then anything
    active that looks like a chat model. Audio, guard and embedding models are
    excluded because they cannot serve chat completions.
    """
    available = groq_list_models(api_key)
    if not available:
        return requested or GROQ_PREFERRED[0]

    if requested:
        if requested in available:
            return requested
        log.warning("Requested model %r is not active on this account", requested)

    for m in GROQ_PREFERRED:
        if m in available:
            return m

    skip = ("whisper", "orpheus", "guard", "tts", "embedding", "prompt-guard")
    for m in available:
        if not any(t in m.lower() for t in skip):
            log.info("Falling back to the first active chat model: %s", m)
            return m

    log.error("No usable chat model found. Active: %s", available[:10])
    return None


def groq_completion(model: Optional[str] = None,
                    temperature: float = 0.2,
                    cooldown_s: float = 65.0) -> Optional[Completion]:
    """
    Build a Groq-backed completion callable backed by a rotating key pool.

    Free-tier limits are per key and per minute (8,000 TPM on the gpt-oss
    models), and scoring 150 candidates needs roughly 9,600 tokens in total, so
    one key cannot complete a scoring pass alone. The pool spreads calls across
    keys and cools down any key that reports a rate limit.

    Returns None when no key is configured, in which case the deterministic
    scorer runs and the audit trail records used_llm=False.
    """
    from .keypool import GroqKeyPool, is_rate_limit, retry_after_seconds, mask

    pool = GroqKeyPool.from_env(cooldown_s=cooldown_s)
    if not len(pool):
        log.info("No Groq key configured; the deterministic scorer will be used")
        return None
    try:
        from groq import Groq
    except ImportError:
        log.warning("groq package not installed: pip install groq")
        return None

    resolved = pick_groq_model(pool.keys[0].key, model)
    if not resolved:
        return None
    log.info("Preference Agent model: %s across %d key(s)", resolved, len(pool))

    clients: dict[str, object] = {}

    def client_for(key: str):
        if key not in clients:
            clients[key] = Groq(api_key=key)
        return clients[key]

    # The gpt-oss family are REASONING models: at default effort they spend the
    # completion budget on internal reasoning and return empty content. Scoring
    # a list needs no deliberation, so reasoning is set low.
    is_reasoning = any(t in resolved.lower()
                       for t in ("gpt-oss", "qwen3", "deepseek"))

    # Deliberately small. max_completion_tokens counts toward the per-minute
    # token budget, so an 8,000 budget alone consumed the entire 8,000 TPM
    # allowance and every request failed with HTTP 413.
    MAX_OUT = 1200

    def _once(key: str, system: str, user: str, json_mode: bool) -> str:
        kwargs = dict(
            model=resolved, temperature=temperature,
            max_completion_tokens=MAX_OUT,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        if is_reasoning:
            kwargs["reasoning_effort"] = "low"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client_for(key).chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def complete(system: str, user: str) -> str:
        last: Optional[Exception] = None

        for _ in range(len(pool) * 2):
            state = pool.next_key()
            if state is None:
                wait = pool.seconds_until_any_available()
                if wait <= 0 or wait > 90:
                    break
                log.info("All %d keys cooling down; waiting %.0fs", len(pool), wait)
                time.sleep(wait)
                continue

            state.calls += 1
            for json_mode in (True, False):
                try:
                    return _once(state.key, system, user, json_mode)
                except Exception as e:                  # noqa: BLE001
                    last = e
                    if is_rate_limit(e):
                        pool.penalise(state, retry_after_seconds(e))
                        break                            # rotate to the next key
                    if json_mode and any(t in str(e) for t in
                                         ("response_format", "json_validate_failed")):
                        log.warning("Retrying without JSON mode on %s",
                                    mask(state.key))
                        continue                         # same key, no JSON mode
                    raise

        stats = pool.stats()
        raise RuntimeError(
            f"All Groq keys exhausted ({stats['rate_limits']} rate limits across "
            f"{stats['keys']} key(s)). Add more keys to GROQ_API_KEYS or wait for "
            f"the per-minute window to reset. Last error: {last}")

    complete.pool = pool          # exposed for the audit trail
    complete.model = resolved
    return complete
