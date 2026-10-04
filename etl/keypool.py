"""
Groq API key pool with rotation and cooldown.

Why a pool
----------
Free-tier Groq limits are per key and per minute: 8,000 tokens per minute on the
gpt-oss models. Scoring 150 candidates takes six batched calls of roughly 1,600
tokens each, about 9,600 tokens, so a single key exceeds its own minute budget
part-way through and the whole scoring pass fails.

Rotating across keys spreads the load so each key stays under its own limit.
When a key is rate-limited it is put on cooldown and the next key takes over;
only when every key is cooling down does the pool report failure - and it
reports it rather than silently degrading.

Configuration (in .env, never in the shell history)
---------------------------------------------------
    GROQ_API_KEYS=gsk_aaa,gsk_bbb,gsk_ccc
  or
    GROQ_API_KEY_1=gsk_aaa
    GROQ_API_KEY_2=gsk_bbb
  or a single
    GROQ_API_KEY=gsk_aaa
"""
from __future__ import annotations
import logging
import os
import re
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

DEFAULT_COOLDOWN_S = 65.0     # a little over the one-minute TPM window
MAX_KEY_INDEX = 20


def load_keys() -> list[str]:
    """Collect keys from the environment, de-duplicated, order preserved."""
    keys: list[str] = []

    bulk = os.getenv("GROQ_API_KEYS", "")
    if bulk:
        keys.extend(k.strip() for k in re.split(r"[,\s;]+", bulk) if k.strip())

    for i in range(1, MAX_KEY_INDEX + 1):
        v = os.getenv(f"GROQ_API_KEY_{i}", "").strip()
        if v:
            keys.append(v)

    single = os.getenv("GROQ_API_KEY", "").strip()
    if single:
        keys.append(single)

    seen, unique = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


def mask(key: str) -> str:
    """Never log a key in full."""
    return f"{key[:7]}...{key[-4:]}" if len(key) > 12 else "***"


@dataclass
class KeyState:
    key: str
    cooldown_until: float = 0.0
    calls: int = 0
    rate_limits: int = 0

    @property
    def available(self) -> bool:
        return time.monotonic() >= self.cooldown_until


@dataclass
class GroqKeyPool:
    keys: list[KeyState] = field(default_factory=list)
    cooldown_s: float = DEFAULT_COOLDOWN_S
    _cursor: int = 0

    @classmethod
    def from_env(cls, cooldown_s: float = DEFAULT_COOLDOWN_S) -> "GroqKeyPool":
        keys = load_keys()
        if keys:
            log.info("Groq key pool: %d key(s) loaded [%s]",
                     len(keys), ", ".join(mask(k) for k in keys))
        return cls(keys=[KeyState(k) for k in keys], cooldown_s=cooldown_s)

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def available_count(self) -> int:
        return sum(1 for k in self.keys if k.available)

    def next_key(self) -> KeyState | None:
        """Round-robin over keys that are not cooling down."""
        n = len(self.keys)
        for offset in range(n):
            state = self.keys[(self._cursor + offset) % n]
            if state.available:
                self._cursor = (self._cursor + offset + 1) % n
                return state
        return None

    def penalise(self, state: KeyState, seconds: float | None = None) -> None:
        state.cooldown_until = time.monotonic() + (seconds or self.cooldown_s)
        state.rate_limits += 1
        log.warning("Key %s rate-limited; cooling down %.0fs (%d/%d keys still available)",
                    mask(state.key), seconds or self.cooldown_s,
                    self.available_count, len(self.keys))

    def seconds_until_any_available(self) -> float:
        if not self.keys:
            return 0.0
        if self.available_count:
            return 0.0
        soonest = min(k.cooldown_until for k in self.keys)
        return max(0.0, soonest - time.monotonic())

    def stats(self) -> dict:
        return {
            "keys": len(self.keys),
            "available": self.available_count,
            "calls": sum(k.calls for k in self.keys),
            "rate_limits": sum(k.rate_limits for k in self.keys),
        }


def is_rate_limit(error: Exception) -> bool:
    """
    Distinguish a rate limit from a real fault.

    Groq reports token-per-minute exhaustion as HTTP 413 with code
    `rate_limit_exceeded`, not only as 429, so matching on the status code alone
    misses it and the pool would never rotate.
    """
    msg = str(error).lower()
    return any(t in msg for t in
               ("rate_limit_exceeded", "rate limit", "429",
                "too many requests", "tokens per minute", "413"))


def retry_after_seconds(error: Exception) -> float | None:
    """Honour an explicit retry hint when Groq supplies one."""
    m = re.search(r"try again in ([\d.]+)s", str(error), re.I)
    return float(m.group(1)) + 1.0 if m else None
