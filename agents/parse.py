"""
Free-text trip request -> structured form fields.

This agent fills in a form. It never sees candidate places and never proposes
one, so it cannot introduce an invented destination: its output is validated
against the same bounds as the HTTP request model and then handed to the
solver like any typed request. A rule-based parser runs when no language model
is configured and doubles as the comparison point for parse accuracy.
"""
from __future__ import annotations
import json
import re

from retrieval.models import Category

FIELDS = ("days", "budget_lkr", "party_size", "interests")

SYSTEM_PROMPT = f"""You convert a traveller's message about a Sri Lanka trip into JSON.
Extract only what the message states. Use null for anything not stated; do not guess.

Return exactly:
{{"days": int|null, "budget_lkr": number|null, "party_size": int|null,
  "interests": [string]}}

interests must be chosen only from: {", ".join(c.value for c in Category)}.
Map synonyms (temples -> religious, safari/elephants -> wildlife, hiking -> adventure,
ruins/ancient -> heritage, hills/waterfalls -> nature, city/shopping -> urban).
Budget is total for the trip in Sri Lankan rupees; convert "100k" to 100000 and
"1.5 lakh" to 150000. If a currency other than LKR is used, return null for budget."""

_SYNONYMS = {
    "heritage": ["heritage", "ruin", "ancient", "fort", "historic", "history", "unesco"],
    "wildlife": ["wildlife", "safari", "elephant", "leopard", "whale", "bird", "national park"],
    "nature": ["nature", "waterfall", "hill", "tea", "mountain", "forest", "lake", "garden"],
    "beach": ["beach", "surf", "coast", "sea", "snorkel", "diving"],
    "religious": ["religious", "temple", "kovil", "church", "mosque", "stupa", "dagoba"],
    "adventure": ["adventure", "hike", "hiking", "trek", "rafting", "climb", "zipline"],
    "cultural": ["cultural", "culture", "museum", "dance", "festival", "art", "craft"],
    "urban": ["urban", "city", "shopping", "market", "nightlife", "colombo"],
}


def _money(num: str, unit: str) -> float:
    v = float(num.replace(",", ""))
    unit = (unit or "").lower()
    if unit in ("k", "thousand"):
        v *= 1_000
    elif unit in ("lakh", "lakhs", "lac"):
        v *= 100_000
    elif unit in ("m", "mn", "million"):
        v *= 1_000_000
    return v


def parse_rules(text: str) -> dict:
    t = (text or "").lower()
    out: dict = {"days": None, "budget_lkr": None, "party_size": None, "interests": []}

    m = re.search(r"(\d+)\s*(?:-|\s)?(?:day|days|night|nights)\b", t)
    if m:
        out["days"] = int(m.group(1))
    elif re.search(r"\b(a|one)\s+week\b", t):
        out["days"] = 7
    elif re.search(r"\bweekend\b", t):
        out["days"] = 2

    num = r"(\d[\d,]*(?:\.\d+)?)"
    unit = r"(k|lakhs?|lac|mn|million|thousand|m\b)?"
    m = re.search(rf"\b(?:lkr|rs\.?|rupees?)\s*{num}\s*{unit}"
                  rf"|{num}\s*{unit}\s*(?:lkr|rs\b|rupees?)"
                  rf"|budget\D{{0,12}}{num}\s*{unit}", t)
    if m:
        g = m.groups()
        num, unit = next(((g[i], g[i + 1]) for i in (0, 2, 4) if g[i]), (None, None))
        if num:
            out["budget_lkr"] = _money(num, unit)

    m = re.search(r"(\d+)\s*(?:people|persons|pax|adults|travellers|travelers|of us|friends)", t)
    if m:
        out["party_size"] = int(m.group(1))
    elif re.search(r"\b(solo|alone|myself)\b", t):
        out["party_size"] = 1
    elif re.search(r"\b(couple|my (wife|husband|partner)|two of us)\b", t):
        out["party_size"] = 2

    out["interests"] = [c for c, words in _SYNONYMS.items()
                        if any(re.search(rf"\b{re.escape(w)}", t) for w in words)]
    return out


def _extract_json(raw: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model response")
    return json.loads(text[start:end + 1])


def validate(raw: dict) -> tuple[dict, list[str]]:
    """Clamp to the HTTP model's bounds; drop what cannot be used and say so."""
    notes: list[str] = []
    out: dict = {}

    def as_int(v, lo, hi, name):
        try:
            n = int(round(float(v)))
        except (TypeError, ValueError):
            return None
        if not lo <= n <= hi:
            notes.append(f"{name} {n} is outside {lo}-{hi} and was ignored")
            return None
        return n

    out["days"] = as_int(raw.get("days"), 1, 21, "days")
    out["party_size"] = as_int(raw.get("party_size"), 1, 20, "party size")

    try:
        b = float(raw.get("budget_lkr")) if raw.get("budget_lkr") is not None else None
    except (TypeError, ValueError):
        b = None
    if b is not None and b <= 0:
        notes.append("budget must be above zero and was ignored")
        b = None
    out["budget_lkr"] = b

    valid = {c.value for c in Category}
    given = [str(i).strip().lower() for i in (raw.get("interests") or [])]
    dropped = [i for i in given if i not in valid]
    if dropped:
        notes.append(f"unrecognised interests ignored: {', '.join(dropped)}")
    out["interests"] = list(dict.fromkeys(i for i in given if i in valid))
    return out, notes


def parse_trip_text(text: str, complete=None) -> dict:
    """
    Returns {"fields", "missing", "notes", "source"}. `source` is "llm" or
    "rules"; a model failure falls back to rules and is recorded in notes.
    """
    notes: list[str] = []
    source = "rules"
    raw = None
    if complete is not None:
        try:
            raw = _extract_json(complete(SYSTEM_PROMPT, text))
            source = "llm"
        except Exception as e:                          # noqa: BLE001
            notes.append(f"language model unavailable ({str(e)[:80]}); used rule parser")
    if raw is None:
        raw = parse_rules(text)

    fields, vnotes = validate(raw)
    notes.extend(vnotes)
    missing = [f for f in FIELDS if fields.get(f) in (None, [])]
    return {"fields": fields, "missing": missing, "notes": notes, "source": source}
