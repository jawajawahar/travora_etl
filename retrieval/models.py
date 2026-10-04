"""
Typed models for the Travora agent pipeline.

These are the shared blackboard. Agents read and write fields of declared type
rather than exchanging free natural language, which is the mechanism that stops
ambiguity and invented content propagating through a multi-agent system.

The single most important rule in the codebase lives here, in
ScoreVector.ids_must_match_candidates(): a scoring agent may only return scores
for the exact candidate IDs it was given. It cannot add a place, drop a place,
or rename one. This is what makes the old system's failure mode - a language
model emitting famous destinations from its own priors - structurally
impossible rather than merely discouraged.
"""
from __future__ import annotations
from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator, field_validator


class Category(str, Enum):
    HERITAGE = "heritage"
    WILDLIFE = "wildlife"
    NATURE = "nature"
    BEACH = "beach"
    RELIGIOUS = "religious"
    ADVENTURE = "adventure"
    CULTURAL = "cultural"
    URBAN = "urban"


# ---------------------------------------------------------------------------
class TripRequest(BaseModel):
    """What the traveller asked for. The only untyped input to the system."""
    start_date: date
    days: int = Field(ge=1, le=21)
    budget_lkr: float = Field(gt=0)
    party_size: int = Field(default=1, ge=1, le=20)
    interests: list[Category] = Field(min_length=1)
    start_lat: float = Field(ge=5.85, le=9.90)
    start_lon: float = Field(ge=79.60, le=81.95)
    w_pref: float = Field(default=0.5, ge=0.0, le=1.0)
    seed: int = 42
    max_travel_radius_km: float = Field(default=250.0, gt=0)

    @property
    def w_sust(self) -> float:
        return round(1.0 - self.w_pref, 4)

    @property
    def poi_budget_lkr(self) -> float:
        """
        Share of the budget available for attraction entry fees.

        Accommodation and transport dominate a Sri Lankan trip budget, so 20%
        is a deliberate, stated assumption rather than an arbitrary constant. It
        is used only to filter obviously unaffordable candidates; the solver
        enforces the real total-budget constraint.
        """
        return self.budget_lkr * 0.20


class Candidate(BaseModel):
    """One POI retrieved from the knowledge graph. All values come from Neo4j."""
    poi_id: str
    name: str
    category: Category
    lat: float
    lon: float
    district: str
    open_min: int = Field(ge=0, le=1440)
    close_min: int = Field(ge=0, le=1440)
    entry_fee_lkr: float = Field(ge=0)
    typical_dwell_min: int = Field(gt=0)
    popularity_index: float = Field(ge=0, le=1)
    prominence: float = Field(ge=0, le=1)

    # popularity_index is 0.0 both for a genuinely quiet destination AND for one
    # with no Wikipedia article. Only 7.3% of POIs resolve, so the two cases must
    # be distinguished: 0.0 means UNKNOWN far more often than it means UNCROWDED.
    popularity_known: bool = False
    retrieval_score: float = Field(default=0.0, ge=-1, le=1)
    rail_access_score: float = Field(default=0.0, ge=0, le=1)
    distance_from_start_km: float = Field(ge=0)

    # Provenance flags. The solver must treat a constraint as HARD only where
    # the underlying value is real; imputed values are soft preferences.
    hours_estimated: bool = True
    fee_estimated: bool = True

    @model_validator(mode="after")
    def hours_must_be_ordered(self):
        if self.close_min <= self.open_min:
            raise ValueError(
                f"{self.poi_id}: close_min ({self.close_min}) must exceed "
                f"open_min ({self.open_min})")
        return self

    @property
    def hours_are_hard(self) -> bool:
        return not self.hours_estimated

    @property
    def fee_is_hard(self) -> bool:
        return not self.fee_estimated


class CandidateSet(BaseModel):
    """The retrieval result. Everything downstream selects only from here."""
    request_id: str
    candidates: list[Candidate]
    query_widened: bool = False
    widening_reason: Optional[str] = None
    districts_covered: int = 0
    retrieval_ms: float = 0.0

    @property
    def ids(self) -> set[str]:
        return {c.poi_id for c in self.candidates}

    def __len__(self) -> int:
        return len(self.candidates)

    @field_validator("candidates")
    @classmethod
    def no_duplicate_ids(cls, v: list[Candidate]) -> list[Candidate]:
        seen = [c.poi_id for c in v]
        if len(seen) != len(set(seen)):
            dupes = {i for i in seen if seen.count(i) > 1}
            raise ValueError(f"duplicate candidate ids: {sorted(dupes)}")
        return v


# ---------------------------------------------------------------------------
class ScoreVector(BaseModel):
    """
    A scoring agent's output.

    THE ID CONTRACT. `scores` must contain exactly the keys in
    `candidate_ids` - no additions, no omissions. A language model that invents
    "Sigiriya" from its own priors, or silently drops a candidate it does not
    recognise, fails validation here and the response is rejected.
    """
    agent: str
    candidate_ids: list[str]
    scores: dict[str, float]

    @model_validator(mode="after")
    def ids_must_match_candidates(self):
        expected = set(self.candidate_ids)
        got = set(self.scores)

        invented = got - expected
        if invented:
            raise ValueError(
                f"{self.agent} returned {len(invented)} id(s) that were not in "
                f"the candidate set: {sorted(invented)[:5]}. A scoring agent may "
                f"only score the candidates it was given.")

        missing = expected - got
        if missing:
            raise ValueError(
                f"{self.agent} omitted {len(missing)} candidate(s): "
                f"{sorted(missing)[:5]}. Every candidate must receive a score.")

        bad = {k: v for k, v in self.scores.items() if not 0.0 <= v <= 1.0}
        if bad:
            raise ValueError(
                f"{self.agent} returned out-of-range scores: "
                f"{dict(list(bad.items())[:5])}. Scores must lie in [0, 1].")
        return self

    def as_series(self):
        import pandas as pd
        return pd.Series(self.scores, name=self.agent)


class RetrievalDiagnostics(BaseModel):
    """Recorded for the audit trail and for the anti-regression tests."""
    candidates_before_filter: int = 0
    candidates_after_filter: int = 0
    widening_steps: list[str] = Field(default_factory=list)
    category_counts: dict[str, int] = Field(default_factory=dict)
    district_counts: dict[str, int] = Field(default_factory=dict)
    mean_popularity: float = 0.0     # over candidates with KNOWN popularity only
    mean_prominence: float = 0.0
    popularity_known_pct: float = 0.0
