"""
Sustainability Agent - pure arithmetic, no language model.

Deliberately contains no LLM call. Sustainability scoring must be reproducible:
the same POI under the same configuration must always receive the same score,
or the weight-sensitivity analysis in Chapter 5 cannot be repeated and the
evaluation is not replicable.

"Agentic" refers to the coordinated multi-agent architecture, not to every
component being a language model. Sapkota et al. define Agentic AI by multi-agent
collaboration, task decomposition and orchestrated autonomy - none of which
requires that each agent generate text.

Score (place level)
-------------------
    S_place(p) = 1 - (w_c * C_crowd(p) + w_e * C_eco(p))      w_c = 0.75, w_e = 0.25

Transport is no longer a property of the place. It used to enter here as
C_transport = 1 - rail_access_score, which scored a place near a station the
same whether the traveller reached it on foot or after six hours by car. The
journey is now measured directly (solver/emissions.py) and combined at trip
level as S_trip = 0.6 * mean S_place + 0.4 * S_journey; keeping C_transport as
well would count transport twice. The 3:1 crowding:ecology ratio of the
original weights (0.45 : 0.15) is kept.

  C_transport  still reported in explain() for transparency, at weight 0.
  C_crowd      popularity_index where measured. Where it is NOT measured the
               neutral prior is used: absence of a Wikipedia article is not
               evidence that a place is quiet.
  C_eco        ecological sensitivity from the protected-area class, so strictly
               protected sites are not loaded with visitors.
"""
from __future__ import annotations
import logging
from pathlib import Path

import yaml

from retrieval.models import Candidate, ScoreVector

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "sustainability.yaml"

# Applied where popularity_known is False. Neutral rather than optimistic: an
# unmeasured POI must not be rewarded for the absence of data, which is the
# error that made retrieval rank every unmeasured POI above Sigiriya.
UNKNOWN_CROWDING_PRIOR = 0.35


class SustainabilityAgent:
    """Scores candidates 0-1, where 1 is the most sustainable."""

    agent_name = "sustainability"

    def __init__(self, config_path: Path | None = None,
                 w_transport: float = 0.0, w_crowd: float = 0.75,
                 w_eco: float = 0.25):
        total = w_transport + w_crowd + w_eco
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {total}")
        self.w_transport = w_transport
        self.w_crowd = w_crowd
        self.w_eco = w_eco

        path = config_path or CONFIG_PATH
        self.cfg = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f) or {}
        else:
            log.warning("sustainability.yaml not found at %s; using defaults", path)

    # -- components --------------------------------------------------------
    def transport_cost(self, c: Candidate) -> float:
        return round(1.0 - max(0.0, min(1.0, c.rail_access_score)), 4)

    def crowding_cost(self, c: Candidate) -> float:
        if not c.popularity_known:
            return UNKNOWN_CROWDING_PRIOR
        return round(max(0.0, min(1.0, c.popularity_index)), 4)

    def eco_cost(self, c: Candidate) -> float:
        """
        Ecological sensitivity. Wildlife and nature sites carry more visitor
        impact per head than an urban museum, so they are weighted accordingly.
        This is a category proxy: OSM protect_class is not carried through to
        the Candidate model, and inventing a value would be worse than using a
        stated approximation.
        """
        by_category = {"wildlife": 0.85, "nature": 0.55, "beach": 0.45,
                       "adventure": 0.40, "heritage": 0.25, "religious": 0.20,
                       "cultural": 0.15, "urban": 0.10}
        return by_category.get(c.category.value, 0.30)

    # -- scoring -----------------------------------------------------------
    def score_one(self, c: Candidate) -> float:
        cost = (self.w_transport * self.transport_cost(c)
                + self.w_crowd * self.crowding_cost(c)
                + self.w_eco * self.eco_cost(c))
        return round(max(0.0, min(1.0, 1.0 - cost)), 4)

    def score(self, candidates: list[Candidate]) -> ScoreVector:
        ids = [c.poi_id for c in candidates]
        scores = {c.poi_id: self.score_one(c) for c in candidates}
        # Validated by the same contract every agent obeys, even though this one
        # cannot invent an id: the guarantee should hold by construction, not by
        # trusting that a particular implementation happens to behave.
        return ScoreVector(agent=self.agent_name, candidate_ids=ids, scores=scores)

    def explain(self, c: Candidate) -> dict:
        """Per-component breakdown for the transparency view."""
        return {
            "poi_id": c.poi_id,
            "transport_cost": self.transport_cost(c),
            "crowding_cost": self.crowding_cost(c),
            "crowding_measured": c.popularity_known,
            "eco_cost": self.eco_cost(c),
            "weights": {"transport": self.w_transport, "crowd": self.w_crowd,
                        "eco": self.w_eco},
            "score": self.score_one(c),
        }
