"""
CrewAI orchestration adapter.

What this does, and what it honestly does not
---------------------------------------------
CrewAI's design centre is orchestrating LLM-driven agents that reason about
which tool to call next. Travora deliberately has only ONE agent that calls a
language model: Preference. Sustainability is arithmetic and Decision is AC-3
plus branch and bound, and both are non-generative BY DESIGN, because a
sustainability score that varied between runs would make the weight-sensitivity
analysis in Chapter 5 unrepeatable.

So CrewAI here supplies declarative role structure and a per-task audit record.
It does not supply autonomous tool selection, because three of the four agents
must not select anything. That is a deliberate architectural choice and should
be described that way rather than implying more autonomy than exists.

Execution is identical either way: this adapter delegates to TravoraPipeline,
so enabling or disabling CrewAI cannot change an itinerary. The deterministic
pipeline remains the production path; this module documents the agent topology
in CrewAI's vocabulary and emits the same audit trail.

    from agents.crew import TravoraCrew
    crew = TravoraCrew(session)
    result = crew.kickoff(trip_request)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from retrieval.models import TripRequest
from solver.models import SolverConfig
from .pipeline import TravoraPipeline, PipelineResult
from .preference import PreferenceAgent
from .sustainability import SustainabilityAgent

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent topology, declared once and used both by this adapter and by the
# transparency view. Keeping it as data rather than prose means the roles shown
# to a user cannot drift from the roles the system actually implements.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AgentSpec:
    name: str
    role: str
    goal: str
    uses_llm: bool
    may_select: bool
    rationale: str


AGENT_SPECS: tuple[AgentSpec, ...] = (
    AgentSpec(
        name="retrieval",
        role="Tourism Data Retrieval Specialist",
        goal="Fetch candidate destinations from the knowledge graph that match "
             "the traveller's interests, budget and reachable area.",
        uses_llm=False,
        may_select=False,
        rationale="Every place the system can propose originates here, from a "
                  "Cypher query against real data. Nothing downstream may add "
                  "a place that this step did not return.",
    ),
    AgentSpec(
        name="preference",
        role="Traveller Preference Analyst",
        goal="Score each retrieved candidate for how well it matches the "
             "traveller's stated interests.",
        uses_llm=True,
        may_select=False,
        rationale="The only language model in the system. It scores a list it "
                  "is given and may not add, remove or reorder anything; the "
                  "ScoreVector contract rejects any response containing an id "
                  "that was not supplied.",
    ),
    AgentSpec(
        name="sustainability",
        role="Sustainable Tourism Assessor",
        goal="Score each candidate on transport emissions, visitor pressure "
             "and ecological sensitivity.",
        uses_llm=False,
        may_select=False,
        rationale="Pure arithmetic over stored attributes. Deterministic by "
                  "design: identical inputs must always give identical scores "
                  "or the weight-sensitivity analysis cannot be reproduced.",
    ),
    AgentSpec(
        name="decision",
        role="Itinerary Decision Arbiter",
        goal="Establish the feasible region with AC-3 and select the itinerary "
             "maximising the weighted objective within it.",
        uses_llm=False,
        may_select=True,
        rationale="The only component that selects. Deterministic constraint "
                  "propagation and search, so feasibility is a structural "
                  "guarantee rather than an emergent property of generation.",
    ),
)


def topology() -> list[dict]:
    """Agent topology as plain data, for the API and the transparency view."""
    return [
        {"name": a.name, "role": a.role, "goal": a.goal,
         "uses_llm": a.uses_llm, "may_select": a.may_select,
         "rationale": a.rationale}
        for a in AGENT_SPECS
    ]


def crewai_available() -> bool:
    try:
        import crewai  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
@dataclass
class TaskRecord:
    """Per-task audit entry, mirroring CrewAI's task output structure."""
    agent: str
    task: str
    status: str = "pending"
    output_summary: str = ""
    duration_ms: float = 0.0


@dataclass
class TravoraCrew:
    """
    Role-declared orchestration over the deterministic pipeline.

    `use_crewai` controls only whether CrewAI Agent objects are constructed for
    documentation and inspection. Execution is delegated to TravoraPipeline
    either way, so the flag cannot change an itinerary - which is exactly what
    makes it safe to enable for a demonstration and disable for evaluation.
    """
    session: Any
    preference: Optional[PreferenceAgent] = None
    sustainability: Optional[SustainabilityAgent] = None
    solver_config: Optional[SolverConfig] = None
    use_crewai: bool = True
    tasks: list[TaskRecord] = field(default_factory=list)

    def __post_init__(self):
        self.preference = self.preference or PreferenceAgent()
        self.sustainability = self.sustainability or SustainabilityAgent()
        self.solver_config = self.solver_config or SolverConfig()
        self._crew_agents = None

        if self.use_crewai and crewai_available():
            self._crew_agents = self._build_crew_agents()
            log.info("CrewAI agents declared: %s",
                     [a.name for a in AGENT_SPECS])
        elif self.use_crewai:
            log.info("crewai not installed; running the deterministic pipeline "
                     "(behaviour is identical either way)")

    def _build_crew_agents(self):
        """Declare the agents in CrewAI's vocabulary for inspection."""
        from crewai import Agent
        agents = {}
        for spec in AGENT_SPECS:
            agents[spec.name] = Agent(
                role=spec.role,
                goal=spec.goal,
                backstory=spec.rationale,
                allow_delegation=False,     # no agent may act for another
                verbose=False,
            )
        return agents

    # -- execution ---------------------------------------------------------
    def kickoff(self, req: TripRequest) -> PipelineResult:
        import time
        self.tasks = []
        t0 = time.perf_counter()

        pipeline = TravoraPipeline(self.session, self.preference,
                                   self.sustainability, self.solver_config)
        result = pipeline.run(req)
        a = result.audit

        self.tasks = [
            TaskRecord("retrieval", "retrieve_candidates", "done",
                       f"{a.candidates_retrieved} candidates across "
                       f"{a.districts_covered} districts", a.retrieval_ms),
            TaskRecord("preference", "score_preferences", "done",
                       f"mean {a.preference_mean:.3f} "
                       f"({'LLM' if a.preference_used_llm else 'fallback'})",
                       a.scoring_ms),
            TaskRecord("sustainability", "score_sustainability", "done",
                       f"mean {a.sustainability_mean:.3f} (deterministic)", 0.0),
            TaskRecord("decision", "solve_itinerary",
                       "done" if result.solver.feasible else "infeasible",
                       (f"{len(result.solver.itinerary.stops)} stops"
                        if result.solver.feasible
                        else result.solver.conflict.conflict.value),
                       a.solve_ms),
        ]
        log.info("Crew finished in %.0f ms", (time.perf_counter() - t0) * 1000)
        return result

    def task_report(self) -> list[dict]:
        return [{"agent": t.agent, "task": t.task, "status": t.status,
                 "output": t.output_summary, "duration_ms": round(t.duration_ms, 1)}
                for t in self.tasks]
