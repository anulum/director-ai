# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CrewAI multi-agent swarm guardian adapter
"""CrewAI swarm adapter: wrap crew tasks with guardian scoring.

Hooks into CrewAI task lifecycle to score agent outputs before
passing them to the next agent in the crew.

Usage::

    from director_ai.integrations.crewai_swarm import CrewGuardian
    from director_ai.agentic import AgentProfile, SwarmGuardian

    guardian = SwarmGuardian()
    guardian.register_agent(AgentProfile.for_role("researcher"))
    guardian.register_agent(AgentProfile.for_role("writer"))

    crew_guardian = CrewGuardian(guardian)
    result = crew_guardian.guard_task_output(
        agent_id="researcher-0",
        task_output="Paris is the capital of France.",
        context="European geography.",
        next_agent_id="writer-0",
    )
    if not result.approved:
        print("Task output blocked:", result.reason)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from director_ai.agentic.handoff_scorer import HandoffScorer
from director_ai.agentic.swarm_guardian import SwarmGuardian

logger = logging.getLogger("DirectorAI.CrewAISwarm")

__all__ = ["CrewGuardian", "TaskGuardResult"]


@dataclass(frozen=True)
class TaskGuardResult:
    """Result of guarding a CrewAI task output."""

    approved: bool
    agent_id: str
    score: float
    reason: str = ""
    quarantined: bool = False


class CrewGuardian:
    """Guard CrewAI task outputs via SwarmGuardian.

    Parameters
    ----------
    guardian : SwarmGuardian
        The swarm guardian coordinating agents.
    scorer : HandoffScorer | None
        Optional custom scorer. Defaults to keyword-based.
    auto_quarantine : bool
        Quarantine agent on rejection (default True).
    """

    def __init__(
        self,
        guardian: SwarmGuardian,
        scorer: HandoffScorer | None = None,
        auto_quarantine: bool = True,
    ) -> None:
        self._guardian = guardian
        self._scorer = scorer or HandoffScorer()
        self._auto_quarantine = auto_quarantine
        self._guarded_count = 0
        self._blocked_count = 0

    def guard_task_output(
        self,
        agent_id: str,
        task_output: str,
        context: str = "",
        next_agent_id: str = "",
    ) -> TaskGuardResult:
        """Score a task output before handoff.

        Parameters
        ----------
        agent_id : str
            The agent that produced the output.
        task_output : str
            The agent's output text.
        context : str
            Ground truth or source material.
        next_agent_id : str
            The next agent in the crew pipeline.

        Returns
        -------
        TaskGuardResult
        """
        self._guarded_count += 1

        # Check if source is already quarantined
        if self._guardian.is_quarantined(agent_id):
            self._blocked_count += 1
            return TaskGuardResult(
                approved=False,
                agent_id=agent_id,
                score=1.0,
                reason=f"agent {agent_id} is quarantined",
                quarantined=True,
            )

        # Score the handoff
        result = self._guardian.score_handoff(
            from_agent=agent_id,
            to_agent=next_agent_id,
            message=task_output,
            context=context,
        )

        if result.should_halt:
            self._blocked_count += 1
            reason = (
                "; ".join(result.reasons)
                if result.reasons
                else "score exceeded threshold"
            )

            if self._auto_quarantine:
                self._guardian.quarantine_agent(agent_id, reason=reason)

            return TaskGuardResult(
                approved=False,
                agent_id=agent_id,
                score=result.score,
                reason=reason,
                quarantined=self._auto_quarantine,
            )

        return TaskGuardResult(
            approved=True,
            agent_id=agent_id,
            score=result.score,
        )

    def guard_crew_output(
        self,
        crew_result: Any,
        final_agent_id: str = "",
        context: str = "",
    ) -> TaskGuardResult:
        """Score the final crew output (last agent's result).

        Parameters
        ----------
        crew_result : Any
            The crew's final output (stringified).
        final_agent_id : str
            The last agent in the pipeline.
        context : str
            Ground truth context.
        """
        output_text = str(crew_result)
        return self.guard_task_output(
            agent_id=final_agent_id,
            task_output=output_text,
            context=context,
            next_agent_id="__output__",
        )

    @property
    def stats(self) -> dict[str, int]:
        """Guarding statistics."""
        return {
            "guarded": self._guarded_count,
            "blocked": self._blocked_count,
            "approved": self._guarded_count - self._blocked_count,
        }
