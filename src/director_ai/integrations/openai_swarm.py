# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — OpenAI Swarm SDK guardian adapter
"""OpenAI Swarm adapter: guard agent handoffs in OpenAI's Swarm SDK.

Wraps function handoffs between Swarm agents with Director-AI scoring.
When an agent's output is flagged as hallucinated, the handoff is
blocked and the agent is quarantined.

The OpenAI Swarm SDK uses function-based agent handoffs. This adapter
wraps those functions with a guardian layer.

Usage::

    from director_ai.integrations.openai_swarm import GuardedHandoff
    from director_ai.agentic import AgentProfile, SwarmGuardian

    guardian = SwarmGuardian()
    guardian.register_agent(AgentProfile.for_role("researcher"))
    guardian.register_agent(AgentProfile.for_role("writer"))

    # Wrap a handoff function
    guarded = GuardedHandoff(guardian, "researcher-0", "writer-0")
    safe_result = guarded.check("Agent output text", context="Source doc")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from director_ai.agentic.swarm_guardian import SwarmGuardian

logger = logging.getLogger("DirectorAI.OpenAISwarm")

__all__ = ["GuardedHandoff", "HandoffCheckResult"]


@dataclass(frozen=True)
class HandoffCheckResult:
    """Result of checking a handoff in OpenAI Swarm."""

    approved: bool
    from_agent: str
    to_agent: str
    score: float
    blocked_reason: str = ""


class GuardedHandoff:
    """Guard a single agent→agent handoff in OpenAI Swarm.

    Parameters
    ----------
    guardian : SwarmGuardian
        The swarm guardian instance.
    from_agent : str
        Source agent ID.
    to_agent : str
        Destination agent ID.
    """

    def __init__(
        self,
        guardian: SwarmGuardian,
        from_agent: str,
        to_agent: str,
    ) -> None:
        self._guardian = guardian
        self._from = from_agent
        self._to = to_agent

    def check(
        self,
        message: str,
        context: str = "",
    ) -> HandoffCheckResult:
        """Check if a handoff message is safe.

        Parameters
        ----------
        message : str
            The output being handed off.
        context : str
            Ground truth or source material.

        Returns
        -------
        HandoffCheckResult
        """
        result = self._guardian.score_handoff(
            from_agent=self._from,
            to_agent=self._to,
            message=message,
            context=context,
        )

        if result.should_halt:
            reason = (
                "; ".join(result.reasons) if result.reasons else "threshold exceeded"
            )
            self._guardian.quarantine_agent(self._from, reason=reason)
            return HandoffCheckResult(
                approved=False,
                from_agent=self._from,
                to_agent=self._to,
                score=result.score,
                blocked_reason=reason,
            )

        return HandoffCheckResult(
            approved=True,
            from_agent=self._from,
            to_agent=self._to,
            score=result.score,
        )

    def wrap_function(self, fn: Any) -> Any:
        """Wrap a Swarm handoff function with guardian check.

        The wrapper calls the original function, then checks the
        result. If blocked, returns None instead of the agent.

        Parameters
        ----------
        fn : callable
            The original handoff function (returns an Agent or str).

        Returns
        -------
        callable
            Wrapped function with guardian check.
        """
        guardian = self

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            output_text = str(result) if result is not None else ""
            check = guardian.check(output_text)
            if not check.approved:
                logger.warning(
                    "Handoff blocked %s → %s: %s",
                    guardian._from,
                    guardian._to,
                    check.blocked_reason,
                )
                return None
            return result

        _wrapped.__name__ = (
            f"guarded_{fn.__name__}" if hasattr(fn, "__name__") else "guarded_handoff"
        )
        return _wrapped
