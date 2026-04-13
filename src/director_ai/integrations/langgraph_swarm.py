# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — LangGraph multi-agent swarm guardian adapter
"""LangGraph swarm adapter: insert guardian nodes between agents.

Wraps a LangGraph multi-agent graph with Director-AI guardrails.
Each agent's output passes through a guardian node that scores the
handoff before forwarding to the next agent.

Usage::

    from director_ai.integrations.langgraph_swarm import (
        SwarmGraphBuilder,
        guardian_edge,
    )
    from director_ai.agentic import AgentProfile, SwarmGuardian

    guardian = SwarmGuardian()
    guardian.register_agent(AgentProfile.for_role("researcher"))
    guardian.register_agent(AgentProfile.for_role("summariser"))

    # Insert guardian between agents in LangGraph
    builder = SwarmGraphBuilder(guardian)
    builder.add_guarded_edge("researcher-0", "summariser-0")
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from director_ai.agentic.handoff_scorer import HandoffScorer
from director_ai.agentic.swarm_guardian import SwarmGuardian

logger = logging.getLogger("DirectorAI.LangGraphSwarm")

__all__ = ["SwarmGraphBuilder", "GuardianNodeResult", "guardian_edge"]


@dataclass
class GuardianNodeResult:
    """Result from a guardian edge evaluation."""

    approved: bool
    from_agent: str
    to_agent: str
    score: float
    quarantined: bool = False
    reason: str = ""


def guardian_edge(
    guardian: SwarmGuardian,
    from_agent: str,
    to_agent: str,
    context_key: str = "context",
    message_key: str = "messages",
) -> Callable:
    """Create a LangGraph conditional edge function.

    Returns a function suitable for ``graph.add_conditional_edges()``.
    The function inspects the state, scores the handoff, and returns
    either the ``to_agent`` node name or ``"__end__"`` if quarantined.

    Parameters
    ----------
    guardian : SwarmGuardian
        The swarm guardian instance.
    from_agent : str
        Source agent ID.
    to_agent : str
        Destination agent ID.
    context_key : str
        State key containing the ground truth context.
    message_key : str
        State key containing the message list.
    """

    def _edge_fn(state: dict[str, Any]) -> str:
        context = state.get(context_key, "")
        messages = state.get(message_key, [])

        # Extract last message text
        if messages and isinstance(messages[-1], dict):
            message = messages[-1].get("content", str(messages[-1]))
        elif messages:
            message = str(messages[-1])
        else:
            message = ""

        result = guardian.score_handoff(
            from_agent=from_agent,
            to_agent=to_agent,
            message=message,
            context=str(context),
        )

        if result.should_halt:
            logger.warning(
                "Guardian blocked handoff %s → %s: %s",
                from_agent,
                to_agent,
                result.reasons,
            )
            guardian.quarantine_agent(from_agent, reason="handoff_rejected")
            return "__end__"

        return to_agent

    _edge_fn.__name__ = f"guardian_{from_agent}_to_{to_agent}"
    return _edge_fn


class SwarmGraphBuilder:
    """Helper for building LangGraph graphs with guardian edges.

    Accumulates guarded edges and provides a ``build()`` method that
    returns the edge configuration for LangGraph's ``StateGraph``.

    Parameters
    ----------
    guardian : SwarmGuardian
        The swarm guardian coordinating all agents.
    scorer : HandoffScorer | None
        Optional custom handoff scorer. Defaults to guardian's built-in.
    """

    def __init__(
        self,
        guardian: SwarmGuardian,
        scorer: HandoffScorer | None = None,
    ) -> None:
        self._guardian = guardian
        self._scorer = scorer
        self._edges: list[dict[str, Any]] = []

    def add_guarded_edge(
        self,
        from_agent: str,
        to_agent: str,
        context_key: str = "context",
        message_key: str = "messages",
    ) -> None:
        """Register a guarded edge between two agents.

        At build time, this creates a conditional edge that scores
        the handoff before forwarding.
        """
        self._edges.append(
            {
                "from": from_agent,
                "to": to_agent,
                "edge_fn": guardian_edge(
                    self._guardian,
                    from_agent,
                    to_agent,
                    context_key,
                    message_key,
                ),
            }
        )

    def get_edges(self) -> list[dict[str, Any]]:
        """Return all registered guarded edges."""
        return list(self._edges)

    @property
    def edge_count(self) -> int:
        """Number of registered guarded edges."""
        return len(self._edges)
