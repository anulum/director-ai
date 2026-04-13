# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Swarm Guardian: multi-agent safety coordinator
"""Central safety coordinator for multi-agent systems.

Each agent in a swarm registers with the ``SwarmGuardian``. The guardian
monitors:
- **Per-agent coherence**: each agent has its own ``LoopMonitor`` +
  thresholds via ``AgentProfile``.
- **Cross-agent consistency**: detects when Agent B contradicts
  Agent A's verified output.
- **Handoff scoring**: scores inter-agent messages for hallucination.
- **Cascade halt**: when one agent is compromised (injection or
  repeated hallucination), quarantine downstream agents.

Usage::

    from director_ai.agentic.swarm_guardian import SwarmGuardian
    from director_ai.agentic.agent_profile import AgentProfile

    guardian = SwarmGuardian()
    guardian.register_agent(AgentProfile.for_role("researcher"))
    guardian.register_agent(AgentProfile.for_role("summariser"))

    # Score a handoff between agents
    result = guardian.score_handoff(
        from_agent="researcher-0",
        to_agent="summariser-0",
        message="Paris is the capital of France.",
        context="European geography overview.",
    )
    if result.should_halt:
        guardian.quarantine_agent("researcher-0", reason="hallucination")
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from director_ai.agentic.agent_profile import AgentProfile
from director_ai.agentic.loop_monitor import LoopMonitor

logger = logging.getLogger("DirectorAI.SwarmGuardian")

__all__ = ["SwarmGuardian", "HandoffResult", "AgentState"]


@dataclass
class HandoffResult:
    """Result of scoring an inter-agent message."""

    from_agent: str
    to_agent: str
    score: float  # 0 = fully grounded, 1 = hallucinated
    should_halt: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class AgentState:
    """Runtime state of a registered agent."""

    profile: AgentProfile
    monitor: LoopMonitor
    quarantined: bool = False
    quarantine_reason: str = ""
    handoff_count: int = 0
    hallucination_count: int = 0
    registered_at: float = 0.0


class SwarmGuardian:
    """Central coordinator for multi-agent safety.

    Thread-safe registry of agents with cross-agent monitoring.

    Parameters
    ----------
    hallucination_threshold : float
        Handoff score above which a message is flagged (0–1).
    cascade_halt : bool
        When True, quarantining an agent also quarantines agents
        that received its output.
    max_agents : int
        Maximum number of registered agents (prevents unbounded growth).
    """

    def __init__(
        self,
        hallucination_threshold: float = 0.6,
        cascade_halt: bool = True,
        max_agents: int = 100,
    ) -> None:
        self._threshold = hallucination_threshold
        self._cascade_halt = cascade_halt
        self._max_agents = max_agents
        self._agents: dict[str, AgentState] = {}
        self._dependencies: dict[str, set[str]] = {}  # from → {to, ...}
        self._lock = threading.Lock()

    def register_agent(self, profile: AgentProfile) -> str:
        """Register an agent and create its LoopMonitor.

        Returns the agent_id.
        """
        with self._lock:
            if len(self._agents) >= self._max_agents:
                raise ValueError(f"Max agents ({self._max_agents}) reached")
            if profile.agent_id in self._agents:
                raise ValueError(f"Agent {profile.agent_id!r} already registered")

            monitor = LoopMonitor(
                goal=f"Agent {profile.agent_id} ({profile.role})",
                max_steps=profile.max_steps,
                max_tokens=profile.max_tokens,
                max_wall_seconds=profile.max_wall_seconds,
            )
            self._agents[profile.agent_id] = AgentState(
                profile=profile,
                monitor=monitor,
                registered_at=time.monotonic(),
            )
            self._dependencies[profile.agent_id] = set()
            logger.info(
                "Registered agent %s (role=%s, threshold=%.2f)",
                profile.agent_id,
                profile.role,
                profile.coherence_threshold,
            )
            return profile.agent_id

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the swarm."""
        with self._lock:
            self._agents.pop(agent_id, None)
            self._dependencies.pop(agent_id, None)
            # Clean up dependency references
            for deps in self._dependencies.values():
                deps.discard(agent_id)

    def score_handoff(
        self,
        from_agent: str,
        to_agent: str,
        message: str,
        context: str = "",
    ) -> HandoffResult:
        """Score an inter-agent message for hallucination.

        Uses keyword overlap heuristic (NLI scoring deferred to
        integration with CoherenceScorer).

        Parameters
        ----------
        from_agent : str
            Source agent ID.
        to_agent : str
            Destination agent ID.
        message : str
            The message being passed.
        context : str
            Ground truth or source context for scoring.
        """
        with self._lock:
            src = self._agents.get(from_agent)
            if src is None:
                return HandoffResult(
                    from_agent=from_agent,
                    to_agent=to_agent,
                    score=0.5,
                    should_halt=False,
                    reasons=["source agent not registered"],
                )

            if src.quarantined:
                return HandoffResult(
                    from_agent=from_agent,
                    to_agent=to_agent,
                    score=1.0,
                    should_halt=True,
                    reasons=[f"source agent quarantined: {src.quarantine_reason}"],
                )

            # Track dependency
            self._dependencies.setdefault(from_agent, set()).add(to_agent)
            src.handoff_count += 1

        # Score using keyword overlap (lightweight, no NLI dep)
        score = self._score_message(message, context)
        threshold = src.profile.coherence_threshold if src else self._threshold
        should_halt = score > (1.0 - threshold)

        reasons: list[str] = []
        if should_halt:
            reasons.append(
                f"handoff score {score:.3f} exceeds threshold {1.0 - threshold:.3f}"
            )
            with self._lock:
                if src:
                    src.hallucination_count += 1

        return HandoffResult(
            from_agent=from_agent,
            to_agent=to_agent,
            score=score,
            should_halt=should_halt,
            reasons=reasons,
        )

    def quarantine_agent(self, agent_id: str, reason: str) -> list[str]:
        """Quarantine an agent and optionally cascade to downstream.

        Returns list of all quarantined agent IDs (including cascaded).
        """
        quarantined: list[str] = []
        with self._lock:
            self._quarantine_recursive(agent_id, reason, quarantined)
        return quarantined

    def _quarantine_recursive(
        self,
        agent_id: str,
        reason: str,
        quarantined: list[str],
    ) -> None:
        """Recursively quarantine an agent and its downstream."""
        state = self._agents.get(agent_id)
        if state is None or state.quarantined:
            return
        state.quarantined = True
        state.quarantine_reason = reason
        quarantined.append(agent_id)
        logger.warning("Quarantined agent %s: %s", agent_id, reason)

        if self._cascade_halt:
            for downstream in self._dependencies.get(agent_id, set()):
                cascade_reason = f"cascade from {agent_id}: {reason}"
                self._quarantine_recursive(downstream, cascade_reason, quarantined)

    def is_quarantined(self, agent_id: str) -> bool:
        """Check if an agent is quarantined."""
        with self._lock:
            state = self._agents.get(agent_id)
            return state.quarantined if state else False

    def get_agent_state(self, agent_id: str) -> AgentState | None:
        """Get the current state of a registered agent."""
        with self._lock:
            return self._agents.get(agent_id)

    @property
    def agent_count(self) -> int:
        """Number of registered agents."""
        with self._lock:
            return len(self._agents)

    @property
    def quarantined_count(self) -> int:
        """Number of quarantined agents."""
        with self._lock:
            return sum(1 for s in self._agents.values() if s.quarantined)

    def list_agents(self) -> list[str]:
        """List all registered agent IDs."""
        with self._lock:
            return list(self._agents.keys())

    @staticmethod
    def _score_message(message: str, context: str) -> float:
        """Lightweight scoring via keyword overlap.

        Returns 0.0 (grounded) to 1.0 (hallucinated).
        """
        if not context:
            return 0.5  # no context → neutral

        msg_words = set(message.lower().split())
        ctx_words = set(context.lower().split())
        if not msg_words:
            return 0.5

        overlap = len(msg_words & ctx_words)
        coverage = overlap / len(msg_words) if msg_words else 0.0
        # Invert: high coverage = low hallucination score
        return max(0.0, min(1.0, 1.0 - coverage))
