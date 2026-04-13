# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Per-agent configuration profiles
"""Per-agent configuration profiles for swarm guardianship.

Each agent in a multi-agent system can have distinct safety thresholds,
tool permissions, and resource budgets. ``AgentProfile`` provides
role-based defaults with full override capability.

Usage::

    from director_ai.agentic.agent_profile import AgentProfile

    profile = AgentProfile.for_role("researcher")
    # AgentProfile(agent_id='researcher-0', role='researcher',
    #              coherence_threshold=0.65, max_steps=100, ...)

    # Custom overrides
    profile = AgentProfile(
        agent_id="med-reviewer",
        role="reviewer",
        coherence_threshold=0.85,
        allowed_tools=["search", "cite"],
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["AgentProfile", "ROLE_DEFAULTS"]

# Role → default overrides. Keys must match AgentProfile field names.
ROLE_DEFAULTS: dict[str, dict] = {
    "researcher": {
        "coherence_threshold": 0.65,
        "max_steps": 100,
        "max_tokens": 500_000,
        "max_wall_seconds": 600.0,
        "escalation_policy": "warn",
    },
    "summariser": {
        "coherence_threshold": 0.72,
        "max_steps": 30,
        "max_tokens": 200_000,
        "max_wall_seconds": 120.0,
        "escalation_policy": "halt",
    },
    "coder": {
        "coherence_threshold": 0.55,
        "max_steps": 200,
        "max_tokens": 1_000_000,
        "max_wall_seconds": 900.0,
        "escalation_policy": "warn",
    },
    "reviewer": {
        "coherence_threshold": 0.80,
        "max_steps": 50,
        "max_tokens": 300_000,
        "max_wall_seconds": 300.0,
        "escalation_policy": "halt",
    },
    "planner": {
        "coherence_threshold": 0.60,
        "max_steps": 40,
        "max_tokens": 250_000,
        "max_wall_seconds": 180.0,
        "escalation_policy": "warn",
    },
    "executor": {
        "coherence_threshold": 0.70,
        "max_steps": 150,
        "max_tokens": 800_000,
        "max_wall_seconds": 600.0,
        "escalation_policy": "halt",
    },
}

_COUNTER: dict[str, int] = {}


def _next_id(role: str) -> str:
    """Generate sequential agent IDs: ``researcher-0``, ``researcher-1``, …"""
    n = _COUNTER.get(role, 0)
    _COUNTER[role] = n + 1
    return f"{role}-{n}"


@dataclass(frozen=True)
class AgentProfile:
    """Immutable per-agent configuration.

    Parameters
    ----------
    agent_id : str
        Unique identifier for this agent instance.
    role : str
        Semantic role (``"researcher"``, ``"summariser"``, ``"coder"``, etc.).
        Used for default threshold selection and swarm metrics grouping.
    coherence_threshold : float
        Minimum coherence score to approve output (0–1).
    allowed_tools : list[str]
        Tool names this agent is permitted to call. Empty list means
        unrestricted.
    max_steps : int
        Maximum agentic loop iterations before forced halt.
    max_tokens : int
        Token budget (prompt + completion) before forced halt.
    max_wall_seconds : float
        Wall-clock timeout in seconds.
    escalation_policy : str
        What happens when threshold is breached:
        ``"warn"`` — log warning, continue.
        ``"halt"`` — stop agent immediately.
        ``"quarantine"`` — halt and notify SwarmGuardian.
    """

    agent_id: str = ""
    role: str = "general"
    coherence_threshold: float = 0.60
    allowed_tools: list[str] = field(default_factory=list)
    max_steps: int = 50
    max_tokens: int = 500_000
    max_wall_seconds: float = 300.0
    escalation_policy: str = "warn"

    def __post_init__(self) -> None:
        if self.escalation_policy not in ("warn", "halt", "quarantine"):
            raise ValueError(
                f"escalation_policy must be 'warn', 'halt', or 'quarantine', "
                f"got {self.escalation_policy!r}"
            )
        if not 0.0 <= self.coherence_threshold <= 1.0:
            raise ValueError(
                f"coherence_threshold must be in [0, 1], got {self.coherence_threshold}"
            )

    @classmethod
    def for_role(cls, role: str, **overrides) -> AgentProfile:
        """Create a profile with role-based defaults.

        Looks up ``ROLE_DEFAULTS[role]`` for sensible defaults, then
        applies any keyword overrides on top.

        Parameters
        ----------
        role : str
            One of the predefined roles or a custom string.
        **overrides
            Any ``AgentProfile`` field to override.

        Returns
        -------
        AgentProfile
        """
        defaults = dict(ROLE_DEFAULTS.get(role, {}))
        defaults["role"] = role
        defaults["agent_id"] = overrides.pop("agent_id", _next_id(role))
        defaults.update(overrides)
        return cls(**defaults)

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "coherence_threshold": self.coherence_threshold,
            "allowed_tools": list(self.allowed_tools),
            "max_steps": self.max_steps,
            "max_tokens": self.max_tokens,
            "max_wall_seconds": self.max_wall_seconds,
            "escalation_policy": self.escalation_policy,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AgentProfile:
        """Deserialise from a plain dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
