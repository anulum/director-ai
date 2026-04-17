# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for ``director_ai.agentic.agent_profile``.

Covers construction, role defaults, validation, serialisation,
factory methods, and edge cases.
"""

from __future__ import annotations

import pytest

from director_ai.agentic.agent_profile import (
    ROLE_DEFAULTS,
    AgentProfile,
    _next_id,
)

# ── Construction ────────────────────────────────────────────────────────


class TestConstruction:
    def test_defaults(self):
        p = AgentProfile(agent_id="a1")
        assert p.role == "general"
        assert p.coherence_threshold == 0.60
        assert p.max_steps == 50
        assert p.escalation_policy == "warn"
        assert p.allowed_tools == []

    def test_custom_fields(self):
        p = AgentProfile(
            agent_id="med-1",
            role="reviewer",
            coherence_threshold=0.85,
            allowed_tools=["search", "cite"],
            max_steps=20,
            escalation_policy="halt",
        )
        assert p.agent_id == "med-1"
        assert p.role == "reviewer"
        assert p.coherence_threshold == 0.85
        assert p.allowed_tools == ["search", "cite"]
        assert p.max_steps == 20

    def test_frozen(self):
        p = AgentProfile(agent_id="a1")
        with pytest.raises(AttributeError):
            p.role = "changed"  # type: ignore[misc]


# ── Validation ──────────────────────────────────────────────────────────


class TestValidation:
    def test_invalid_escalation_policy(self):
        with pytest.raises(ValueError, match="escalation_policy"):
            AgentProfile(agent_id="a1", escalation_policy="invalid")

    def test_threshold_too_high(self):
        with pytest.raises(ValueError, match="coherence_threshold"):
            AgentProfile(agent_id="a1", coherence_threshold=1.5)

    def test_threshold_too_low(self):
        with pytest.raises(ValueError, match="coherence_threshold"):
            AgentProfile(agent_id="a1", coherence_threshold=-0.1)

    def test_threshold_boundary_zero(self):
        p = AgentProfile(agent_id="a1", coherence_threshold=0.0)
        assert p.coherence_threshold == 0.0

    def test_threshold_boundary_one(self):
        p = AgentProfile(agent_id="a1", coherence_threshold=1.0)
        assert p.coherence_threshold == 1.0

    def test_all_valid_policies(self):
        for policy in ("warn", "halt", "quarantine"):
            p = AgentProfile(agent_id="a1", escalation_policy=policy)
            assert p.escalation_policy == policy


# ── Role Defaults ───────────────────────────────────────────────────────


class TestRoleDefaults:
    def test_researcher_defaults(self):
        p = AgentProfile.for_role("researcher")
        assert p.role == "researcher"
        assert p.coherence_threshold == 0.65
        assert p.max_steps == 100
        assert p.escalation_policy == "warn"

    def test_summariser_defaults(self):
        p = AgentProfile.for_role("summariser")
        assert p.coherence_threshold == 0.72
        assert p.max_steps == 30
        assert p.escalation_policy == "halt"

    def test_coder_defaults(self):
        p = AgentProfile.for_role("coder")
        assert p.coherence_threshold == 0.55
        assert p.max_steps == 200

    def test_reviewer_defaults(self):
        p = AgentProfile.for_role("reviewer")
        assert p.coherence_threshold == 0.80

    def test_planner_defaults(self):
        p = AgentProfile.for_role("planner")
        assert p.coherence_threshold == 0.60

    def test_executor_defaults(self):
        p = AgentProfile.for_role("executor")
        assert p.coherence_threshold == 0.70

    def test_all_roles_have_required_fields(self):
        for role in ROLE_DEFAULTS:
            p = AgentProfile.for_role(role)
            assert p.role == role
            assert 0.0 <= p.coherence_threshold <= 1.0
            assert p.max_steps > 0
            assert p.max_tokens > 0
            assert p.max_wall_seconds > 0

    def test_unknown_role_uses_general_defaults(self):
        p = AgentProfile.for_role("alien")
        assert p.role == "alien"
        # Should use AgentProfile defaults (no ROLE_DEFAULTS entry)
        assert p.coherence_threshold == 0.60
        assert p.max_steps == 50

    def test_override_role_defaults(self):
        p = AgentProfile.for_role("researcher", coherence_threshold=0.90)
        assert p.role == "researcher"
        assert p.coherence_threshold == 0.90  # overridden
        assert p.max_steps == 100  # from role default

    def test_custom_agent_id(self):
        p = AgentProfile.for_role("coder", agent_id="my-coder")
        assert p.agent_id == "my-coder"


# ── Auto-ID Generation ─────────────────────────────────────────────────


class TestAutoId:
    def test_sequential_ids(self):
        # _next_id is stateful, but we can test incrementing
        role = "test_unique_role_for_test"
        id1 = _next_id(role)
        id2 = _next_id(role)
        assert id1.startswith(role)
        assert id2.startswith(role)
        assert id1 != id2

    def test_for_role_generates_id(self):
        p = AgentProfile.for_role("summariser")
        assert p.agent_id.startswith("summariser-")


# ── Serialisation ───────────────────────────────────────────────────────


class TestSerialisation:
    def test_to_dict(self):
        p = AgentProfile(
            agent_id="s1",
            role="summariser",
            coherence_threshold=0.72,
            allowed_tools=["search"],
            max_steps=30,
            max_tokens=200_000,
            max_wall_seconds=120.0,
            escalation_policy="halt",
        )
        d = p.to_dict()
        assert d["agent_id"] == "s1"
        assert d["role"] == "summariser"
        assert d["coherence_threshold"] == 0.72
        assert d["allowed_tools"] == ["search"]
        assert d["max_steps"] == 30
        assert d["escalation_policy"] == "halt"

    def test_from_dict(self):
        d = {
            "agent_id": "r1",
            "role": "reviewer",
            "coherence_threshold": 0.85,
            "allowed_tools": ["cite"],
            "max_steps": 20,
            "max_tokens": 100_000,
            "max_wall_seconds": 60.0,
            "escalation_policy": "halt",
        }
        p = AgentProfile.from_dict(d)
        assert p.agent_id == "r1"
        assert p.role == "reviewer"
        assert p.coherence_threshold == 0.85

    def test_roundtrip(self):
        original = AgentProfile.for_role("coder", agent_id="c1")
        d = original.to_dict()
        restored = AgentProfile.from_dict(d)
        assert original == restored

    def test_from_dict_ignores_extra_keys(self):
        d = {"agent_id": "x", "role": "general", "extra_key": "ignored"}
        p = AgentProfile.from_dict(d)
        assert p.agent_id == "x"

    def test_to_dict_is_json_safe(self):
        import json

        p = AgentProfile.for_role("researcher", agent_id="j1")
        s = json.dumps(p.to_dict())
        assert "j1" in s
