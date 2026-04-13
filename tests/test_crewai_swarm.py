# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""STRONG tests for ``director_ai.integrations.crewai_swarm``.

Covers task guarding, crew output guarding, auto-quarantine,
quarantine blocking, statistics, and edge cases.
"""

from __future__ import annotations

from director_ai.agentic.agent_profile import AgentProfile
from director_ai.agentic.swarm_guardian import SwarmGuardian
from director_ai.integrations.crewai_swarm import (
    CrewGuardian,
    TaskGuardResult,
)


def _make_guardian() -> SwarmGuardian:
    g = SwarmGuardian(hallucination_threshold=0.5)
    g.register_agent(AgentProfile.for_role("researcher", agent_id="r1"))
    g.register_agent(AgentProfile.for_role("summariser", agent_id="w1"))
    return g


# ── Task output guarding ──────────────────────────────────────────────


class TestTaskGuarding:
    def test_approved_output(self):
        g = _make_guardian()
        cg = CrewGuardian(g)
        result = cg.guard_task_output(
            "r1",
            "Paris is the capital of France",
            context="Paris is the capital of France",
            next_agent_id="w1",
        )
        assert result.approved
        assert result.score < 0.5

    def test_rejected_output(self):
        g = _make_guardian()
        g.register_agent(
            AgentProfile(
                agent_id="strict",
                role="reviewer",
                coherence_threshold=0.99,
            )
        )
        cg = CrewGuardian(g)
        result = cg.guard_task_output(
            "strict",
            "completely unrelated gibberish xyz",
            context="actual factual context here",
            next_agent_id="w1",
        )
        assert not result.approved
        assert result.score > 0.3

    def test_auto_quarantine(self):
        g = _make_guardian()
        g.register_agent(
            AgentProfile(
                agent_id="bad",
                role="researcher",
                coherence_threshold=0.99,
            )
        )
        cg = CrewGuardian(g, auto_quarantine=True)
        cg.guard_task_output(
            "bad",
            "unrelated xyz abc",
            context="real context here",
        )
        assert g.is_quarantined("bad")

    def test_no_auto_quarantine(self):
        g = _make_guardian()
        g.register_agent(
            AgentProfile(
                agent_id="bad2",
                role="researcher",
                coherence_threshold=0.99,
            )
        )
        cg = CrewGuardian(g, auto_quarantine=False)
        cg.guard_task_output(
            "bad2",
            "unrelated xyz",
            context="real context",
        )
        assert not g.is_quarantined("bad2")

    def test_quarantined_agent_blocked(self):
        g = _make_guardian()
        g.quarantine_agent("r1", "test reason")
        cg = CrewGuardian(g)
        result = cg.guard_task_output(
            "r1",
            "any output",
            context="any context",
        )
        assert not result.approved
        assert result.quarantined
        assert "quarantined" in result.reason


# ── Crew output guarding ──────────────────────────────────────────────


class TestCrewOutput:
    def test_guard_crew_output(self):
        g = _make_guardian()
        cg = CrewGuardian(g)
        result = cg.guard_crew_output(
            crew_result="The capital of France is Paris.",
            final_agent_id="r1",
            context="Paris is the capital of France.",
        )
        assert result.approved

    def test_non_string_crew_result(self):
        g = _make_guardian()
        cg = CrewGuardian(g)
        result = cg.guard_crew_output(
            crew_result={"answer": "Paris"},
            final_agent_id="r1",
            context="Paris France",
        )
        assert isinstance(result, TaskGuardResult)


# ── Statistics ─────────────────────────────────────────────────────────


class TestStatistics:
    def test_initial_stats(self):
        g = _make_guardian()
        cg = CrewGuardian(g)
        assert cg.stats == {"guarded": 0, "blocked": 0, "approved": 0}

    def test_stats_after_guarding(self):
        g = _make_guardian()
        cg = CrewGuardian(g)
        cg.guard_task_output(
            "r1",
            "grounded content here",
            context="grounded content here context",
        )
        stats = cg.stats
        assert stats["guarded"] == 1
        assert stats["approved"] >= 0

    def test_blocked_count(self):
        g = _make_guardian()
        g.quarantine_agent("r1", "pre-quarantined")
        cg = CrewGuardian(g)
        cg.guard_task_output("r1", "blocked", context="ctx")
        assert cg.stats["blocked"] == 1


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_output(self):
        g = _make_guardian()
        cg = CrewGuardian(g)
        result = cg.guard_task_output("r1", "", context="ctx")
        assert isinstance(result, TaskGuardResult)

    def test_empty_context(self):
        g = _make_guardian()
        cg = CrewGuardian(g)
        result = cg.guard_task_output("r1", "output", context="")
        assert isinstance(result, TaskGuardResult)

    def test_dataclass(self):
        r = TaskGuardResult(True, "a", 0.1)
        assert r.approved
        assert r.reason == ""
