# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — CrewAI swarm guardian tests
"""Tests for ``director_ai.integrations.crewai_swarm``.

Covers task guarding, crew output guarding, auto-quarantine,
quarantine blocking, statistics, and edge cases.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from director_ai.agentic.agent_profile import AgentProfile
from director_ai.agentic.handoff_scorer import HandoffScorer
from director_ai.agentic.swarm_guardian import HandoffResult, SwarmGuardian
from director_ai.integrations.crewai_swarm import (
    CrewGuardian,
    TaskGuardResult,
)


def _make_guardian() -> SwarmGuardian:
    """Register the two real agents used by the handoff scenarios."""
    g = SwarmGuardian(hallucination_threshold=0.5)
    g.register_agent(AgentProfile.for_role("researcher", agent_id="r1"))
    g.register_agent(AgentProfile.for_role("summariser", agent_id="w1"))
    return g


# ── Task output guarding ──────────────────────────────────────────────


class TestTaskGuarding:
    """Exercise real guardian approval, refusal and quarantine."""

    def test_approved_output(self) -> None:
        """Approve an output grounded in the supplied handoff context."""
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

    def test_rejected_output(self) -> None:
        """Reject an ungrounded output under the registered agent policy."""
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

    def test_auto_quarantine(self) -> None:
        """Quarantine a source rejected by the actual guardian."""
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

    def test_no_auto_quarantine(self) -> None:
        """Retain a rejected source when automatic quarantine is disabled."""
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

    def test_quarantined_agent_blocked(self) -> None:
        """Block a quarantined source before another handoff can execute."""
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
    """Exercise final-output conversion and guardian routing."""

    def test_guard_crew_output(self) -> None:
        """Review a final crew output through the same guardian path."""
        g = _make_guardian()
        cg = CrewGuardian(g)
        result = cg.guard_crew_output(
            crew_result="The capital of France is Paris.",
            final_agent_id="r1",
            context="Paris is the capital of France.",
        )
        assert result.approved

    def test_non_string_crew_result(self) -> None:
        """Stringify a structured final result before guarding it."""
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
    """Keep observable wrapper counts consistent with guarded outputs."""

    def test_initial_stats(self) -> None:
        """Start with no guarded, blocked or approved outputs."""
        g = _make_guardian()
        cg = CrewGuardian(g)
        assert cg.stats == {"guarded": 0, "blocked": 0, "approved": 0}

    def test_stats_after_guarding(self) -> None:
        """Count a completed actual handoff in the wrapper statistics."""
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

    def test_blocked_count(self) -> None:
        """Count the refusal of a previously quarantined agent."""
        g = _make_guardian()
        g.quarantine_agent("r1", "pre-quarantined")
        cg = CrewGuardian(g)
        cg.guard_task_output("r1", "blocked", context="ctx")
        assert cg.stats["blocked"] == 1


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    """Preserve empty-input and result-container contracts."""

    def test_empty_output(self) -> None:
        """Return a guardian verdict for an empty output."""
        g = _make_guardian()
        cg = CrewGuardian(g)
        result = cg.guard_task_output("r1", "", context="ctx")
        assert isinstance(result, TaskGuardResult)

    def test_empty_context(self) -> None:
        """Return a guardian verdict without grounding context."""
        g = _make_guardian()
        cg = CrewGuardian(g)
        result = cg.guard_task_output("r1", "output", context="")
        assert isinstance(result, TaskGuardResult)

    def test_dataclass(self) -> None:
        """Retain the result defaults consumed by callers."""
        r = TaskGuardResult(True, "a", 0.1)
        assert r.approved
        assert r.reason == ""


class TestSuppliedScorer:
    """Respect supplied scoring policy without losing guardian bookkeeping."""

    @pytest.mark.parametrize(
        ("profile_threshold", "scorer_threshold", "approved"),
        [(0.9, 0.95, True), (0.1, 0.2, False), (0.1, 0.75, True), (0.9, 0.74, False)],
    )
    def test_supplied_scorer_controls_handoff(
        self,
        profile_threshold: float,
        scorer_threshold: float,
        approved: bool,
    ) -> None:
        """Use the supplied threshold, including equality, on real handoffs.

        Parameters
        ----------
        profile_threshold : float
            Registered agent's default coherence threshold.
        scorer_threshold : float
            Explicit scorer's maximum accepted divergence.
        approved : bool
            Expected verdict for the actual three-quarter divergence.
        """
        guardian = SwarmGuardian()
        guardian.register_agent(
            AgentProfile(
                agent_id="source",
                role="researcher",
                coherence_threshold=profile_threshold,
            ),
        )
        scorer = HandoffScorer(threshold=scorer_threshold)
        wrapper = CrewGuardian(guardian, scorer=scorer, auto_quarantine=False)

        result = wrapper.guard_task_output(
            "source",
            "alpha beta gamma delta",
            context="alpha",
            next_agent_id="sink",
        )

        assert result.approved is approved
        assert result.score == 0.75
        assert not result.quarantined
        assert len(scorer.history) == 1
        recorded = scorer.history[0]
        assert (recorded.from_agent, recorded.to_agent) == ("source", "sink")
        assert recorded.grounded is approved
        state = guardian.get_agent_state("source")
        assert state is not None
        assert state.handoff_count == 1
        assert state.hallucination_count == int(not approved)
        assert wrapper.stats == {
            "guarded": 1,
            "blocked": int(not approved),
            "approved": int(approved),
        }

    def test_absent_scorer_preserves_agent_threshold(self) -> None:
        """Keep the agent policy instead of constructing a default scorer."""
        guardian = SwarmGuardian()
        guardian.register_agent(
            AgentProfile(agent_id="source", role="researcher", coherence_threshold=0.1),
        )
        wrapper = CrewGuardian(guardian)
        result = wrapper.guard_task_output("source", "alpha beta gamma delta", "alpha")
        assert result.approved
        assert result.score == 0.75
        assert not guardian.is_quarantined("source")

    def test_final_crew_output_uses_supplied_scorer(self) -> None:
        """Record the supplied policy and final-output destination."""
        guardian = _make_guardian()
        scorer = HandoffScorer(threshold=0.95)
        wrapper = CrewGuardian(guardian, scorer=scorer)
        result = wrapper.guard_crew_output("alpha beta gamma delta", "r1", "alpha")
        assert result.approved
        assert len(scorer.history) == 1
        assert scorer.history[0].to_agent == "__output__"
        assert wrapper.stats == {"guarded": 1, "blocked": 0, "approved": 1}

    def test_quarantined_source_does_not_invoke_scorer(self) -> None:
        """Keep quarantine authoritative even with a permissive scorer."""
        guardian = _make_guardian()
        guardian.quarantine_agent("r1", "operator stop")
        scorer = HandoffScorer(threshold=1.0)
        wrapper = CrewGuardian(guardian, scorer=scorer)
        result = wrapper.guard_task_output("r1", "alpha", "alpha")
        assert not result.approved
        assert result.quarantined
        assert scorer.history == []
        state = guardian.get_agent_state("r1")
        assert state is not None
        assert state.handoff_count == 0
        assert wrapper.stats == {"guarded": 1, "blocked": 1, "approved": 0}

    def test_supplied_scorer_retains_dependency_quarantine(self) -> None:
        """Preserve actual dependency tracking and downstream quarantine."""
        guardian = _make_guardian()
        guardian.score_handoff("r1", "w1", "shared", "shared")
        scorer = HandoffScorer(threshold=0.2)
        wrapper = CrewGuardian(guardian, scorer=scorer)
        result = wrapper.guard_task_output(
            "r1", "alpha beta gamma delta", "alpha", "w1"
        )
        assert not result.approved
        assert result.quarantined
        assert guardian.is_quarantined("r1")
        assert guardian.is_quarantined("w1")
        state = guardian.get_agent_state("r1")
        assert state is not None
        assert state.handoff_count == 2
        assert state.hallucination_count == 1
        assert len(scorer.history) == 1
        assert "threshold 0.200" in result.reason

    def test_halted_handoff_without_diagnostics_has_reason(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Keep a refusal reason when the provider omits optional diagnostics.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Boundary-response transformation; the actual guardian still scores
            and records the handoff before diagnostics are removed.
        """
        guardian = _make_guardian()
        original_score = guardian.score_handoff

        def without_reasons(
            from_agent: str,
            to_agent: str,
            message: str,
            context: str = "",
            *,
            scorer: HandoffScorer | None = None,
        ) -> HandoffResult:
            """Retain the real handoff verdict and state, omit its diagnostics."""
            result = original_score(
                from_agent, to_agent, message, context, scorer=scorer
            )
            return replace(result, reasons=[])

        monkeypatch.setattr(guardian, "score_handoff", without_reasons)
        scorer = HandoffScorer(threshold=0.2)
        wrapper = CrewGuardian(guardian, scorer=scorer, auto_quarantine=False)
        result = wrapper.guard_task_output("r1", "alpha beta gamma delta", "alpha")
        assert not result.approved
        assert result.reason == "score exceeded threshold"
        assert not result.quarantined
        assert len(scorer.history) == 1
        state = guardian.get_agent_state("r1")
        assert state is not None
        assert state.handoff_count == 1
        assert state.hallucination_count == 1
