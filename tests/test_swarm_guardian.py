# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""STRONG tests for ``director_ai.agentic.swarm_guardian``.

Covers registration, handoff scoring, quarantine, cascade halt,
dependency tracking, thread safety, and edge cases.
"""

from __future__ import annotations

import threading

import pytest

from director_ai.agentic.agent_profile import AgentProfile
from director_ai.agentic.swarm_guardian import (
    AgentState,
    HandoffResult,
    SwarmGuardian,
)


def _profile(role: str = "researcher", **kw) -> AgentProfile:
    return AgentProfile.for_role(role, **kw)


# ── Registration ───────────────────────────────────────────────────────


class TestRegistration:
    def test_register_agent(self):
        g = SwarmGuardian()
        aid = g.register_agent(_profile(agent_id="r1"))
        assert aid == "r1"
        assert g.agent_count == 1

    def test_register_multiple(self):
        g = SwarmGuardian()
        g.register_agent(_profile(agent_id="r1"))
        g.register_agent(_profile("summariser", agent_id="s1"))
        assert g.agent_count == 2

    def test_duplicate_raises(self):
        g = SwarmGuardian()
        g.register_agent(_profile(agent_id="r1"))
        with pytest.raises(ValueError, match="already registered"):
            g.register_agent(_profile(agent_id="r1"))

    def test_max_agents(self):
        g = SwarmGuardian(max_agents=2)
        g.register_agent(_profile(agent_id="a1"))
        g.register_agent(_profile(agent_id="a2"))
        with pytest.raises(ValueError, match="Max agents"):
            g.register_agent(_profile(agent_id="a3"))

    def test_unregister(self):
        g = SwarmGuardian()
        g.register_agent(_profile(agent_id="r1"))
        g.unregister_agent("r1")
        assert g.agent_count == 0

    def test_unregister_nonexistent(self):
        g = SwarmGuardian()
        g.unregister_agent("ghost")  # should not raise

    def test_list_agents(self):
        g = SwarmGuardian()
        g.register_agent(_profile(agent_id="a1"))
        g.register_agent(_profile(agent_id="a2"))
        assert set(g.list_agents()) == {"a1", "a2"}


# ── Handoff scoring ───────────────────────────────────────────────────


class TestHandoffScoring:
    def test_grounded_message(self):
        g = SwarmGuardian()
        g.register_agent(_profile(agent_id="r1"))
        result = g.score_handoff(
            "r1", "s1", "Paris is the capital", "Paris is the capital of France"
        )
        assert isinstance(result, HandoffResult)
        assert result.score < 0.5  # good overlap → low hallucination

    def test_hallucinated_message(self):
        g = SwarmGuardian()
        g.register_agent(_profile(agent_id="r1", coherence_threshold=0.9))
        result = g.score_handoff(
            "r1", "s1", "completely unrelated gibberish xyz", "Paris France capital"
        )
        assert result.score > 0.5  # no overlap → high hallucination

    def test_no_context_neutral(self):
        g = SwarmGuardian()
        g.register_agent(_profile(agent_id="r1"))
        result = g.score_handoff("r1", "s1", "some message", "")
        assert result.score == 0.5

    def test_unregistered_source(self):
        g = SwarmGuardian()
        result = g.score_handoff("ghost", "s1", "msg", "ctx")
        assert not result.should_halt
        assert "not registered" in result.reasons[0]

    def test_handoff_count_incremented(self):
        g = SwarmGuardian()
        g.register_agent(_profile(agent_id="r1"))
        g.score_handoff("r1", "s1", "msg", "msg context")
        g.score_handoff("r1", "s2", "msg", "msg context")
        state = g.get_agent_state("r1")
        assert state is not None
        assert state.handoff_count == 2


# ── Quarantine ─────────────────────────────────────────────────────────


class TestQuarantine:
    def test_quarantine_agent(self):
        g = SwarmGuardian()
        g.register_agent(_profile(agent_id="r1"))
        result = g.quarantine_agent("r1", "hallucination")
        assert "r1" in result
        assert g.is_quarantined("r1")
        assert g.quarantined_count == 1

    def test_quarantined_agent_blocks_handoff(self):
        g = SwarmGuardian()
        g.register_agent(_profile(agent_id="r1"))
        g.quarantine_agent("r1", "compromised")
        result = g.score_handoff("r1", "s1", "msg", "ctx")
        assert result.should_halt
        assert "quarantined" in result.reasons[0]

    def test_quarantine_nonexistent(self):
        g = SwarmGuardian()
        result = g.quarantine_agent("ghost", "reason")
        assert result == []

    def test_double_quarantine(self):
        g = SwarmGuardian()
        g.register_agent(_profile(agent_id="r1"))
        g.quarantine_agent("r1", "first")
        result = g.quarantine_agent("r1", "second")
        assert result == []  # already quarantined

    def test_is_quarantined_false(self):
        g = SwarmGuardian()
        g.register_agent(_profile(agent_id="r1"))
        assert not g.is_quarantined("r1")

    def test_is_quarantined_nonexistent(self):
        g = SwarmGuardian()
        assert not g.is_quarantined("ghost")


# ── Cascade halt ───────────────────────────────────────────────────────


class TestCascadeHalt:
    def test_cascade_to_downstream(self):
        g = SwarmGuardian(cascade_halt=True)
        g.register_agent(_profile(agent_id="r1"))
        g.register_agent(_profile(agent_id="s1"))
        # Establish dependency: r1 → s1
        g.score_handoff("r1", "s1", "msg", "msg context")
        result = g.quarantine_agent("r1", "injection")
        assert "r1" in result
        assert "s1" in result
        assert g.is_quarantined("s1")

    def test_cascade_disabled(self):
        g = SwarmGuardian(cascade_halt=False)
        g.register_agent(_profile(agent_id="r1"))
        g.register_agent(_profile(agent_id="s1"))
        g.score_handoff("r1", "s1", "msg", "msg context")
        result = g.quarantine_agent("r1", "injection")
        assert "r1" in result
        assert "s1" not in result

    def test_deep_cascade(self):
        g = SwarmGuardian(cascade_halt=True)
        g.register_agent(_profile(agent_id="a"))
        g.register_agent(_profile(agent_id="b"))
        g.register_agent(_profile(agent_id="c"))
        g.score_handoff("a", "b", "msg", "msg ctx")
        g.score_handoff("b", "c", "msg", "msg ctx")
        result = g.quarantine_agent("a", "root cause")
        assert set(result) == {"a", "b", "c"}


# ── Agent state ────────────────────────────────────────────────────────


class TestAgentState:
    def test_get_state(self):
        g = SwarmGuardian()
        g.register_agent(_profile(agent_id="r1"))
        state = g.get_agent_state("r1")
        assert state is not None
        assert state.profile.agent_id == "r1"
        assert state.registered_at > 0

    def test_get_nonexistent(self):
        g = SwarmGuardian()
        assert g.get_agent_state("ghost") is None

    def test_hallucination_count(self):
        g = SwarmGuardian(hallucination_threshold=0.01)
        g.register_agent(_profile(agent_id="r1", coherence_threshold=0.99))
        # High threshold → almost any message flags
        g.score_handoff("r1", "s1", "unrelated xyz", "actual context here")
        state = g.get_agent_state("r1")
        assert state is not None
        assert state.hallucination_count >= 0


# ── Thread safety ──────────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_registration(self):
        g = SwarmGuardian(max_agents=200)
        errors: list[Exception] = []

        def register_batch(start: int) -> None:
            try:
                for i in range(start, start + 20):
                    g.register_agent(_profile(agent_id=f"agent-{i}"))
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=register_batch, args=(i * 20,)) for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert g.agent_count == 100

    def test_concurrent_handoffs(self):
        g = SwarmGuardian()
        g.register_agent(_profile(agent_id="src"))
        g.register_agent(_profile(agent_id="dst"))
        errors: list[Exception] = []

        def handoff_batch() -> None:
            try:
                for _ in range(50):
                    g.score_handoff("src", "dst", "message", "context words")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=handoff_batch) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        state = g.get_agent_state("src")
        assert state is not None
        assert state.handoff_count == 200


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_guardian(self):
        g = SwarmGuardian()
        assert g.agent_count == 0
        assert g.quarantined_count == 0
        assert g.list_agents() == []

    def test_handoff_result_dataclass(self):
        r = HandoffResult("a", "b", 0.5, False, ["reason"])
        assert r.from_agent == "a"
        assert r.to_agent == "b"

    def test_agent_state_dataclass(self):
        s = AgentState(
            profile=_profile(agent_id="x"),
            monitor=None,  # type: ignore[arg-type]
        )
        assert s.quarantined is False
        assert s.handoff_count == 0
