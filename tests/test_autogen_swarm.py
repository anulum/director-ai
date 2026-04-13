# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""STRONG tests for ``director_ai.integrations.autogen_swarm``.

Covers message filtering, quarantine, statistics, and edge cases.
"""

from __future__ import annotations

from director_ai.agentic.agent_profile import AgentProfile
from director_ai.agentic.swarm_guardian import SwarmGuardian
from director_ai.integrations.autogen_swarm import (
    GroupChatGuardian,
    MessageFilterResult,
)


def _make_guardian() -> SwarmGuardian:
    g = SwarmGuardian(hallucination_threshold=0.5)
    g.register_agent(AgentProfile.for_role("researcher", agent_id="r1"))
    g.register_agent(AgentProfile.for_role("summariser", agent_id="c1"))
    return g


# ── Message filtering ─────────────────────────────────────────────────


class TestMessageFilter:
    def test_grounded_message_passes(self):
        g = _make_guardian()
        cg = GroupChatGuardian(g)
        result = cg.filter_message(
            "r1",
            "Paris is the capital of France",
            chat_context="Paris is the capital of France topic",
        )
        assert not result.suppressed

    def test_hallucinated_message_suppressed(self):
        g = _make_guardian()
        g.register_agent(
            AgentProfile(
                agent_id="strict",
                role="reviewer",
                coherence_threshold=0.99,
            )
        )
        cg = GroupChatGuardian(g)
        result = cg.filter_message(
            "strict",
            "completely unrelated gibberish xyz",
            chat_context="actual topic here",
        )
        assert result.suppressed
        assert result.reason != ""

    def test_quarantined_sender_blocked(self):
        g = _make_guardian()
        g.quarantine_agent("r1", "pre-quarantined")
        cg = GroupChatGuardian(g)
        result = cg.filter_message("r1", "any message", chat_context="ctx")
        assert result.suppressed
        assert "quarantined" in result.reason

    def test_auto_quarantine(self):
        g = _make_guardian()
        g.register_agent(
            AgentProfile(
                agent_id="bad",
                role="researcher",
                coherence_threshold=0.99,
            )
        )
        cg = GroupChatGuardian(g, auto_quarantine=True)
        cg.filter_message("bad", "unrelated xyz", chat_context="real context")
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
        cg = GroupChatGuardian(g, auto_quarantine=False)
        cg.filter_message("bad2", "xyz", chat_context="real context")
        assert not g.is_quarantined("bad2")

    def test_with_recipients(self):
        g = _make_guardian()
        cg = GroupChatGuardian(g)
        result = cg.filter_message(
            "r1",
            "shared topic content here",
            chat_context="shared topic content",
            recipients=["c1"],
        )
        assert isinstance(result, MessageFilterResult)


# ── Statistics ─────────────────────────────────────────────────────────


class TestStatistics:
    def test_initial_stats(self):
        g = _make_guardian()
        cg = GroupChatGuardian(g)
        assert cg.stats == {"messages": 0, "suppressed": 0, "passed": 0}

    def test_counts_after_messages(self):
        g = _make_guardian()
        cg = GroupChatGuardian(g)
        cg.filter_message("r1", "msg1", chat_context="msg1 context")
        cg.filter_message("r1", "msg2", chat_context="msg2 context")
        assert cg.stats["messages"] == 2


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_message(self):
        g = _make_guardian()
        cg = GroupChatGuardian(g)
        result = cg.filter_message("r1", "", chat_context="ctx")
        assert isinstance(result, MessageFilterResult)

    def test_dataclass(self):
        r = MessageFilterResult(False, "a", 0.1)
        assert not r.suppressed
        assert r.reason == ""
