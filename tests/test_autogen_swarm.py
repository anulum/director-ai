# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for ``director_ai.integrations.autogen_swarm``.

Covers message filtering, quarantine, statistics, and edge cases.
"""

from __future__ import annotations

from director_ai.agentic.agent_profile import AgentProfile
from director_ai.agentic.swarm_guardian import SwarmGuardian
from director_ai.integrations.autogen_swarm import (
    AutoGenReplyGuard,
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


# ── AutoGen register_reply hook ───────────────────────────────────────


class _FakeAutoGenAgent:
    def __init__(self, name: str):
        self.name = name
        self.registered_replies = []

    def register_reply(self, *args, **kwargs):
        self.registered_replies.append((args, kwargs))


class TestAutoGenReplyGuard:
    def test_install_registers_reply_guard_without_autogen_import(self):
        g = _make_guardian()
        cg = GroupChatGuardian(g)
        receiver = _FakeAutoGenAgent("c1")

        hook = AutoGenReplyGuard(cg).install(receiver, trigger=["r1", None])

        assert hook is not None
        assert len(receiver.registered_replies) == 1
        args, kwargs = receiver.registered_replies[0]
        assert args[0] == ["r1", None]
        assert args[1] is hook
        assert kwargs["position"] == 0

    def test_reply_hook_allows_grounded_message_to_continue(self):
        g = _make_guardian()
        cg = GroupChatGuardian(g)
        sender = _FakeAutoGenAgent("r1")
        recipient = _FakeAutoGenAgent("c1")
        hook = AutoGenReplyGuard(cg).reply_func

        handled, reply = hook(
            recipient,
            messages=[{"content": "Paris is the capital of France"}],
            sender=sender,
            config={"chat_context": "Paris is the capital of France topic"},
        )

        assert handled is False
        assert reply is None
        assert cg.stats == {"messages": 1, "suppressed": 0, "passed": 1}

    def test_reply_hook_suppresses_hallucinated_message(self):
        g = _make_guardian()
        g.register_agent(
            AgentProfile(
                agent_id="strict",
                role="reviewer",
                coherence_threshold=0.99,
            )
        )
        cg = GroupChatGuardian(g)
        sender = _FakeAutoGenAgent("strict")
        recipient = _FakeAutoGenAgent("c1")
        hook = AutoGenReplyGuard(cg).reply_func

        handled, reply = hook(
            recipient,
            messages=[{"content": "completely unrelated gibberish xyz"}],
            sender=sender,
            config={"chat_context": "actual topic here"},
        )

        assert handled is True
        assert reply == {
            "role": "assistant",
            "content": "Message suppressed by swarm guardian.",
        }
        assert g.is_quarantined("strict")

    def test_reply_hook_extracts_text_from_multimodal_blocks(self):
        g = _make_guardian()
        cg = GroupChatGuardian(g)
        sender = _FakeAutoGenAgent("r1")
        recipient = _FakeAutoGenAgent("c1")
        hook = AutoGenReplyGuard(cg).reply_func

        handled, reply = hook(
            recipient,
            messages=[
                {
                    "content": [
                        {"type": "text", "text": "Paris is the capital of France"},
                        {"type": "image_url", "image_url": {"url": "local"}},
                    ],
                }
            ],
            sender=sender,
            config={"chat_context": "Paris is the capital of France topic"},
        )

        assert handled is False
        assert reply is None
