# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for ``director_ai.integrations.autogen_swarm``.

Covers message filtering, quarantine, statistics, and edge cases.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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

    def test_broadcast_target_and_fallback_reason_with_injected_guardian(self):
        calls = []

        class FakeGuardian:
            def is_quarantined(self, sender):
                return False

            def score_handoff(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    should_halt=True,
                    reasons=(),
                    score=0.87,
                )

            def quarantine_agent(self, sender, *, reason):
                calls.append({"quarantined": sender, "reason": reason})

        cg = GroupChatGuardian(FakeGuardian())

        result = cg.filter_message(
            "sender-a",
            "claim",
            chat_context="ctx",
            recipients=None,
        )

        assert result.suppressed
        assert result.reason == "hallucination detected"
        assert result.score == pytest.approx(0.87)
        assert calls[0] == {
            "from_agent": "sender-a",
            "to_agent": "__broadcast__",
            "message": "claim",
            "context": "ctx",
        }
        assert calls[1] == {
            "quarantined": "sender-a",
            "reason": "hallucination detected",
        }

    def test_filter_message_uses_first_recipient_only_for_group_routing(self):
        calls = []

        class FakeGuardian:
            def is_quarantined(self, sender):
                return False

            def score_handoff(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    should_halt=False,
                    reasons=(),
                    score=0.12,
                )

        cg = GroupChatGuardian(FakeGuardian())

        result = cg.filter_message(
            "sender-a",
            "claim",
            chat_context="ctx",
            recipients=["first", "second"],
        )

        assert not result.suppressed
        assert result.reason == ""
        assert calls[0]["to_agent"] == "first"


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
    def test_install_rejects_agent_without_register_reply(self):
        g = _make_guardian()
        cg = GroupChatGuardian(g)

        with pytest.raises(TypeError, match="register_reply"):
            AutoGenReplyGuard(cg).install(object())

    def test_install_registers_reply_guard_without_autogen_import(self):
        g = _make_guardian()
        cg = GroupChatGuardian(g)
        receiver = _FakeAutoGenAgent("c1")

        hook = AutoGenReplyGuard(cg).install(
            receiver,
            trigger=["r1", None],
            position=3,
            remove_other_reply_funcs=True,
        )

        assert hook is not None
        assert len(receiver.registered_replies) == 1
        args, kwargs = receiver.registered_replies[0]
        assert args[0] == ["r1", None]
        assert args[1] is hook
        assert kwargs["position"] == 3
        assert kwargs["remove_other_reply_funcs"] is True

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

    def test_reply_hook_returns_custom_suppression_reply_and_fallback_names(self):
        class FakeGuardian:
            def filter_message(self, **kwargs):
                self.kwargs = kwargs
                return MessageFilterResult(
                    suppressed=True,
                    sender=kwargs["sender"],
                    score=1.0,
                    reason="blocked",
                )

        fake_guardian = FakeGuardian()
        hook = AutoGenReplyGuard(
            fake_guardian,
            suppression_reply="blocked by policy",
        ).reply_func

        handled, reply = hook(
            object(),
            messages=[{"content": 123}],
            sender=None,
            config=None,
        )

        assert handled is True
        assert reply == "blocked by policy"
        assert fake_guardian.kwargs["sender"] == "__unknown__"
        assert fake_guardian.kwargs["recipients"][0].startswith("<object object at ")
        assert fake_guardian.kwargs["message"] == "123"
        assert fake_guardian.kwargs["chat_context"] == ""

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

    def test_reply_hook_uses_latest_dict_message_and_joins_text_blocks(self):
        class FakeGuardian:
            def filter_message(self, **kwargs):
                self.kwargs = kwargs
                return MessageFilterResult(False, kwargs["sender"], 0.0)

        fake_guardian = FakeGuardian()
        hook = AutoGenReplyGuard(fake_guardian).reply_func

        handled, reply = hook(
            "recipient-name",
            messages=[
                {"content": "ignored older"},
                "not a dict",
                {
                    "content": [
                        "plain",
                        {"type": "text", "text": "structured"},
                        {"type": "image_url", "image_url": {"url": "local"}},
                    ]
                },
            ],
            sender="sender-name",
            config={"chat_context": "ctx"},
        )

        assert handled is False
        assert reply is None
        assert fake_guardian.kwargs["message"] == "plain structured"
        assert fake_guardian.kwargs["sender"] == "sender-name"
        assert fake_guardian.kwargs["recipients"] == ["recipient-name"]


def test_latest_autogen_message_returns_empty_without_dict():
    from director_ai.integrations.autogen_swarm import _latest_autogen_message

    assert _latest_autogen_message([]) == {}
    assert _latest_autogen_message([1, "x", None]) == {}
