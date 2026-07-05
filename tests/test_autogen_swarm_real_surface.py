# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - AutoGen swarm real-surface tests
"""Real public-surface coverage for the AutoGen swarm adapter."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from director_ai.agentic.agent_profile import AgentProfile
from director_ai.agentic.swarm_guardian import SwarmGuardian
from director_ai.integrations.autogen_swarm import AutoGenReplyGuard, GroupChatGuardian
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

AutoGenReply = str | dict[str, Any] | None
AutoGenReplyHook = Callable[
    [Any, list[dict[str, Any]] | None, Any, dict[str, Any] | None],
    tuple[bool, AutoGenReply],
]


@dataclass(frozen=True)
class RegisteredReply:
    """AutoGen-compatible reply registration captured by the protocol harness."""

    trigger: object
    reply_func: AutoGenReplyHook
    position: int
    remove_other_reply_funcs: bool


class AutoGenCompatibleAgent:
    """Small AutoGen ``register_reply`` protocol harness for adapter tests."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._registered_replies: list[RegisteredReply] = []

    @property
    def registered_replies(self) -> tuple[RegisteredReply, ...]:
        """Return installed reply hooks in their registered order."""
        return tuple(self._registered_replies)

    def register_reply(
        self,
        trigger: object,
        reply_func: AutoGenReplyHook,
        *,
        position: int = 0,
        remove_other_reply_funcs: bool = False,
    ) -> None:
        """Store a reply hook using AutoGen's public registration shape."""
        self._registered_replies.append(
            RegisteredReply(
                trigger=trigger,
                reply_func=reply_func,
                position=position,
                remove_other_reply_funcs=remove_other_reply_funcs,
            )
        )

    def receive(
        self,
        *,
        sender: object,
        content: object,
        chat_context: str,
    ) -> tuple[bool, AutoGenReply]:
        """Execute registered reply hooks the way an AutoGen recipient would."""
        messages: list[dict[str, Any]] = [{"content": content}]
        config = {"chat_context": chat_context}
        for registered in sorted(
            self._registered_replies,
            key=lambda candidate: candidate.position,
        ):
            handled, reply = registered.reply_func(self, messages, sender, config)
            if handled:
                return handled, reply
        return False, None


def _swarm_guardian() -> SwarmGuardian:
    """Build a real swarm guardian with grounded and strict senders."""
    guardian = SwarmGuardian(hallucination_threshold=0.5)
    guardian.register_agent(AgentProfile.for_role("researcher", agent_id="r1"))
    guardian.register_agent(AgentProfile.for_role("summariser", agent_id="c1"))
    guardian.register_agent(
        AgentProfile(
            agent_id="strict",
            role="reviewer",
            coherence_threshold=0.99,
        )
    )
    return guardian


def test_autogen_swarm_unit_guard_declares_real_surface_companion() -> None:
    """The helper-heavy AutoGen guard should name this public companion."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_autogen_swarm.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_autogen_swarm_real_surface.py" in category


def test_autogen_reply_guard_installs_and_suppresses_unsafe_reply() -> None:
    """Registered AutoGen hooks should suppress unsafe cross-agent messages."""
    guardian = _swarm_guardian()
    chat_guard = GroupChatGuardian(guardian)
    recipient = AutoGenCompatibleAgent("c1")
    sender = AutoGenCompatibleAgent("strict")
    suppression_reply = {"role": "assistant", "content": "blocked by guardian"}

    hook = AutoGenReplyGuard(
        chat_guard,
        suppression_reply=suppression_reply,
    ).install(
        recipient,
        trigger=sender,
        position=-10,
        remove_other_reply_funcs=True,
    )
    handled, reply = recipient.receive(
        sender=sender,
        content="completely unrelated gibberish xyz",
        chat_context="actual topic here",
    )

    registration = recipient.registered_replies[0]
    assert registration.trigger is sender
    assert registration.reply_func is hook
    assert registration.position == -10
    assert registration.remove_other_reply_funcs is True
    assert handled is True
    assert reply == suppression_reply
    assert guardian.is_quarantined("strict") is True
    assert chat_guard.stats == {"messages": 1, "suppressed": 1, "passed": 0}


def test_autogen_reply_guard_allows_grounded_multimodal_text_blocks() -> None:
    """Grounded text blocks should pass through the installed AutoGen hook."""
    guardian = _swarm_guardian()
    chat_guard = GroupChatGuardian(guardian)
    recipient = AutoGenCompatibleAgent("c1")
    sender = AutoGenCompatibleAgent("r1")
    AutoGenReplyGuard(chat_guard).install(recipient, trigger=None)

    handled, reply = recipient.receive(
        sender=sender,
        content=[
            {"type": "text", "text": "Paris is the capital of France"},
            {"type": "image_url", "image_url": {"url": "local-only"}},
        ],
        chat_context="Paris is the capital of France topic",
    )

    assert handled is False
    assert reply is None
    assert guardian.is_quarantined("r1") is False
    assert chat_guard.stats == {"messages": 1, "suppressed": 0, "passed": 1}
