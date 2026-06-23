# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — AutoGen group chat guardian adapter
"""AutoGen swarm adapter: filter group chat messages via guardian.

Hooks into AutoGen group chat conversations to score each agent's
message before it is visible to other agents. Hallucinated messages
are suppressed and the offending agent is quarantined.

Usage::

    from director_ai.integrations.autogen_swarm import GroupChatGuardian
    from director_ai.agentic import AgentProfile, SwarmGuardian

    guardian = SwarmGuardian()
    guardian.register_agent(AgentProfile.for_role("researcher"))
    guardian.register_agent(AgentProfile.for_role("critic"))

    chat_guard = GroupChatGuardian(guardian)
    result = chat_guard.filter_message(
        sender="researcher-0",
        message="Paris is the capital of France.",
        chat_context="Discuss European capitals.",
    )
    if result.suppressed:
        print("Message blocked:", result.reason)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # advanced tier (director-ai-pro) — annotations only
    from director_ai.agentic.swarm_guardian import SwarmGuardian

logger = logging.getLogger("DirectorAI.AutoGenSwarm")

__all__ = ["AutoGenReplyGuard", "GroupChatGuardian", "MessageFilterResult"]


@dataclass(frozen=True)
class MessageFilterResult:
    """Result of filtering a group chat message."""

    suppressed: bool
    sender: str
    score: float
    reason: str = ""


class GroupChatGuardian:
    """Guard AutoGen group chat messages via SwarmGuardian.

    Parameters
    ----------
    guardian : SwarmGuardian
        The swarm guardian coordinating agents.
    auto_quarantine : bool
        Quarantine agent on suppression (default True).
    """

    def __init__(
        self,
        guardian: SwarmGuardian,
        auto_quarantine: bool = True,
    ) -> None:
        self._guardian = guardian
        self._auto_quarantine = auto_quarantine
        self._message_count = 0
        self._suppressed_count = 0

    def filter_message(
        self,
        sender: str,
        message: str,
        chat_context: str = "",
        recipients: list[str] | None = None,
    ) -> MessageFilterResult:
        """Filter a group chat message before delivery.

        Parameters
        ----------
        sender : str
            Agent ID of the message sender.
        message : str
            The message content.
        chat_context : str
            Accumulated chat context or topic description.
        recipients : list[str] | None
            Intended recipients (broadcast if None).
        """
        self._message_count += 1

        # Check quarantine
        if self._guardian.is_quarantined(sender):
            self._suppressed_count += 1
            return MessageFilterResult(
                suppressed=True,
                sender=sender,
                score=1.0,
                reason=f"sender {sender} is quarantined",
            )

        # Score against all recipients (or broadcast)
        target = (recipients or ["__broadcast__"])[0]
        result = self._guardian.score_handoff(
            from_agent=sender,
            to_agent=target,
            message=message,
            context=chat_context,
        )

        if result.should_halt:
            self._suppressed_count += 1
            reason = (
                "; ".join(result.reasons)
                if result.reasons
                else "hallucination detected"
            )

            if self._auto_quarantine:
                self._guardian.quarantine_agent(sender, reason=reason)

            return MessageFilterResult(
                suppressed=True,
                sender=sender,
                score=result.score,
                reason=reason,
            )

        return MessageFilterResult(
            suppressed=False,
            sender=sender,
            score=result.score,
        )

    @property
    def stats(self) -> dict[str, int]:
        """Message filtering statistics."""
        return {
            "messages": self._message_count,
            "suppressed": self._suppressed_count,
            "passed": self._message_count - self._suppressed_count,
        }


class AutoGenReplyGuard:
    """Install a dependency-light AutoGen ``register_reply`` guard.

    AutoGen 0.2/AG2-style ``ConversableAgent.register_reply`` accepts a
    function with signature ``(recipient, messages, sender, config)`` and a
    ``(handled, reply)`` return tuple. This wrapper suppresses unsafe incoming
    messages before the recipient's normal reply functions run, without
    importing AutoGen or depending on a specific package version.
    """

    def __init__(
        self,
        guardian: GroupChatGuardian,
        *,
        suppression_reply: str | dict[str, Any] | None = None,
        position: int = 0,
    ) -> None:
        self._guardian = guardian
        self._position = position
        self._suppression_reply = suppression_reply or {
            "role": "assistant",
            "content": "Message suppressed by swarm guardian.",
        }

    def install(
        self,
        agent: Any,
        *,
        trigger: Any = None,
        position: int | None = None,
        **register_kwargs: Any,
    ) -> Callable[..., Any]:
        """Register this guard on an AutoGen-compatible agent.

        Parameters mirror AutoGen's ``register_reply`` enough for stable use:
        ``trigger`` is forwarded unchanged, ``position`` defaults to ``0`` so
        the guard runs before normal auto-reply functions, and additional
        keyword arguments are passed through.
        """
        register_reply = getattr(agent, "register_reply", None)
        if not callable(register_reply):
            raise TypeError("agent must expose a callable register_reply() method")

        hook = self.reply_func
        register_reply(
            trigger,
            hook,
            position=self._position if position is None else position,
            **register_kwargs,
        )
        return hook

    def reply_func(
        self,
        recipient: Any,
        messages: list[dict[str, Any]] | None = None,
        sender: Any = None,
        config: dict[str, Any] | None = None,
    ) -> tuple[bool, str | dict[str, Any] | None]:
        """AutoGen-compatible reply function."""
        latest = _latest_autogen_message(messages or [])
        message = _autogen_message_content(latest)
        sender_id = _autogen_agent_name(sender, fallback="__unknown__")
        recipient_id = _autogen_agent_name(recipient, fallback="__broadcast__")
        chat_context = str((config or {}).get("chat_context", ""))

        result = self._guardian.filter_message(
            sender=sender_id,
            message=message,
            chat_context=chat_context,
            recipients=[recipient_id],
        )
        if result.suppressed:
            return True, self._suppression_reply
        return False, None


def _latest_autogen_message(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for message in reversed(messages):
        if isinstance(message, dict):
            return message
    return {}


def _autogen_message_content(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return " ".join(part for part in parts if part)
    return str(content)


def _autogen_agent_name(agent: Any, *, fallback: str) -> str:
    if agent is None:
        return fallback
    name = getattr(agent, "name", None)
    if name:
        return str(name)
    return str(agent)
