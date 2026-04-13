# SPDX-License-Identifier: AGPL-3.0-or-later
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
from dataclasses import dataclass

from director_ai.agentic.swarm_guardian import SwarmGuardian

logger = logging.getLogger("DirectorAI.AutoGenSwarm")

__all__ = ["GroupChatGuardian", "MessageFilterResult"]


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
