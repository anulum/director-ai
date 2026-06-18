# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Agent preflight policy

"""Policy controlling the five agent preflight hook points."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["PreflightPolicy"]


@dataclass(frozen=True)
class PreflightPolicy:
    """Bounds for the preflight hooks.

    Parameters
    ----------
    reversibility_threshold:
        An action whose reversibility score is below this is treated as
        irreversible and needs a safeguard (a registered rollback or a human
        acknowledgement) before it may proceed.
    require_human_ack_for_irreversible:
        When ``True``, an irreversible action with no registered rollback is
        blocked unless a human has acknowledged it.
    require_evidence_for_answer:
        When ``True``, a final answer with no supporting evidence is blocked.
    block_on_canary:
        When ``True``, a final answer is blocked if a canary tripped for it.
    allowed_handoff_targets:
        Agents a cross-agent handoff may target. Empty means unrestricted;
        non-empty restricts handoffs to the listed targets.
    """

    reversibility_threshold: float = 0.5
    require_human_ack_for_irreversible: bool = True
    require_evidence_for_answer: bool = True
    block_on_canary: bool = True
    allowed_handoff_targets: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate thresholds and normalise handoff targets."""
        if not 0.0 <= self.reversibility_threshold <= 1.0:
            raise ValueError("reversibility_threshold must be in [0, 1]")
        object.__setattr__(
            self,
            "allowed_handoff_targets",
            frozenset(t.strip() for t in self.allowed_handoff_targets if t.strip()),
        )

    def handoff_allowed(self, target: str) -> bool:
        """Return whether a handoff to ``target`` is permitted."""
        if not self.allowed_handoff_targets:
            return True
        return target.strip() in self.allowed_handoff_targets
