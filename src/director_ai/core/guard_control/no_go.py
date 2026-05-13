# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — no-go policy

"""Deterministic no-go policy for high-risk guard decisions."""

from __future__ import annotations

from dataclasses import dataclass

from .decision import GuardDecision


@dataclass(frozen=True)
class NoGoVerdict:
    """Result of applying :class:`NoGoPolicy` to a guard decision."""

    decision: str
    reason: str
    requires_human_review: bool
    original_decision: GuardDecision


class NoGoPolicy:
    """Block irreversible or threshold-exceeding decisions deterministically."""

    def __init__(
        self,
        *,
        default_threshold: float = 0.9,
        irreversible_threshold: float = 0.6,
        require_human_review_for_irreversible: bool = True,
    ) -> None:
        _validate_threshold("default_threshold", default_threshold)
        _validate_threshold("irreversible_threshold", irreversible_threshold)
        self._default_threshold = default_threshold
        self._irreversible_threshold = irreversible_threshold
        self._review_irreversible = require_human_review_for_irreversible

    def evaluate(self, decision: GuardDecision) -> NoGoVerdict:
        """Return the final deterministic no-go verdict."""
        envelope = decision.risk_envelope
        if (
            envelope.reversibility == "irreversible"
            and decision.risk_score >= self._irreversible_threshold
        ):
            return NoGoVerdict(
                decision="block",
                reason="no_go_irreversible_risk",
                requires_human_review=self._review_irreversible,
                original_decision=decision,
            )
        threshold = min(self._default_threshold, envelope.no_go_threshold)
        if decision.risk_score >= threshold:
            return NoGoVerdict(
                decision="block",
                reason="no_go_threshold_exceeded",
                requires_human_review=True,
                original_decision=decision,
            )
        return NoGoVerdict(
            decision=decision.decision,
            reason=decision.reason,
            requires_human_review=False,
            original_decision=decision,
        )


def _validate_threshold(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
