# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Execution-ring authorisation factors

"""The human authorisation factors a high-ring action must collect.

Each factor is an out-of-band human control: a prompt-injected agent cannot
forge an operator approval, wait out a cooling period, summon a second operator,
or notify the CISO. :class:`AuthorizationEvidence` is what a caller has actually
collected; :func:`satisfied_factors` reduces it (with the configured cooling
window) to the set of factors that genuinely hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = ["AuthorizationEvidence", "AuthorizationFactor", "satisfied_factors"]


class AuthorizationFactor(StrEnum):
    """An out-of-band human authorisation control."""

    OPERATOR_APPROVAL = "operator_approval"
    """A human operator explicitly approved the action."""

    COOLING_PERIOD = "cooling_period"
    """The mandatory delay since approval has fully elapsed."""

    SECOND_OPERATOR = "second_operator"
    """A distinct second operator independently approved (two-person rule)."""

    CISO_NOTIFICATION = "ciso_notification"
    """The security officer was notified before the action ran."""


@dataclass(frozen=True)
class AuthorizationEvidence:
    """What a caller has actually collected for an action.

    ``cooling_elapsed_seconds`` is the time since the first operator approval; the
    gate compares it against its configured cooling window. The two-person rule is
    enforced here: ``second_operator_approval`` only counts when a first
    ``operator_approval`` is also present.
    """

    operator_approval: bool = False
    second_operator_approval: bool = False
    ciso_notification: bool = False
    cooling_elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        """Validate the elapsed cooling-period evidence."""
        if self.cooling_elapsed_seconds < 0:
            raise ValueError("cooling_elapsed_seconds must be non-negative")


def satisfied_factors(
    evidence: AuthorizationEvidence,
    *,
    cooling_period_seconds: float,
) -> frozenset[AuthorizationFactor]:
    """Reduce collected evidence to the factors that genuinely hold."""
    held: set[AuthorizationFactor] = set()
    if evidence.operator_approval:
        held.add(AuthorizationFactor.OPERATOR_APPROVAL)
        # The cooling clock only has meaning once an approval has started it.
        if evidence.cooling_elapsed_seconds >= cooling_period_seconds:
            held.add(AuthorizationFactor.COOLING_PERIOD)
        # Two-person rule: a second approver only counts alongside the first.
        if evidence.second_operator_approval:
            held.add(AuthorizationFactor.SECOND_OPERATOR)
    if evidence.ciso_notification:
        held.add(AuthorizationFactor.CISO_NOTIFICATION)
    return frozenset(held)
