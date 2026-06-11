# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Risk-adaptive threshold

"""Compute a per-request approval threshold from a risk profile.

:class:`RiskAdaptiveThreshold` maps a :class:`RiskFactors` profile onto a
threshold by summing the per-factor deltas defined in the policy and clamping to
the policy bounds. The returned :class:`RiskThresholdDecision` lists every
factor's contribution, so why a request was held to a stricter (or looser) bar
is fully auditable — and the function is pure, so the same profile always yields
the same threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .factors import RiskFactors
from .policy import RiskThresholdPolicy

__all__ = ["RiskAdaptiveThreshold", "RiskThresholdDecision"]


@dataclass(frozen=True)
class RiskThresholdDecision:
    """The adapted threshold and the contributions that produced it.

    Parameters
    ----------
    threshold:
        The clamped, adapted approval threshold.
    base_threshold:
        The policy's base threshold before adjustment.
    contributions:
        Per-factor delta applied, by factor name (positive = stricter).
    """

    threshold: float
    base_threshold: float
    contributions: dict[str, float] = field(default_factory=dict)

    @property
    def total_delta(self) -> float:
        """Sum of the per-factor contributions (before clamping)."""
        return sum(self.contributions.values())

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "threshold": self.threshold,
            "base_threshold": self.base_threshold,
            "total_delta": self.total_delta,
            "contributions": dict(self.contributions),
        }


class RiskAdaptiveThreshold:
    """Map a :class:`RiskFactors` profile to an approval threshold.

    Parameters
    ----------
    policy:
        The :class:`RiskThresholdPolicy` of base threshold, bounds, and weights.
    """

    def __init__(self, policy: RiskThresholdPolicy | None = None) -> None:
        self.policy = policy or RiskThresholdPolicy()

    def evaluate(self, factors: RiskFactors) -> RiskThresholdDecision:
        """Return the adapted threshold and its per-factor contributions."""
        policy = self.policy
        contributions: dict[str, float] = {}

        role_delta = policy.role_deltas.get(factors.user_role, 0.0)
        if role_delta:
            contributions["user_role"] = role_delta

        domain_delta = policy.domain_deltas.get(factors.domain, 0.0)
        if domain_delta:
            contributions["domain"] = domain_delta

        if factors.tenant_risk:
            contributions["tenant_risk"] = (
                factors.tenant_risk * policy.tenant_risk_weight
            )
        if factors.retrieval_confidence < 1.0:
            contributions["retrieval_confidence"] = (
                1.0 - factors.retrieval_confidence
            ) * policy.retrieval_weight
        if factors.action_reversibility < 1.0:
            contributions["action_reversibility"] = (
                1.0 - factors.action_reversibility
            ) * policy.reversibility_weight
        if factors.external_exposure:
            contributions["external_exposure"] = policy.external_exposure_delta
        if factors.pii_present:
            contributions["pii_present"] = policy.pii_delta
        if factors.freshness < 1.0:
            contributions["freshness"] = (
                1.0 - factors.freshness
            ) * policy.freshness_weight
        if factors.historical_fpr:
            contributions["historical_fpr"] = (
                -factors.historical_fpr * policy.historical_fpr_weight
            )

        raw = policy.base_threshold + sum(contributions.values())
        threshold = max(policy.min_threshold, min(policy.max_threshold, raw))
        return RiskThresholdDecision(
            threshold=round(threshold, 4),
            base_threshold=policy.base_threshold,
            contributions={k: round(v, 4) for k, v in contributions.items()},
        )
