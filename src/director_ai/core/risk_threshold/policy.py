# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Risk threshold policy

"""How each risk factor maps to a threshold delta.

A positive delta raises the approval threshold (stricter: more grounding
required); a negative delta lowers it. The weights are explicit so the resulting
threshold is fully auditable and deterministic. Defaults are conservative —
risk raises the bar, and only a demonstrated high false-halt rate lowers it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["RiskThresholdPolicy"]


def _default_role_deltas() -> dict[str, float]:
    # Trusted, authenticated roles relax slightly; anonymous tightens.
    return {"admin": -0.05, "service": -0.03, "anonymous": 0.10}


def _default_domain_deltas() -> dict[str, float]:
    # High-stakes regulated domains tighten.
    return {"medical": 0.10, "finance": 0.10, "legal": 0.08, "general": 0.0}


@dataclass(frozen=True)
class RiskThresholdPolicy:
    """Base threshold, clamp bounds, and per-factor weights.

    Parameters
    ----------
    base_threshold:
        The threshold used when every factor is at its safe end.
    min_threshold, max_threshold:
        Clamp bounds for the adapted threshold.
    role_deltas, domain_deltas:
        Per-category deltas added when the factor matches a key.
    tenant_risk_weight:
        Multiplies ``tenant_risk`` (higher tenant risk → stricter).
    retrieval_weight:
        Multiplies ``1 - retrieval_confidence`` (weak retrieval → stricter).
    reversibility_weight:
        Multiplies ``1 - action_reversibility`` (irreversible → stricter).
    external_exposure_delta, pii_delta:
        Added when the respective boolean factor is set.
    freshness_weight:
        Multiplies ``1 - freshness`` (stale → stricter).
    historical_fpr_weight:
        Multiplies ``historical_fpr`` and is *subtracted* (high false-halt rate
        → relax to cut over-blocking).
    """

    base_threshold: float = 0.6
    min_threshold: float = 0.3
    max_threshold: float = 0.95
    role_deltas: dict[str, float] = field(default_factory=_default_role_deltas)
    domain_deltas: dict[str, float] = field(default_factory=_default_domain_deltas)
    tenant_risk_weight: float = 0.15
    retrieval_weight: float = 0.15
    reversibility_weight: float = 0.15
    external_exposure_delta: float = 0.05
    pii_delta: float = 0.08
    freshness_weight: float = 0.10
    historical_fpr_weight: float = 0.15

    def __post_init__(self) -> None:
        """Validate threshold bounds."""
        for name in ("base_threshold", "min_threshold", "max_threshold"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.min_threshold > self.max_threshold:
            raise ValueError("min_threshold must not exceed max_threshold")
