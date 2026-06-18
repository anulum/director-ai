# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Differentially Private Score Release

"""Optional DP release layer for externally exposed coherence scores."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .accountant import AccountantEntry, PrivacyAccountant
from .mechanisms import LaplaceMechanism

__all__ = ["DifferentialPrivacyScoreReleaser", "PrivacyScoreRelease"]


def _unit_interval(value: float, name: str) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return float(value)


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class PrivacyScoreRelease:
    """Tenant-safe differentially private score release record."""

    released_score: float
    noise: float
    epsilon_spent: float
    sensitivity: float
    mechanism: str
    label: str
    tenant_id: str = ""
    threshold: float | None = None
    raw_score_included: bool = False

    @property
    def approved_at_threshold(self) -> bool | None:
        """Return the threshold decision for the released score, if configured."""
        if self.threshold is None:
            return None
        return self.released_score >= self.threshold

    def to_dict(self) -> dict[str, Any]:
        """Return a tenant-safe payload with the raw score suppressed."""
        return {
            "released_score": self.released_score,
            "raw_score": None,
            "raw_score_included": self.raw_score_included,
            "noise": self.noise,
            "label": self.label,
            "tenant_id": self.tenant_id,
            "threshold": self.threshold,
            "approved_at_threshold": self.approved_at_threshold,
            "privacy": {
                "mechanism": self.mechanism,
                "epsilon": self.epsilon_spent,
                "sensitivity": self.sensitivity,
                "delta": 0.0,
            },
        }


class DifferentialPrivacyScoreReleaser:
    """Laplace mechanism wrapper for public or cross-tenant score release.

    Internal guard decisions should continue to use raw scores. This class
    exists for dashboards, analytics exports, public benchmark summaries, or
    other disclosure surfaces where releasing the exact score could leak
    membership information about the underlying retrieval corpus.
    """

    def __init__(
        self,
        *,
        epsilon: float,
        sensitivity: float = 1.0,
        accountant: PrivacyAccountant | None = None,
        seed: int | None = None,
        allow_insecure_seed: bool = False,
    ) -> None:
        if epsilon <= 0.0 or not math.isfinite(epsilon):
            raise ValueError("epsilon must be positive and finite")
        if sensitivity < 0.0 or not math.isfinite(sensitivity):
            raise ValueError("sensitivity must be non-negative and finite")
        self._mechanism = LaplaceMechanism(
            epsilon=epsilon,
            sensitivity=sensitivity,
            seed=seed,
            allow_insecure_seed=allow_insecure_seed,
        )
        self._accountant = accountant or PrivacyAccountant(max_epsilon=float("inf"))

    @property
    def epsilon(self) -> float:
        """Return the per-release privacy loss parameter."""
        return self._mechanism.epsilon

    @property
    def sensitivity(self) -> float:
        """Return the declared score sensitivity."""
        return self._mechanism.sensitivity

    def release_score(
        self,
        score: float,
        *,
        label: str,
        tenant_id: str = "",
        threshold: float | None = None,
    ) -> PrivacyScoreRelease:
        """Release a clamped, Laplace-noised coherence score."""
        raw_score = _unit_interval(score, "score")
        release_label = str(label).strip()
        if not release_label:
            raise ValueError("label must be non-empty")
        release_threshold = (
            None if threshold is None else _unit_interval(threshold, "threshold")
        )
        self._accountant.charge(
            AccountantEntry(
                label=release_label,
                epsilon=self.epsilon,
                delta=0.0,
                sensitivity=self.sensitivity,
                metadata={"surface": "score_release", "tenant_id": str(tenant_id)},
            )
        )
        noised = self._mechanism.apply(raw_score)
        released = _clamp_unit(noised)
        return PrivacyScoreRelease(
            released_score=released,
            noise=noised - raw_score,
            epsilon_spent=self.epsilon,
            sensitivity=self.sensitivity,
            mechanism="laplace",
            label=release_label,
            tenant_id=str(tenant_id),
            threshold=release_threshold,
        )
