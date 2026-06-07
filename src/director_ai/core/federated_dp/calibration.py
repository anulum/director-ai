# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Federated Differentially Private Calibration
"""Federated, differentially private calibration of a shared guardrail parameter.

Each tenant keeps its labelled data local and submits only a single clipped
scalar update (the direction it would move the shared calibration parameter — for
example the coherence threshold). The server averages the clipped updates with
added Gaussian noise (DP-SGD-style) behind a minimum-cohort gate, so the global
parameter improves without any tenant's raw data — or even its un-noised update —
being centralised, and no single tenant can dominate the round.

This complements the federated *aggregators* in
:mod:`director_ai.core.federated_privacy` (which aggregate counts and safety
signals) by aggregating *parameter updates* for online calibration.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

_SAFE_TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class CohortTooSmallError(RuntimeError):
    """Raised when a round has fewer contributing tenants than the cohort gate."""


def _validate_tenant_id(tenant_id: str) -> str:
    if not tenant_id or not _SAFE_TENANT_RE.fullmatch(tenant_id):
        raise ValueError(f"invalid tenant_id: {tenant_id!r}")
    return tenant_id


@dataclass(frozen=True)
class RoundResult:
    """The outcome of one federated calibration round."""

    previous_value: float
    new_value: float
    cohort_size: int
    clipped_mean: float
    noise_scale: float

    def to_dict(self) -> dict[str, float | int]:
        """Serialisable round summary (no per-tenant updates)."""
        return {
            "previous_value": self.previous_value,
            "new_value": self.new_value,
            "cohort_size": self.cohort_size,
            "clipped_mean": self.clipped_mean,
            "noise_scale": self.noise_scale,
        }


class FederatedCalibrationRound:
    """Aggregate clipped per-tenant updates into a DP global parameter step.

    Parameters
    ----------
    initial_value:
        Starting value of the shared parameter (e.g. the coherence threshold).
    clip_norm:
        Per-tenant update is clamped to ``[-clip_norm, clip_norm]`` before
        aggregation — this bounds any one tenant's influence (the DP
        sensitivity).
    noise_multiplier:
        Gaussian noise scale is ``noise_multiplier * clip_norm``; ``0`` disables
        noise (for testing the aggregation arithmetic only — not private).
    min_cohort:
        Minimum number of distinct contributing tenants before a round may be
        aggregated.
    learning_rate:
        Step size applied to the noisy mean update.
    value_bounds:
        ``(lo, hi)`` clamp for the updated parameter.
    seed:
        Optional deterministic seed for the DP noise (tests/simulations).
    """

    def __init__(
        self,
        initial_value: float,
        *,
        clip_norm: float = 0.1,
        noise_multiplier: float = 1.0,
        min_cohort: int = 3,
        learning_rate: float = 1.0,
        value_bounds: tuple[float, float] = (0.0, 1.0),
        seed: int | None = None,
    ) -> None:
        lo, hi = value_bounds
        if clip_norm <= 0:
            raise ValueError("clip_norm must be positive")
        if noise_multiplier < 0:
            raise ValueError("noise_multiplier must be non-negative")
        if min_cohort < 1:
            raise ValueError("min_cohort must be >= 1")
        if not lo <= initial_value <= hi:
            raise ValueError("initial_value must lie within value_bounds")
        self._value = float(initial_value)
        self._clip_norm = float(clip_norm)
        self._noise_multiplier = float(noise_multiplier)
        self._min_cohort = int(min_cohort)
        self._lr = float(learning_rate)
        self._bounds = (float(lo), float(hi))
        self._rng = random.Random(seed)  # noqa: S311 — DP noise, not crypto
        self._updates: dict[str, float] = {}
        self._rounds = 0

    @property
    def value(self) -> float:
        """The current shared parameter value."""
        return self._value

    @property
    def cohort_size(self) -> int:
        """Number of distinct tenants whose update is pending this round."""
        return len(self._updates)

    @property
    def rounds_applied(self) -> int:
        """Number of aggregated rounds applied so far."""
        return self._rounds

    def submit_update(self, *, tenant_id: str, update: float) -> None:
        """Submit one tenant's clipped local update for the pending round.

        Only the clipped scalar is retained; a tenant resubmitting overwrites its
        own update (one vote per tenant). The raw update is never stored.
        """
        tid = _validate_tenant_id(tenant_id)
        clipped = max(-self._clip_norm, min(self._clip_norm, float(update)))
        self._updates[tid] = clipped

    def aggregate(self) -> RoundResult:
        """Apply the DP-aggregated update to the shared parameter.

        Raises :class:`CohortTooSmallError` when fewer than ``min_cohort`` tenants
        contributed. Clears the pending updates so the next round starts fresh.
        """
        cohort = len(self._updates)
        if cohort < self._min_cohort:
            raise CohortTooSmallError(f"cohort {cohort} < required {self._min_cohort}")
        clipped_mean = sum(self._updates.values()) / cohort
        noise_scale = self._noise_multiplier * self._clip_norm
        noise = self._rng.gauss(0.0, noise_scale) if noise_scale > 0 else 0.0
        noisy_update = clipped_mean + noise / cohort
        previous = self._value
        lo, hi = self._bounds
        self._value = max(lo, min(hi, previous + self._lr * noisy_update))
        self._updates.clear()
        self._rounds += 1
        return RoundResult(
            previous_value=previous,
            new_value=self._value,
            cohort_size=cohort,
            clipped_mean=clipped_mean,
            noise_scale=noise_scale,
        )
