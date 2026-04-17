# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Laplace and Gaussian mechanisms

"""Two differential-privacy mechanisms.

:class:`LaplaceMechanism` adds noise drawn from
``Laplace(0, sensitivity / ε)``. The output obeys
``ε``-differential privacy against neighbouring datasets that
differ by at most ``sensitivity`` in the query value.

:class:`GaussianMechanism` adds zero-mean Gaussian noise with
``σ = sensitivity · √(2 · ln(1.25 / δ)) / ε``. It provides
``(ε, δ)``-DP and is a better fit for L2-sensitivity queries
over vector outputs.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class _MechanismParams:
    epsilon: float
    delta: float
    sensitivity: float


class LaplaceMechanism:
    """Pure ``ε``-DP via Laplace noise.

    Parameters
    ----------
    epsilon :
        The privacy loss parameter. Must be strictly positive;
        smaller values mean stronger privacy.
    sensitivity :
        Upper bound on how much a single individual can change
        the query value. Must be non-negative.
    seed :
        Optional integer seed for the internal RNG so CI runs
        are reproducible. Without a seed the mechanism reads
        from the system entropy pool every call.
    """

    def __init__(
        self,
        *,
        epsilon: float,
        sensitivity: float,
        seed: int | None = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if sensitivity < 0:
            raise ValueError("sensitivity must be non-negative")
        self._params = _MechanismParams(
            epsilon=epsilon, delta=0.0, sensitivity=sensitivity
        )
        self._rng = random.Random(seed) if seed is not None else random.SystemRandom()

    @property
    def epsilon(self) -> float:
        return self._params.epsilon

    @property
    def sensitivity(self) -> float:
        return self._params.sensitivity

    @property
    def scale(self) -> float:
        """The Laplace ``b`` parameter. Equals
        ``sensitivity / epsilon`` (and ``0.0`` when the query has
        zero sensitivity — a trivial query that needs no noise)."""
        if self._params.sensitivity == 0.0:
            return 0.0
        return self._params.sensitivity / self._params.epsilon

    def noise(self) -> float:
        """Draw one Laplace sample with mean 0 and scale
        :attr:`scale`."""
        scale = self.scale
        if scale == 0.0:
            return 0.0
        u = self._rng.uniform(-0.5, 0.5)
        return -scale * _signum(u) * math.log(1.0 - 2.0 * abs(u))

    def apply(self, value: float) -> float:
        """Return ``value + Laplace(0, scale)``."""
        return value + self.noise()


class GaussianMechanism:
    """``(ε, δ)``-DP via Gaussian noise.

    Parameters
    ----------
    epsilon :
        Privacy loss. Must be in ``(0, 1)`` for the closed-form
        ``σ`` calibration to hold.
    delta :
        Probability the privacy guarantee fails. Must be in
        ``(0, 1)``. Typical deployment values are much smaller
        than ``1 / n`` where ``n`` is the dataset size.
    sensitivity :
        L2 sensitivity of the query. Non-negative.
    seed :
        Optional RNG seed.
    """

    def __init__(
        self,
        *,
        epsilon: float,
        delta: float,
        sensitivity: float,
        seed: int | None = None,
    ) -> None:
        if not 0.0 < epsilon < 1.0:
            raise ValueError("epsilon must be in (0, 1)")
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if sensitivity < 0:
            raise ValueError("sensitivity must be non-negative")
        self._params = _MechanismParams(
            epsilon=epsilon, delta=delta, sensitivity=sensitivity
        )
        self._rng = random.Random(seed) if seed is not None else random.SystemRandom()

    @property
    def epsilon(self) -> float:
        return self._params.epsilon

    @property
    def delta(self) -> float:
        return self._params.delta

    @property
    def sensitivity(self) -> float:
        return self._params.sensitivity

    @property
    def sigma(self) -> float:
        """Standard deviation of the Gaussian noise."""
        if self._params.sensitivity == 0.0:
            return 0.0
        return (
            self._params.sensitivity
            * math.sqrt(2.0 * math.log(1.25 / self._params.delta))
            / self._params.epsilon
        )

    def noise(self) -> float:
        sigma = self.sigma
        if sigma == 0.0:
            return 0.0
        return self._rng.gauss(0.0, sigma)

    def apply(self, value: float) -> float:
        return value + self.noise()


def _signum(value: float) -> float:
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0
