# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Quantum Shot-Noise Arithmetic

r"""Shot-noise variance and the Hoeffding radius of one instance (Q-1, Q-2).

Each Pauli group contributes one bounded outcome per shot. Its
``coefficient`` is the group's half-range: :math:`\sum_p |c_p|` over the
jointly measured terms, never one term of a jointly measured group, since
separate terms would drop their covariance and understate the radius.
``mean`` is the group's normalised mean in :math:`[-1, 1]`, and ``gamma``
bounds the per-shot range of the mitigated estimator, the L1 norm of the
mitigation map, not merely a variance inflation. For one instance with
shots :math:`S_g \ge 256` and :math:`\gamma \ge 1`:

- Q-1, the shot variance:
  :math:`\gamma^2 \sum_g c_g^2 (1 - \hat\mu_g^2) / (S_g - 1)`, unbiased for a
  single ±1 term and an upper bound for a normalised observable;
- Q-2, the joint Hoeffding radius at level :math:`\alpha`:
  :math:`\gamma \sqrt{2 \ln(2/\alpha) \sum_g c_g^2 / S_g}`.

The paired forms over all instances are in :mod:`.quantum_paired_noise`.
Every function returns a finite float or refuses by name: an overflow, an
undefined product or an underflowing level is never returned. A zero
coefficient gives a zero radius.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ._checks import require_finite, require_int
from .errors import QuantumAcceptanceError
from .quantum_contracts import MIN_SHOTS_PER_PAULI_GROUP

E = QuantumAcceptanceError
MAX_SHOTS_PER_PAULI_GROUP = 2**53
"""Shot counts above this are not exact as floats, so they are refused."""


@dataclass(frozen=True, slots=True)
class PauliGroupShots:
    """One jointly measured Pauli group: its half-range, normalised mean and shots."""

    coefficient: float
    mean: float
    shots: int


def finite_number(value: float, what: str) -> float:
    """Return ``value`` if it is a finite number, else refuse by name."""
    if not math.isfinite(value):
        raise E(f"{what} is not a finite number")
    return value


def sum_finite(terms: Iterable[float], what: str) -> float:
    """Return the exact-rounded sum of ``terms``, refusing an overflow by name."""
    try:
        return finite_number(math.fsum(terms), what)
    except OverflowError as error:
        raise E(f"{what} overflows") from error


def log_two_over(level: float, what: str) -> float:
    """Return ``ln(2 / level)``, refusing a level so small that it overflows."""
    return finite_number(math.log(finite_number(2.0 / level, what)), what)


def gamma_factor(value: float) -> float:
    """Return a range factor of at least one."""
    gamma = require_finite(value, "gamma", E)
    if gamma < 1.0:
        raise E("a mitigation overhead gamma must be at least 1")
    return gamma


def error_rate(value: float) -> float:
    """Return an error rate in ``(0, 1)``."""
    alpha = require_finite(value, "alpha", E)
    if not 0.0 < alpha < 1.0:
        raise E("an error rate must lie in (0, 1)")
    return alpha


def pauli_groups(groups: Sequence[PauliGroupShots]) -> tuple[PauliGroupShots, ...]:
    """Return non-empty Pauli groups with finite values and a valid shot count."""
    checked = tuple(groups)
    if not checked:
        raise E("an instance has at least one Pauli group")
    for group in checked:
        if not isinstance(group, PauliGroupShots):
            raise E("Pauli groups must be PauliGroupShots records")
        require_finite(group.coefficient, "coefficient", E)
        if not -1.0 <= require_finite(group.mean, "mean", E) <= 1.0:
            raise E("a Pauli group mean must lie in [-1, 1]")
        shots = require_int(group.shots, "shots", E, minimum=MIN_SHOTS_PER_PAULI_GROUP)
        if shots > MAX_SHOTS_PER_PAULI_GROUP:
            raise E("shots must not exceed 2**53")
    return checked


def shot_weight(groups: Sequence[PauliGroupShots], what: str) -> float:
    """Return the sum of squared half-ranges over shots."""
    return sum_finite((g.coefficient * g.coefficient / g.shots for g in groups), what)


def shot_variance(gamma: float, groups: Sequence[PauliGroupShots]) -> float:
    """Return one instance's shot and mitigation variance (Q-1).

    Parameters
    ----------
    gamma : float
        The instance's range factor, at least 1.
    groups : Sequence[PauliGroupShots]
        The instance's jointly measured groups.

    Returns
    -------
    float
        The variance estimate, finite.

    Raises
    ------
    QuantumAcceptanceError
        If a value is outside its domain or the variance is not finite.
    """
    overhead, checked = gamma_factor(gamma), pauli_groups(groups)
    what = "the shot variance"
    total = sum_finite(
        (
            g.coefficient * g.coefficient * (1.0 - g.mean * g.mean) / (g.shots - 1)
            for g in checked
        ),
        what,
    )
    return finite_number(overhead * overhead * total, what)


def hoeffding_radius(
    gamma: float, groups: Sequence[PauliGroupShots], alpha: float
) -> float:
    """Return one instance's joint Hoeffding radius at level ``alpha`` (Q-2).

    Parameters
    ----------
    gamma : float
        The instance's range factor, at least 1.
    groups : Sequence[PauliGroupShots]
        The instance's jointly measured groups.
    alpha : float
        The error rate, in ``(0, 1)``.

    Returns
    -------
    float
        The radius, finite; zero when every coefficient is zero.

    Raises
    ------
    QuantumAcceptanceError
        If a value is outside its domain or the radius is not finite.
    """
    overhead, checked, level = (
        gamma_factor(gamma),
        pauli_groups(groups),
        error_rate(alpha),
    )
    what = "the Hoeffding radius"
    weight = shot_weight(checked, what)
    if weight == 0.0:
        return 0.0
    return finite_number(
        overhead * math.sqrt(2.0 * log_two_over(level, what) * weight), what
    )
