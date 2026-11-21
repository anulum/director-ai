# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Quantum Paired Shot-Noise Arithmetic

r"""The paired shot radius and standard error over all instances of a look.

Over :math:`N = RK` paired instances (R seeds, K sealed clusters), with
:math:`w_n = \sum_g c_g^2 / S_g` for instance :math:`n` and the spec's shot
share :math:`\alpha` of the error rate:

- with independent readout calibration:
  :math:`\frac{1}{N}\sqrt{2 \ln(2/\alpha) \sum_n (\gamma_B^2 + \gamma_C^2) w_n}`;
- with one shared calibration, the union bound:
  :math:`\frac{1}{N}\sum_n (\varepsilon_B(\alpha/2N) + \varepsilon_C(\alpha/2N))`;
- the shot standard error of the mean improvement:
  :math:`\frac{1}{N}\sqrt{\sum_n (\mathrm{Var}_B + \mathrm{Var}_C)}`.

Both arms of an instance share its coefficients and shots. Whether the
calibration is shared is the caller's statement, a trust that the evaluator
port carries. Every function returns a finite float or refuses by name.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ._checks import require_finite
from .errors import QuantumAcceptanceError
from .quantum_contracts import validate_shot_noise_spec
from .quantum_records import QuantumShotNoiseAcceptanceSpec
from .quantum_shot_noise import (
    PauliGroupShots,
    error_rate,
    finite_number,
    gamma_factor,
    hoeffding_radius,
    log_two_over,
    pauli_groups,
    shot_variance,
    shot_weight,
    sum_finite,
)

E = QuantumAcceptanceError


@dataclass(frozen=True, slots=True)
class PairedShotInstance:
    """One paired instance: both arms' range factors and Pauli groups."""

    baseline_gamma: float
    candidate_gamma: float
    baseline_groups: tuple[PauliGroupShots, ...]
    candidate_groups: tuple[PauliGroupShots, ...]


def _instances(
    instances: Sequence[PairedShotInstance],
) -> tuple[PairedShotInstance, ...]:
    """Return a non-empty sequence of paired instances whose arms share shots."""
    checked = tuple(instances)
    if not checked:
        raise E("a paired estimate needs at least one instance")
    for item in checked:
        if not isinstance(item, PairedShotInstance):
            raise E("instances must be PairedShotInstance records")
        baseline, candidate = (
            pauli_groups(item.baseline_groups),
            pauli_groups(item.candidate_groups),
        )
        if [(g.coefficient, g.shots) for g in baseline] != [
            (g.coefficient, g.shots) for g in candidate
        ]:
            raise E("both arms of an instance share its coefficients and shots")
        gamma_factor(item.baseline_gamma)
        gamma_factor(item.candidate_gamma)
    return checked


def paired_shot_radius(
    instances: Sequence[PairedShotInstance],
    spec: QuantumShotNoiseAcceptanceSpec,
    *,
    shared_calibration: bool,
) -> float:
    """Return the paired shot radius of the mean improvement at ``spec.alpha_shot``.

    Parameters
    ----------
    instances : Sequence[PairedShotInstance]
        The R x K paired instances.
    spec : QuantumShotNoiseAcceptanceSpec
        The study's sampled-mode acceptance spec, whose shot share sets the
        level.
    shared_calibration : bool
        Whether one readout calibration is shared across a batch, in which
        case the conservative union bound is used. This is the caller's
        statement, a trust the evaluator port carries.

    Returns
    -------
    float
        The radius, finite; zero when every coefficient is zero.

    Raises
    ------
    QuantumAcceptanceError
        If the spec is an exact statevector or refused, an instance is
        outside its domain, the per-instance level underflows, or the radius
        is not finite.
    """
    validate_shot_noise_spec(spec)
    if spec.execution_mode == "exact_statevector":
        raise E("an exact statevector has no shot radius")
    checked, level = (
        _instances(instances),
        error_rate(require_finite(spec.alpha_shot, "alpha_shot", E)),
    )
    count, what = len(checked), "the paired shot radius"
    if shared_calibration:
        per_instance = level / (2 * count)
        if per_instance <= 0.0:
            raise E("the per-instance shot level underflows to zero")
        total = sum_finite(
            (
                hoeffding_radius(i.baseline_gamma, i.baseline_groups, per_instance)
                + hoeffding_radius(i.candidate_gamma, i.candidate_groups, per_instance)
                for i in checked
            ),
            what,
        )
        return finite_number(total / count, what)
    total = sum_finite(
        (
            (
                i.baseline_gamma * i.baseline_gamma
                + i.candidate_gamma * i.candidate_gamma
            )
            * shot_weight(i.baseline_groups, what)
            for i in checked
        ),
        what,
    )
    if total == 0.0:
        return 0.0
    return finite_number(
        math.sqrt(2.0 * log_two_over(level, what) * total) / count, what
    )


def paired_shot_standard_error(instances: Sequence[PairedShotInstance]) -> float:
    """Return the shot standard error of the mean paired improvement.

    Parameters
    ----------
    instances : Sequence[PairedShotInstance]
        The R x K paired instances.

    Returns
    -------
    float
        The standard error, finite.

    Raises
    ------
    QuantumAcceptanceError
        If an instance is outside its domain, a nonzero term underflows, or
        the error is not finite.
    """
    checked, what = _instances(instances), "the shot standard error"
    total = sum_finite(
        (
            shot_variance(i.baseline_gamma, i.baseline_groups)
            + shot_variance(i.candidate_gamma, i.candidate_groups)
            for i in checked
        ),
        what,
    )
    return finite_number(math.sqrt(total) / len(checked), what)
