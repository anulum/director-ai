# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Quantum Acceptance Arithmetic

r"""The lower-bound conjunct of a quantum arm's admission, and its improvement.

The shot-noise arithmetic (Q-1, Q-2, the paired radii and standard error)
is in :mod:`.quantum_shot_noise`. For K < 30 sealed clusters the lower bound
is the composite :math:`\bar\Delta - r_{\mathrm{boot}} - \varepsilon_{\mathrm{paired}}`
(Q-3). The kernel computes neither the bootstrap lower bound nor the cluster
radius, and it does not see K: which path applies, and the bounds
themselves at their shares of the error rate, come from the evaluator port
of sub-unit 8, so the guarantee is conditional on that port. On the K < 30
path it is also conditional on the readout calibration: any readout bias is
assumed to lie within ``min_physical_margin``, and the evaluation review's
admission states the calibration evidence it relied on.

:func:`admits_quantum_arm` checks one conjunct of the admission, that the
lower bound reaches the physical margin and the mode's floor: three solver
tolerances on an exact statevector (Q-4), or at least three shot standard
errors otherwise. A ``True`` is not an admission. The receipts (Q-6), the
seeds (Q-7), the pins (Q-9) and the barren-plateau witness are composed at
promotion; the classical baseline and the absent optimum claim (Q-5) are
in the spec. The margin and the multiplier are declared trusts. Only the
final comparison is exact: its inputs carry their own floating-point
rounding, though the composite itself is computed exactly and rounded down.

The improvement is ``baseline - candidate`` when minimising energy and its
negation when maximising fidelity.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ._checks import require_finite
from .errors import QuantumAcceptanceError
from .quantum_contracts import validate_shot_noise_spec
from .quantum_records import QuantumShotNoiseAcceptanceSpec
from .quantum_shot_noise import finite_number

E = QuantumAcceptanceError
EXACT_TOLERANCE_MULTIPLE = 3
"""Q-4: an exact-statevector arm's bound must span three solver tolerances."""


def composite_lower_bound(
    mean_improvement: float, cluster_bootstrap_radius: float, paired_radius: float
) -> float:
    """Return the K < 30 composite lower bound of the mean improvement (Q-3).

    The bound is computed exactly and rounded down to a float, so it never
    exceeds the exact composite of its inputs.

    Parameters
    ----------
    mean_improvement : float
        The mean paired improvement.
    cluster_bootstrap_radius : float
        The evaluator's cluster bootstrap radius at ``alpha_cluster``. The
        kernel does not compute it, so the bound is only as sound as the
        evaluator port that supplies it, and it holds conditional on the
        readout calibration.
    paired_radius : float
        The paired shot radius at ``alpha_shot``.

    Returns
    -------
    float
        The lower bound, finite.

    Raises
    ------
    QuantumAcceptanceError
        If a value is not finite, a radius is negative, or the bound
        overflows or rounds down to a non-finite value.
    """
    mean = require_finite(mean_improvement, "mean_improvement", E)
    radii = [
        require_finite(cluster_bootstrap_radius, "cluster_bootstrap_radius", E),
        require_finite(paired_radius, "paired_radius", E),
    ]
    if min(radii) < 0.0:
        raise E("a radius must not be negative")
    exact = Fraction(mean) - Fraction(radii[0]) - Fraction(radii[1])
    try:
        bound = float(exact)
    except OverflowError as error:
        raise E("the composite lower bound overflows") from error
    lower = math.nextafter(bound, -math.inf) if Fraction(bound) > exact else bound
    return finite_number(lower, "the composite lower bound")


def admits_quantum_arm(
    lower_bound: float,
    spec: QuantumShotNoiseAcceptanceSpec,
    *,
    shot_standard_error: float | None,
) -> bool:
    """Return whether an arm's lower bound clears the lower-bound conjunct.

    This is one conjunct of the admission, not the admission: Q-6, Q-7, Q-9
    and the barren-plateau witness are composed at promotion.

    Parameters
    ----------
    lower_bound : float
        The improvement's lower confidence bound at ``alpha_total``, from the
        evaluator's bootstrap or the composite bound; which applies, by K,
        is the evaluator port's choice.
    spec : QuantumShotNoiseAcceptanceSpec
        The study's acceptance spec; its margin and multiplier are declared
        trusts.
    shot_standard_error : float or None
        The shot standard error of the mean improvement; ``None`` exactly
        for an exact statevector.

    Returns
    -------
    bool
        Whether the bound reaches the physical margin and the mode's floor,
        compared exactly.

    Raises
    ------
    QuantumAcceptanceError
        If the spec is refused, or the standard error is given against the
        mode or is negative.
    """
    validate_shot_noise_spec(spec)
    bound = require_finite(lower_bound, "lower_bound", E)
    if spec.execution_mode == "exact_statevector":
        if shot_standard_error is not None:
            raise E("an exact statevector has no shot standard error")
        tolerance = spec.exact_solver_tolerance
        floor = EXACT_TOLERANCE_MULTIPLE * Fraction(
            require_finite(tolerance, "tolerance", E)
        )
    else:
        error = require_finite(shot_standard_error, "shot_standard_error", E)
        if error < 0.0:
            raise E("a shot standard error must not be negative")
        floor = Fraction(spec.shot_noise_floor_multiplier) * Fraction(error)
    return Fraction(bound) >= max(Fraction(spec.min_physical_margin), floor)


def paired_improvement(
    baseline_estimate: float,
    candidate_estimate: float,
    spec: QuantumShotNoiseAcceptanceSpec,
) -> float:
    """Return one instance's improvement, oriented so that larger is better.

    Parameters
    ----------
    baseline_estimate : float
        The baseline arm's estimate.
    candidate_estimate : float
        The candidate arm's estimate.
    spec : QuantumShotNoiseAcceptanceSpec
        The study's acceptance spec, which names the optimisation sense.

    Returns
    -------
    float
        ``baseline - candidate`` when minimising energy, its negation when
        maximising fidelity; finite.

    Raises
    ------
    QuantumAcceptanceError
        If an estimate is not finite, the difference overflows, or the spec
        is refused.
    """
    validate_shot_noise_spec(spec)
    delta = require_finite(baseline_estimate, "baseline_estimate", E) - require_finite(
        candidate_estimate, "candidate_estimate", E
    )
    if not math.isfinite(delta):
        raise E("the paired improvement overflows")
    return delta if spec.optimisation_sense == "minimise_energy" else -delta
