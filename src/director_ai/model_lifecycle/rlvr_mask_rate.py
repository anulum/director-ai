# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Verifier Masking Rate

"""The per-batch masking ceiling and the drift test of verifier faults.

A batch is halted when it masks more rollouts than its spec's ceiling, in
exact rational arithmetic, or when the lower confidence bound of its excess
masking over the reference rate, from two Wilson score intervals, exceeds
the allowed drift.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from decimal import Decimal
from fractions import Fraction
from statistics import NormalDist

from ._checks import require_int
from .errors import (
    RlvrContractError,
    VerifierMaskRateDriftError,
    VerifierMaskRateExceededError,
)
from .rlvr_contracts import (
    FAULT,
    validate_rlvr_spec,
    validate_verifiable_reward_outcome,
)
from .rlvr_records import RlvrAlgorithmSpec, VerifiableRewardOutcome

E = RlvrContractError


def wilson_interval(count: int, total: int, z: float) -> tuple[float, float]:
    """Return the two-sided Wilson score interval of ``count / total``.

    Parameters
    ----------
    count, total : int
        The events and the trials, with ``0 <= count <= total`` and
        ``total > 0``.
    z : float
        The standard normal quantile of the interval.

    Returns
    -------
    tuple[float, float]
        The lower and upper bounds.
    """
    p = count / total
    scale = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / scale
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4 * total * total)) / scale
    return max(0.0, centre - half), min(1.0, centre + half)


def validate_rlvr_batch_mask_rate(
    outcomes: Sequence[VerifiableRewardOutcome],
    *,
    spec: RlvrAlgorithmSpec,
    reference_masked_count: int,
    reference_total_count: int,
) -> float:
    """Halt a batch that masks too many rollouts or drifts above the reference (TR-2).

    Parameters
    ----------
    outcomes : Sequence[VerifiableRewardOutcome]
        The batch's outcomes.
    spec : RlvrAlgorithmSpec
        The algorithm spec with the ceiling, the drift and its alpha.
    reference_masked_count, reference_total_count : int
        The reference policy's masked and total rollouts.

    Returns
    -------
    float
        The lower confidence bound of the batch's excess masking.

    Raises
    ------
    VerifierMaskRateExceededError
        If the masked fraction exceeds the per-batch ceiling.
    VerifierMaskRateDriftError
        If the lower bound of the excess over the reference exceeds the
        allowed drift.
    RlvrContractError
        If the batch is empty, an outcome or the spec is malformed, or the
        reference counts are not ``0 <= masked <= total`` with ``total > 0``.
    """
    validate_rlvr_spec(spec)
    if not outcomes:
        raise E("a batch needs at least one outcome")
    for outcome in outcomes:
        validate_verifiable_reward_outcome(outcome, spec=spec)
    total = len(outcomes)
    masked = sum(outcome.status == FAULT for outcome in outcomes)
    ceiling = Fraction(Decimal(str(spec.max_masked_rollout_fraction_per_batch)))
    if Fraction(masked, total) > ceiling:
        raise VerifierMaskRateExceededError(
            f"{masked} of {total} rollouts are masked, above the ceiling"
        )
    reference_total = require_int(
        reference_total_count, "reference_total_count", E, minimum=1
    )
    reference_masked = require_int(
        reference_masked_count, "reference_masked_count", E, minimum=0
    )
    if reference_masked > reference_total:
        raise E("reference_masked_count must not exceed reference_total_count")
    z = NormalDist().inv_cdf(1.0 - spec.masked_drift_alpha / 2.0)
    lower, _ = wilson_interval(masked, total, z)
    _, reference_upper = wilson_interval(reference_masked, reference_total, z)
    batch_rate, reference_rate = masked / total, reference_masked / reference_total
    bound = (batch_rate - reference_rate) - math.sqrt(
        (batch_rate - lower) ** 2 + (reference_upper - reference_rate) ** 2
    )
    if bound > spec.max_masked_fraction_drift_vs_reference:
        raise VerifierMaskRateDriftError(
            f"excess masking of at least {bound:.4f} exceeds the allowed drift"
        )
    return bound
