# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Paired Sample Size

"""Paired non-inferiority sample size with exact normal quantiles.

``n = ceil((z_alpha + z_beta) ** 2 * d / m ** 2)`` where ``d`` is the
discordance rate and ``m`` the margin, both as proportions. The quantiles
come from :class:`statistics.NormalDist`; the rounded ``1.96 + 0.84`` and a
truncating ``int`` both give wrong sizes, so neither is used.
"""

from __future__ import annotations

import math
from statistics import NormalDist

from ._checks import require_finite
from .errors import SampleSizeError


def required_paired_sample_size(
    discordance_rate: float,
    margin: float,
    *,
    alpha: float = 0.025,
    power: float = 0.80,
) -> int:
    """Return the paired sample size for a one-sided non-inferiority test.

    Parameters
    ----------
    discordance_rate : float
        Expected share of discordant pairs, in ``(0, 1]``.
    margin : float
        Non-inferiority margin as a proportion, in ``(0, 1)``.
    alpha : float, optional
        One-sided significance level, in ``(0, 0.5)``. Default 0.025.
    power : float, optional
        Target power, in ``(0.5, 1)``. Default 0.80.

    Returns
    -------
    int
        The smallest whole number of pairs; the formula's ceiling.

    Raises
    ------
    SampleSizeError
        If an input is not a finite float in its range.

    Examples
    --------
    >>> required_paired_sample_size(0.05, 0.01)
    3925
    """
    d = require_finite(discordance_rate, "discordance_rate", SampleSizeError)
    m = require_finite(margin, "margin", SampleSizeError)
    a = require_finite(alpha, "alpha", SampleSizeError)
    p = require_finite(power, "power", SampleSizeError)
    if not 0.0 < d <= 1.0:
        raise SampleSizeError("discordance_rate must be in (0, 1]")
    if not 0.0 < m < 1.0:
        raise SampleSizeError("margin must be in (0, 1)")
    if not 0.0 < a < 0.5:
        raise SampleSizeError("alpha must be in (0, 0.5)")
    if not 0.5 < p < 1.0:
        raise SampleSizeError("power must be in (0.5, 1)")
    normal = NormalDist()
    z_sum = normal.inv_cdf(1.0 - a) + normal.inv_cdf(p)
    return math.ceil(z_sum**2 * d / m**2)
