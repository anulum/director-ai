# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — IrreversibilityForecaster

"""Monte-Carlo forecast of point-of-no-return probability.

Given a :class:`~director_ai.core.irreversibility.reversibility.ReversibilityEstimator`
and a sequence of candidate actions, draw ``n_samples`` Bernoulli
draws per action (success = reversible) and count the draws that
cross the irreversibility threshold. Return both the point
estimate and a Wilson-score credible interval so the caller can
size the interval-based halt band.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from .reversibility import ReversibilityEstimator, RuleReversibility


@dataclass(frozen=True)
class Forecast:
    """Result of one forecast.

    ``p_irreversible`` — point estimate of the probability the
    full action sequence crosses the irreversibility threshold.
    ``ci_low`` / ``ci_high`` — Wilson-score bounds for a
    caller-configured confidence level (default 0.95). ``crossed``
    counts the sampled draws that crossed; ``samples`` is the
    total sample count.
    """

    p_irreversible: float
    ci_low: float
    ci_high: float
    crossed: int
    samples: int


class IrreversibilityForecaster:
    """Seeded Monte-Carlo forecaster.

    Parameters
    ----------
    estimator :
        Any :class:`ReversibilityEstimator`. Defaults to a fresh
        :class:`RuleReversibility`.
    threshold :
        Probability under which a draw is flagged as crossing into
        the irreversible region. Default 0.2 — "below a 20 % chance
        of reversal, treat the path as irreversible".
    n_samples :
        Monte-Carlo budget per forecast. Default 512 — enough for
        tight Wilson bounds at 95 % on most practical p values.
    confidence :
        Confidence level for the Wilson-score interval. Default 0.95.
    """

    def __init__(
        self,
        *,
        estimator: ReversibilityEstimator | None = None,
        threshold: float = 0.2,
        n_samples: int = 512,
        confidence: float = 0.95,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1); got {threshold!r}")
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive; got {n_samples!r}")
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"confidence must be in (0, 1); got {confidence!r}")
        self._estimator: ReversibilityEstimator = estimator or RuleReversibility()
        self._threshold = threshold
        self._n_samples = n_samples
        self._confidence = confidence

    def forecast(
        self,
        actions: Sequence[str],
        *,
        seed: int = 0,
    ) -> Forecast:
        """Run Monte-Carlo over ``actions``.

        Independent-action sampling: each action is scored once (the
        estimator is deterministic on the stub implementation), and
        the cumulative reversibility of the sequence is the product
        of per-action reversibilities. Each Monte-Carlo draw then
        compares a uniform sample against that cumulative product
        and records whether it crosses the threshold.
        """
        if not actions:
            raise ValueError("actions must be non-empty")
        per_action = [self._estimator.score(a).score for a in actions]
        cumulative_reversible = 1.0
        for s in per_action:
            cumulative_reversible *= s
        # Deterministic seeded RNG so regression tests are stable.
        rng = random.Random(seed)
        crossed = 0
        for _ in range(self._n_samples):
            draw = rng.random()
            if draw > cumulative_reversible and (1.0 - cumulative_reversible) >= (
                1.0 - self._threshold
            ):
                crossed += 1
        p_hat = crossed / self._n_samples
        low, high = _wilson_score(p_hat, self._n_samples, self._confidence)
        return Forecast(
            p_irreversible=p_hat,
            ci_low=low,
            ci_high=high,
            crossed=crossed,
            samples=self._n_samples,
        )


def _wilson_score(p_hat: float, n: int, confidence: float) -> tuple[float, float]:
    """Wilson score interval — well-behaved at p near 0 / 1 where
    the normal approximation collapses.

    ``z`` is the standard normal quantile for (1 + confidence) / 2.
    """
    if n <= 0:
        return (0.0, 0.0)
    z = _standard_normal_quantile((1.0 + confidence) / 2.0)
    denominator = 1.0 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denominator
    halfwidth = (
        z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4 * n * n))
    ) / denominator
    return (max(0.0, centre - halfwidth), min(1.0, centre + halfwidth))


def _standard_normal_quantile(p: float) -> float:
    """Inverse of the standard-normal CDF using the Beasley-Springer /
    Moro rational approximation. Accurate to about 1e-4 across
    ``p`` in ``[0.001, 0.999]`` — enough for Wilson-interval math.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1); got {p!r}")
    # Beasley-Springer coefficients.
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
