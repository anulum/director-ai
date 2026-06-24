# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Adaptive conformal inference for a drifting score stream.

Split conformal assumes the calibration distribution matches live traffic. When
the query stream drifts — the fleet asks different things week to week — a static
quantile keeps returning the same half-width while the realised coverage silently
slips away from target. That is the covariate-shift failure mode the abstention
design exists to defend against.

Adaptive Conformal Inference (Gibbs & Candès, NeurIPS 2021) closes the loop:
instead of a fixed miscoverage level it keeps a time-varying ``alpha_t`` and, after
each prediction whose outcome becomes known, nudges it toward the target,

    alpha_{t+1} = clip(alpha_t + gamma * (alpha_target - err_t), 0, 1)

where ``err_t`` is 1 when the interval missed the truth and 0 when it covered. The
interval at step ``t`` is taken at coverage ``1 - alpha_t``. The long-run realised
miscoverage then tracks ``alpha_target`` under *arbitrary* distribution shift — no
exchangeability assumption — which is exactly what a drifting recall stream needs.

It pairs with :class:`~director_ai.core.calibration.miscoverage.MiscoverageMonitor`:
the monitor watches realised coverage and fails closed when it cannot be held;
this predictor moves the threshold to hold it. Both consume the same
``covered``/miss signal.
"""

from __future__ import annotations

from .conformal import ConformalPredictor, PredictionInterval

__all__ = ["AdaptiveConformalPredictor"]


def _clip01(value: float) -> float:
    """Clamp to the closed unit interval."""
    return max(0.0, min(1.0, value))


class AdaptiveConformalPredictor(ConformalPredictor):
    """Gibbs–Candès adaptive split-conformal predictor over a drifting stream.

    Calibrate exactly as the static :class:`ConformalPredictor` (the residuals
    are shared), then call :meth:`update` with each realised outcome so the
    effective coverage self-corrects. :meth:`predict` evaluates the half-width at
    the current adapted level rather than the fixed target.
    """

    def __init__(
        self,
        coverage: float = 0.95,
        *,
        gamma: float = 0.05,
        min_samples: int = 30,
    ) -> None:
        super().__init__(coverage, min_samples)
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        self._gamma = gamma
        self._alpha_target = 1.0 - coverage
        self._alpha_t = self._alpha_target

    @property
    def gamma(self) -> float:
        """The ACI step size."""
        return self._gamma

    @property
    def current_alpha(self) -> float:
        """The current adapted miscoverage level ``alpha_t``."""
        return self._alpha_t

    @property
    def effective_coverage(self) -> float:
        """The coverage level the next interval is built at (``1 - alpha_t``)."""
        return _clip01(1.0 - self._alpha_t)

    def update(self, *, covered: bool) -> None:
        """Adapt ``alpha_t`` from one realised outcome.

        ``covered=True`` means the emitted interval held the true outcome (a miss
        is ``covered=False``). Covering repeatedly widens ``alpha_t`` (narrower
        intervals — we were over-conservative); missing shrinks it (wider
        intervals — we need more coverage).
        """
        err = 0.0 if covered else 1.0
        self._alpha_t = _clip01(
            self._alpha_t + self._gamma * (self._alpha_target - err)
        )

    def reset_adaptation(self) -> None:
        """Return ``alpha_t`` to the target level, keeping the calibration set."""
        self._alpha_t = self._alpha_target

    def predict(self, score: float) -> PredictionInterval:
        """Interval at the current adapted coverage; reports it in ``coverage``."""
        n = len(self._scores)
        point_est = self._score_to_prob(score)
        eff = self.effective_coverage

        if n == 0:
            return PredictionInterval(
                point_estimate=point_est,
                lower=0.0,
                upper=1.0,
                coverage=eff,
                calibration_size=0,
                is_reliable=False,
            )

        half_width = 0.0 if eff <= 0.0 else (self._quantile_at(eff) or 0.0)

        return PredictionInterval(
            point_estimate=point_est,
            lower=max(0.0, point_est - half_width),
            upper=min(1.0, point_est + half_width),
            coverage=eff,
            calibration_size=n,
            is_reliable=n >= self._min_samples,
        )
