# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ConformalDemandForecaster

"""Inductive conformal prediction over a per-day demand series.

The forecaster keeps a recent window of daily demand
observations and the residuals produced by the point predictor
(EMA by default). For a new day's forecast it returns a
prediction interval at the caller's coverage target, built from
the ``(1 − α)(n + 1) / n`` order statistic of the residuals.

Under exchangeability the interval's coverage matches the target
in expectation. The forecaster does not attempt online
conformal recalibration — it refreshes the residual window every
time :meth:`observe` is called.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionInterval:
    """Point prediction plus conformal bounds."""

    point: float
    lower: float
    upper: float
    coverage: float
    residual_sample_size: int

    @property
    def width(self) -> float:
        return self.upper - self.lower


class ConformalDemandForecaster:
    """EMA point predictor + conformal interval.

    Parameters
    ----------
    alpha :
        EMA smoothing. Larger ``alpha`` weights recent days more;
        default 0.4.
    residual_window :
        Maximum retained residuals for the conformal quantile.
        Default 90.
    min_samples :
        Minimum residuals required before the conformal interval
        is trusted. Below this the interval falls back to
        ``[point, point]``. Default 10.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.4,
        residual_window: int = 90,
        min_samples: int = 10,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if residual_window <= 0:
            raise ValueError("residual_window must be positive")
        if min_samples <= 0:
            raise ValueError("min_samples must be positive")
        self._alpha = alpha
        self._residuals: deque[float] = deque(maxlen=residual_window)
        self._min_samples = min_samples
        self._ema: float | None = None
        self._lock = threading.Lock()

    @property
    def last_forecast(self) -> float | None:
        with self._lock:
            return self._ema

    def observe(self, demand: float) -> None:
        """Record one day's observed demand. The residual is
        measured against the current EMA before the EMA is
        updated so the stored residuals are honest
        out-of-sample differences."""
        if demand < 0:
            raise ValueError("demand must be non-negative")
        with self._lock:
            if self._ema is not None:
                self._residuals.append(abs(demand - self._ema))
            self._update_locked(demand)

    def predict(self, *, coverage: float = 0.9) -> PredictionInterval:
        """Return the next-day prediction interval.

        Raises :class:`ValueError` when no observations have been
        recorded. When fewer than ``min_samples`` residuals are
        available the interval collapses to ``[point, point]`` so
        the caller sees "no uncertainty estimate yet"."""
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must be in (0, 1)")
        with self._lock:
            if self._ema is None:
                raise ValueError("no observations yet")
            residuals = list(self._residuals)
            point = self._ema
        if len(residuals) < self._min_samples:
            return PredictionInterval(
                point=point,
                lower=point,
                upper=point,
                coverage=coverage,
                residual_sample_size=len(residuals),
            )
        sorted_residuals = sorted(residuals)
        n = len(sorted_residuals)
        q_index = min(max(int(coverage * (n + 1) - 1), 0), n - 1)
        width = sorted_residuals[q_index]
        return PredictionInterval(
            point=point,
            lower=max(0.0, point - width),
            upper=point + width,
            coverage=coverage,
            residual_sample_size=n,
        )

    def reset(self) -> None:
        with self._lock:
            self._residuals.clear()
            self._ema = None

    def _update_locked(self, demand: float) -> None:
        if self._ema is None:
            self._ema = demand
        else:
            self._ema = self._alpha * demand + (1.0 - self._alpha) * self._ema
