# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — streaming robust anomaly detector

"""A streaming, distribution-free outlier detector for one metric.

Real-time self-protection needs an anomaly score per observation without a
trained model or heavy dependency. :class:`StreamingRobustDetector` keeps a
bounded window of recent values and scores each new one with a robust z-score —
median absolute deviation, not mean/standard-deviation, so a few extreme values
(an attack already in progress) do not inflate the baseline and mask the next
one. It is the building block the RASP monitor instantiates per tracked metric.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median

__all__ = ["AnomalyScore", "StreamingRobustDetector"]

# Scale that makes the MAD a consistent estimator of the standard deviation for
# normally distributed data (1 / Φ⁻¹(3/4)).
_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class AnomalyScore:
    """The robust score for one observation against its rolling baseline."""

    value: float
    robust_z: float
    anomalous: bool
    cold_start: bool


class StreamingRobustDetector:
    """Score observations of one metric by robust (median/MAD) z-score."""

    def __init__(
        self,
        *,
        window_size: int = 128,
        min_samples: int = 12,
        z_threshold: float = 3.5,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        if not 1 <= min_samples <= window_size:
            raise ValueError("min_samples must be in [1, window_size]")
        if z_threshold <= 0:
            raise ValueError("z_threshold must be positive")
        self._window: deque[float] = deque(maxlen=window_size)
        self._min_samples = min_samples
        self._z_threshold = z_threshold

    @property
    def z_threshold(self) -> float:
        """The robust-z above which an observation is flagged anomalous."""
        return self._z_threshold

    def update(self, value: float) -> AnomalyScore:
        """Score ``value`` against the current window, then fold it in.

        While fewer than ``min_samples`` have been seen the detector is in
        cold start: it records the value and reports no anomaly (there is no
        baseline yet to judge against).
        """
        value = float(value)
        if len(self._window) < self._min_samples:
            self._window.append(value)
            return AnomalyScore(
                value=value, robust_z=0.0, anomalous=False, cold_start=True
            )

        centre = median(self._window)
        deviations = [abs(v - centre) for v in self._window]
        mad = median(deviations)
        if mad > 0.0:
            scale = _MAD_TO_SIGMA * mad
        else:
            # MAD is zero whenever over half the window equals the median, even
            # when the window has real spread (its weakness). Fall back to the
            # mean absolute deviation so an in-range value is not flagged inf.
            mean_ad = sum(deviations) / len(deviations)
            scale = _MAD_TO_SIGMA * mean_ad
        if scale > 0.0:
            robust_z = abs(value - centre) / scale
        else:
            # A truly constant baseline: any deviation is maximally surprising.
            robust_z = 0.0 if value == centre else float("inf")

        self._window.append(value)
        return AnomalyScore(
            value=value,
            robust_z=robust_z,
            anomalous=robust_z > self._z_threshold,
            cold_start=False,
        )
