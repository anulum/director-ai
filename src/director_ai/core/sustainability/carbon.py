# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CarbonIntensityTracker

"""Rolling window of grid carbon-intensity readings.

The tracker accepts a time series of ``(timestamp, gCO₂/kWh)``
tuples from the deployment's data source (electricity-map.org,
Cloud provider APIs, or a static sim) and exposes:

* ``current()`` — latest observation (or a caller-supplied
  fallback when the window is empty).
* ``percentile(value)`` — rank of ``value`` against the window.
  Lower percentile = lower-carbon periods. Used by the budget
  to decide throttling.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

try:
    from backfire_kernel import (
        rust_mean,
        rust_percentile_rank,
        rust_sum_f64,
        rust_sum_i64,
    )

    _RUST_CARBON = True
except Exception:  # pragma: no cover - optional dependency
    _RUST_CARBON = False

    def rust_percentile_rank(_values: list[float], _value: float) -> float:
        raise RuntimeError("backfire_kernel rust_percentile_rank is unavailable")

    def rust_mean(_values: list[float]) -> float:
        raise RuntimeError("backfire_kernel rust_mean is unavailable")

    def rust_sum_i64(_values: list[int]) -> int:
        raise RuntimeError("backfire_kernel rust_sum_i64 is unavailable")

    def rust_sum_f64(_values: list[float]) -> float:
        raise RuntimeError("backfire_kernel rust_sum_f64 is unavailable")


@dataclass(frozen=True)
class CarbonReading:
    """One ``(timestamp, intensity)`` pair."""

    timestamp: float
    intensity: float

    def __post_init__(self) -> None:
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        if self.intensity < 0:
            raise ValueError("intensity must be non-negative")


class CarbonIntensityTracker:
    """Rolling-window carbon intensity tracker.

    Parameters
    ----------
    window_size :
        Maximum readings retained. Default 672 — one week at
        15-minute resolution, which matches what most providers
        publish. FIFO eviction.
    clock :
        Timestamp source; injection point for tests that want
        deterministic percentile queries.
    fallback_intensity :
        Value returned by :meth:`current` when the tracker has
        no readings. Default 500 gCO₂/kWh — a cautious estimate
        that errs on the side of throttling.
    """

    def __init__(
        self,
        *,
        window_size: int = 672,
        clock: Callable[[], float] | None = None,
        fallback_intensity: float = 500.0,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if fallback_intensity < 0:
            raise ValueError("fallback_intensity must be non-negative")
        self._window = window_size
        self._clock = clock or time.time
        self._fallback = fallback_intensity
        self._lock = threading.Lock()
        self._readings: deque[CarbonReading] = deque(maxlen=window_size)

    def record(self, reading: CarbonReading) -> None:
        with self._lock:
            self._readings.append(reading)

    def record_many(self, readings: Iterable[CarbonReading]) -> None:
        with self._lock:
            for reading in readings:
                self._readings.append(reading)

    def current(self) -> float:
        with self._lock:
            if not self._readings:
                return self._fallback
            return self._readings[-1].intensity

    def window(self) -> tuple[CarbonReading, ...]:
        with self._lock:
            return tuple(self._readings)

    def percentile(self, value: float) -> float:
        """Return the fraction of window readings with
        intensity ``<= value``. Returns 1.0 when the window is
        empty (caller should treat as "high intensity"
        fallback)."""
        with self._lock:
            if not self._readings:
                return 1.0
            intensities = [r.intensity for r in self._readings]
            if _RUST_CARBON:
                try:
                    return float(rust_percentile_rank(intensities, value))
                except Exception:
                    pass
            below = _sum_int([1 if intensity <= value else 0 for intensity in intensities])
            return below / len(intensities)

    def mean(self) -> float:
        with self._lock:
            if not self._readings:
                return self._fallback
            intensities = [r.intensity for r in self._readings]
            if _RUST_CARBON:
                try:
                    return float(rust_mean(intensities))
                except Exception:
                    pass
            return _sum_float(intensities) / len(intensities)


def _sum_int(values: list[int]) -> int:
    if _RUST_CARBON:
        try:
            return int(rust_sum_i64(values))
        except Exception:
            pass
    return sum(values)


def _sum_float(values: list[float]) -> float:
    if _RUST_CARBON:
        try:
            return float(rust_sum_f64(values))
        except Exception:
            pass
    return sum(values)
