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
            below = sum(1 for r in self._readings if r.intensity <= value)
            return below / len(self._readings)

    def mean(self) -> float:
        with self._lock:
            if not self._readings:
                return self._fallback
            return sum(r.intensity for r in self._readings) / len(
                self._readings
            )
