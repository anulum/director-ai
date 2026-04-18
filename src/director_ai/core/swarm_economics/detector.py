# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — TragedyDetector

"""Detect tragedy-of-the-commons over-consumption.

The detector compares the observed aggregate draw rate against
the pool's sustainable rate (``regeneration_rate``). If the
observed draw exceeds the sustainable rate for longer than
``grace_seconds`` the detector reports a :class:`TragedySignal`.

``grace_factor`` lets a swarm burn above sustainable rate briefly
(bursty workloads need headroom); the detector only fires when
the smoothed average draw exceeds
``grace_factor * sustainable_rate`` for the grace window.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from .pool import ResourcePool


@dataclass(frozen=True)
class TragedySignal:
    """Result of one :meth:`TragedyDetector.check` call."""

    firing: bool
    observed_rate: float
    sustainable_rate: float
    pressure: float
    grace_elapsed: float


class TragedyDetector:
    """Streaming detector over a :class:`ResourcePool` ledger.

    Parameters
    ----------
    pool :
        The resource pool to monitor.
    window_seconds :
        Length of the observation window for the smoothed draw
        rate. Default 60 s.
    grace_seconds :
        How long the smoothed rate may exceed the sustainable
        rate before the detector fires. Default 30 s.
    grace_factor :
        Multiplier on the sustainable rate that counts as
        "over budget". ``1.0`` means any sustained draw above
        ``regeneration_rate`` fires the alarm. Default 1.25.
    clock :
        Timestamp source.
    """

    def __init__(
        self,
        *,
        pool: ResourcePool,
        window_seconds: float = 60.0,
        grace_seconds: float = 30.0,
        grace_factor: float = 1.25,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if grace_seconds < 0:
            raise ValueError("grace_seconds must be non-negative")
        if grace_factor < 1.0:
            raise ValueError("grace_factor must be >= 1.0")
        self._pool = pool
        self._window = window_seconds
        self._grace = grace_seconds
        self._grace_factor = grace_factor
        self._clock = clock or time.time
        self._lock = threading.Lock()
        self._over_since: float | None = None

    def check(self) -> TragedySignal:
        sustainable = self._pool.regeneration_rate
        recent = self._pool.recent(since_seconds=self._window)
        drawn = sum(record.amount for record in recent)
        observed_rate = drawn / self._window
        now = float(self._clock())
        pressure = _pressure(observed_rate, sustainable, self._grace_factor)
        threshold = sustainable * self._grace_factor
        with self._lock:
            firing = False
            grace_elapsed = 0.0
            if sustainable > 0 and observed_rate > threshold:
                if self._over_since is None:
                    self._over_since = now
                grace_elapsed = now - self._over_since
                firing = grace_elapsed >= self._grace
            else:
                self._over_since = None
        return TragedySignal(
            firing=firing,
            observed_rate=observed_rate,
            sustainable_rate=sustainable,
            pressure=pressure,
            grace_elapsed=grace_elapsed,
        )

    def reset(self) -> None:
        with self._lock:
            self._over_since = None


def _pressure(
    observed_rate: float, sustainable_rate: float, grace_factor: float
) -> float:
    """Map the draw-to-sustainable ratio into ``[0, 1]``.

    Sustainable or below returns 0; at the grace-factor
    threshold returns 0.5; saturates at 1.0 when the observed
    rate reaches twice the threshold.
    """
    if sustainable_rate <= 0:
        # Non-regenerating pool — any draw is "over budget" in
        # the long run; caller calibrates pressure via a
        # separate exhaustion-headroom signal.
        return 1.0 if observed_rate > 0 else 0.0
    threshold = sustainable_rate * grace_factor
    if observed_rate <= sustainable_rate:
        return 0.0
    if observed_rate <= threshold:
        # Linear ramp between sustainable and threshold.
        return 0.5 * (observed_rate - sustainable_rate) / (threshold - sustainable_rate)
    # Above threshold: climb from 0.5 toward 1.0 as the observed
    # rate doubles the threshold.
    excess = min(1.0, (observed_rate - threshold) / threshold)
    return 0.5 + 0.5 * excess
