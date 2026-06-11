# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — runtime application self-protection

"""Runtime self-protection: flag attacks in progress from behavioural anomalies.

When every input filter and guardrail has been bypassed, the last line of defence
is the application watching its own behaviour: a sudden request-rate spike, an
oversized payload, a halt-rate surge. :class:`RuntimeSelfProtection` keeps a
per-metric :class:`~director_ai.core.rasp.detector.StreamingRobustDetector` and
turns each observation into a tenant-safe :class:`AnomalyVerdict` (ok / watch /
alert), tracking how many recent observations were anomalous so a caller can shed
load or escalate. It observes and scores; the host decides whether to block.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from .detector import StreamingRobustDetector

__all__ = ["AnomalyVerdict", "RuntimeSelfProtection"]

OK = "ok"
WATCH = "watch"
ALERT = "alert"


@dataclass(frozen=True)
class AnomalyVerdict:
    """The self-protection assessment of one observation."""

    metric: str
    value: float
    robust_z: float
    severity: str
    anomalous: bool
    cold_start: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a tenant-safe JSON dict (metric name + score, no payload)."""
        return {
            "metric": self.metric,
            "value": self.value,
            "robust_z": self.robust_z,
            "severity": self.severity,
            "anomalous": self.anomalous,
            "cold_start": self.cold_start,
        }


class RuntimeSelfProtection:
    """Per-metric behavioural anomaly monitor (RASP)."""

    def __init__(
        self,
        *,
        window_size: int = 128,
        min_samples: int = 12,
        z_threshold: float = 3.5,
        watch_fraction: float = 0.7,
        recent_window: int = 64,
    ) -> None:
        if not 0.0 < watch_fraction < 1.0:
            raise ValueError("watch_fraction must be in (0, 1)")
        if recent_window < 1:
            raise ValueError("recent_window must be at least 1")
        self._window_size = window_size
        self._min_samples = min_samples
        self._z_threshold = z_threshold
        self._watch_fraction = watch_fraction
        self._detectors: dict[str, StreamingRobustDetector] = {}
        self._recent: deque[bool] = deque(maxlen=recent_window)

    def _detector(self, metric: str) -> StreamingRobustDetector:
        detector = self._detectors.get(metric)
        if detector is None:
            detector = StreamingRobustDetector(
                window_size=self._window_size,
                min_samples=self._min_samples,
                z_threshold=self._z_threshold,
            )
            self._detectors[metric] = detector
        return detector

    def observe(self, metric: str, value: float) -> AnomalyVerdict:
        """Score one observation of ``metric`` and record it in the baseline."""
        if not metric.strip():
            raise ValueError("metric name is required")
        score = self._detector(metric).update(value)
        if score.cold_start:
            severity = OK
        elif score.anomalous:
            severity = ALERT
        elif score.robust_z > self._z_threshold * self._watch_fraction:
            severity = WATCH
        else:
            severity = OK
        self._recent.append(score.anomalous)
        return AnomalyVerdict(
            metric=metric,
            value=score.value,
            robust_z=score.robust_z,
            severity=severity,
            anomalous=score.anomalous,
            cold_start=score.cold_start,
        )

    @property
    def recent_anomaly_count(self) -> int:
        """Number of anomalous observations in the recent-window."""
        return sum(self._recent)

    @property
    def tracked_metrics(self) -> tuple[str, ...]:
        """The metric names with an active detector."""
        return tuple(sorted(self._detectors))

    def under_attack(self, *, min_anomalies: int = 3) -> bool:
        """Whether enough recent observations were anomalous to suspect an attack."""
        if min_anomalies < 1:
            raise ValueError("min_anomalies must be at least 1")
        return self.recent_anomaly_count >= min_anomalies
