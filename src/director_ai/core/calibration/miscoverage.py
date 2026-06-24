# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Empirical-miscoverage drift monitor for conformal abstention gates.

A conformal gate promises that the true outcome is admitted with probability at
least ``1 - alpha``; its calibrated quantile only delivers that promise while the
live distribution matches the calibration set. Under covariate shift the quantile
keeps returning a confident number while the *realised* miss rate silently climbs
past ``alpha`` — the conformal analogue of a daemon that keeps running while it has
stopped being correct.

This monitor measures the realised miss rate directly, per slice, on a rolling
window, and reports a slice UNHEALTHY when a one-sided statistical *upper* bound on
its miss rate exceeds the target ``alpha`` — so the health verdict trips on the
bound, not on the noisy point estimate. A slice with too few recent samples to
certify coverage is also UNHEALTHY: insufficient evidence fails **closed**, so the
calling gate abstains rather than admitting under an uncertified slice.

It is deliberately a pure accounting primitive: it holds no model and no
thresholds of its own, so both the retrieval-side conformal gate and the
claim-side coverage gate can share one instance and read one consistent answer to
"are we still covered?". The Hoeffding upper bound is distribution-free, which
keeps the RED verdict conservative (fail-closed-leaning) without assuming the
miss process is well-behaved.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass

__all__ = [
    "CoverageHealthReport",
    "MiscoverageMonitor",
    "SegmentCoverageHealth",
    "hoeffding_upper_bound",
]

_logger = logging.getLogger(__name__)


def hoeffding_upper_bound(misses: int, samples: int, delta: float) -> float:
    """Return a one-sided Hoeffding upper bound on the true miss rate.

    For ``misses`` observed misses in ``samples`` trials, the bound holds with
    confidence ``1 - delta``: ``p_hat + sqrt(ln(1 / delta) / (2 n))``, clamped to
    ``[0, 1]``. Distribution-free (no normal-quantile / special functions), so it
    stays valid when the miss process is not well-behaved and keeps the verdict
    conservative. ``samples`` of zero returns ``1.0`` — no evidence cannot certify
    coverage.
    """
    if samples <= 0:
        return 1.0
    p_hat = misses / samples
    slack = math.sqrt(math.log(1.0 / delta) / (2.0 * samples))
    return min(1.0, p_hat + slack)


@dataclass(frozen=True)
class SegmentCoverageHealth:
    """Coverage health for a single slice."""

    segment: str
    samples: int
    miscoverage: float
    upper_bound: float
    target_alpha: float
    healthy: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Serialize for metrics emission and health endpoints."""
        return {
            "segment": self.segment,
            "samples": self.samples,
            "miscoverage": round(self.miscoverage, 4),
            "upper_bound": round(self.upper_bound, 4),
            "target_alpha": self.target_alpha,
            "healthy": self.healthy,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CoverageHealthReport:
    """Fleet-wide coverage health: overall verdict plus per-slice detail."""

    healthy: bool
    segments: tuple[SegmentCoverageHealth, ...]
    unhealthy_segments: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize for metrics emission and health endpoints."""
        return {
            "healthy": self.healthy,
            "unhealthy_segments": list(self.unhealthy_segments),
            "segments": [s.to_dict() for s in self.segments],
        }


class MiscoverageMonitor:
    """Track realised per-slice miss rate and fail closed on coverage drift.

    Feed one boolean per recall/decision via :meth:`observe` — ``covered=True``
    when the admitted prediction held the true outcome, ``covered=False`` for a
    miss. A slice is healthy only once it has at least ``min_samples`` recent
    observations *and* the Hoeffding upper bound on its miss rate stays at or below
    ``target_alpha``; otherwise it is UNHEALTHY and the calling gate must abstain
    for that slice.
    """

    def __init__(
        self,
        target_alpha: float,
        *,
        window: int = 500,
        min_samples: int = 30,
        confidence: float = 0.95,
    ) -> None:
        if not 0.0 < target_alpha < 1.0:
            raise ValueError(f"target_alpha must be in (0, 1), got {target_alpha}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {min_samples}")
        if min_samples > window:
            raise ValueError(
                f"min_samples ({min_samples}) must be <= window ({window})"
            )
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")
        self._target_alpha = target_alpha
        self._window = window
        self._min_samples = min_samples
        self._delta = 1.0 - confidence
        self._slices: dict[str, deque[bool]] = {}

    @property
    def target_alpha(self) -> float:
        """The miss-rate ceiling each slice is held to."""
        return self._target_alpha

    def observe(self, segment: str, *, covered: bool) -> None:
        """Record one outcome for ``segment`` (``covered`` False is a miss)."""
        key = segment.strip()
        if not key:
            raise ValueError("segment must be a non-empty string")
        window = self._slices.get(key)
        if window is None:
            window = deque(maxlen=self._window)
            self._slices[key] = window
        window.append(covered)

    def _counts(self, segment: str) -> tuple[int, int]:
        """Return ``(misses, samples)`` for ``segment`` in the current window."""
        window = self._slices.get(segment.strip())
        if not window:
            return 0, 0
        samples = len(window)
        misses = samples - sum(window)
        return misses, samples

    def miscoverage(self, segment: str) -> float | None:
        """Point-estimate miss rate for ``segment``, or ``None`` if unseen."""
        misses, samples = self._counts(segment)
        if samples == 0:
            return None
        return misses / samples

    def miscoverage_upper_bound(self, segment: str) -> float | None:
        """One-sided upper bound on ``segment``'s miss rate, or ``None``."""
        misses, samples = self._counts(segment)
        if samples == 0:
            return None
        return hoeffding_upper_bound(misses, samples, self._delta)

    def segment_health(self, segment: str) -> SegmentCoverageHealth:
        """Full coverage-health verdict for one slice (fail-closed)."""
        key = segment.strip()
        misses, samples = self._counts(key)
        point = misses / samples if samples else 1.0
        upper = hoeffding_upper_bound(misses, samples, self._delta)
        if samples < self._min_samples:
            healthy, reason = False, "insufficient_samples"
        elif upper > self._target_alpha:
            healthy, reason = False, "miscoverage_exceeds_alpha"
        else:
            healthy, reason = True, "within_target"
        return SegmentCoverageHealth(
            segment=key,
            samples=samples,
            miscoverage=point,
            upper_bound=upper,
            target_alpha=self._target_alpha,
            healthy=healthy,
            reason=reason,
        )

    def is_healthy(self, segment: str) -> bool:
        """Whether ``segment`` currently certifies coverage (fail-closed)."""
        return self.segment_health(segment).healthy

    def health(self) -> CoverageHealthReport:
        """Aggregate health: RED if any known slice is unhealthy or none exist.

        An empty monitor is UNHEALTHY: with nothing observed there is nothing to
        certify, so the gate fails closed until calibration data arrives.
        """
        segments = tuple(self.segment_health(key) for key in sorted(self._slices))
        unhealthy = tuple(s.segment for s in segments if not s.healthy)
        overall = bool(segments) and not unhealthy
        if not overall:
            _logger.warning(
                "coverage health RED: %s",
                "no observed slices"
                if not segments
                else f"unhealthy={list(unhealthy)}",
            )
        return CoverageHealthReport(
            healthy=overall,
            segments=segments,
            unhealthy_segments=unhealthy,
        )

    def snapshot(self) -> dict[str, object]:
        """Plain-dict view of current health for metrics emission."""
        return self.health().to_dict()

    def reset(self, segment: str | None = None) -> None:
        """Drop all observations, or just ``segment``'s when given."""
        if segment is None:
            self._slices.clear()
            return
        self._slices.pop(segment.strip(), None)
