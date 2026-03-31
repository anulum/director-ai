# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Statistical drift detection for compliance monitoring.

Uses a two-proportion z-test to determine whether the hallucination rate
has increased significantly between consecutive time windows. No scipy
dependency — the normal CDF is approximated with the Abramowitz & Stegun
rational function (max error ~7.5e-8).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .audit_log import AuditLog

__all__ = ["DriftDetector", "DriftResult", "WindowStats"]


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via Abramowitz & Stegun 26.2.17."""
    if x < -8.0:
        return 0.0
    if x > 8.0:
        return 1.0
    sign = 1.0
    if x < 0:
        sign = -1.0
        x = -x
    t = 1.0 / (1.0 + 0.2316419 * x)
    poly = t * (
        0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    cdf = 1.0 - poly * math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    if sign < 0:
        cdf = 1.0 - cdf
    return cdf


@dataclass
class WindowStats:
    """Metrics for a single time window."""

    start: float
    end: float
    total: int
    rejected: int
    hallucination_rate: float


@dataclass
class DriftResult:
    """Result of a drift detection analysis."""

    detected: bool
    severity: str  # "none", "mild", "moderate", "severe"
    z_score: float
    p_value: float
    rate_change: float  # last window rate minus first window rate
    windows: list[WindowStats] = field(default_factory=list)


class DriftDetector:
    """Detect hallucination rate drift using a two-proportion z-test.

    Compares the first and last time windows in the analysis period.
    If the hallucination rate increase is statistically significant
    (p < alpha), drift is flagged.

    Parameters
    ----------
    audit_log : AuditLog
        Source of scored interaction data.
    window_days : int
        Duration of each comparison window in days. Default 7.
    alpha : float
        Significance level. Default 0.05.
    """

    def __init__(
        self,
        audit_log: AuditLog,
        window_days: int = 7,
        alpha: float = 0.05,
    ):
        self._log = audit_log
        self._window_secs = window_days * 86400
        self._alpha = alpha

    def analyze(
        self,
        since: float | None = None,
        until: float | None = None,
    ) -> DriftResult:
        """Run drift analysis over the given time range.

        Returns a DriftResult with z-score, p-value, severity classification,
        and per-window stats.
        """
        import time as _time

        now = _time.time()
        if until is None:
            until = now
        if since is None:
            since = until - 30 * 86400

        entries = self._log.query(since=since, until=until)
        if not entries:
            return DriftResult(
                detected=False,
                severity="none",
                z_score=0.0,
                p_value=1.0,
                rate_change=0.0,
            )

        windows = self._build_windows(entries, since, until)

        if len(windows) < 2:
            return DriftResult(
                detected=False,
                severity="none",
                z_score=0.0,
                p_value=1.0,
                rate_change=0.0,
                windows=windows,
            )

        first = windows[0]
        last = windows[-1]
        z, p = self._two_proportion_z(
            first.rejected, first.total, last.rejected, last.total
        )
        rate_change = last.hallucination_rate - first.hallucination_rate
        detected = p < self._alpha and rate_change > 0
        severity = self._classify(rate_change, detected, p)

        return DriftResult(
            detected=detected,
            severity=severity,
            z_score=z,
            p_value=p,
            rate_change=rate_change,
            windows=windows,
        )

    def _build_windows(
        self, entries: list, since: float, until: float
    ) -> list[WindowStats]:
        windows: list[WindowStats] = []
        t = since
        while t < until:
            end = min(t + self._window_secs, until)
            w_entries = [e for e in entries if t <= e.timestamp < end]
            if w_entries:
                n = len(w_entries)
                r = sum(1 for e in w_entries if not e.approved)
                windows.append(
                    WindowStats(
                        start=t,
                        end=end,
                        total=n,
                        rejected=r,
                        hallucination_rate=r / n,
                    )
                )
            t = end
        return windows

    @staticmethod
    def _two_proportion_z(r1: int, n1: int, r2: int, n2: int) -> tuple[float, float]:
        """Two-proportion z-test (one-sided: is p2 > p1?).

        Returns (z_score, p_value).
        """
        if n1 == 0 or n2 == 0:
            return 0.0, 1.0
        p1 = r1 / n1
        p2 = r2 / n2
        p_pool = (r1 + r2) / (n1 + n2)
        if p_pool == 0.0 or p_pool == 1.0:
            return 0.0, 1.0
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        if se == 0:
            return 0.0, 1.0
        z = (p2 - p1) / se
        p_value = 1.0 - _norm_cdf(z)
        return z, p_value

    @staticmethod
    def _classify(rate_change: float, detected: bool, p_value: float) -> str:
        if not detected:
            return "none"
        if rate_change < 0.05:
            return "mild"
        if rate_change < 0.15:
            return "moderate"
        return "severe"
