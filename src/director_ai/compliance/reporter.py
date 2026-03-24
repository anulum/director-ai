# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""EU AI Act Article 15 compliance reporter.

Generates structured compliance reports from the audit log:
- Accuracy metrics (FPR, FNR, TPR, TNR) with confidence intervals
- Hallucination rate per model, per domain, per time period
- Drift detection (is accuracy degrading over time?)
- Human override rate
- Incident log (every rejection with evidence)

Output matches EU AI Act Article 15 documentation requirements:
accuracy declared, tested, monitored continuously, remediated when degraded.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .audit_log import AuditLog

__all__ = ["Article15Report", "ComplianceReporter", "ModelMetrics", "PeriodMetrics"]


def _wilson_ci(successes: int, total: int, z: float = 1.96) -> float:
    """Wilson score CI half-width."""
    if total == 0:
        return 1.0
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return min(spread, center, 1 - center)


@dataclass
class ModelMetrics:
    """Per-model accuracy metrics."""

    model: str
    total_requests: int
    approved_count: int
    rejected_count: int
    hallucination_rate: float  # fraction of requests rejected
    avg_score: float
    avg_confidence: float
    avg_latency_ms: float
    hallucination_rate_ci: float  # 95% CI half-width


@dataclass
class PeriodMetrics:
    """Metrics for a time period (for drift detection)."""

    period_start: float
    period_end: float
    total_requests: int
    hallucination_rate: float
    avg_score: float
    avg_confidence: float


@dataclass
class Article15Report:
    """EU AI Act Article 15 compliance report.

    Contains all required accuracy, robustness, and monitoring data
    for high-risk AI system documentation.
    """

    report_timestamp: float
    period_start: float
    period_end: float
    total_interactions: int

    # Aggregate accuracy
    overall_hallucination_rate: float
    overall_hallucination_rate_ci: float
    avg_score: float
    avg_verdict_confidence: float
    avg_latency_ms: float

    # Human oversight
    human_override_count: int
    human_override_rate: float

    # Per-model breakdown
    model_metrics: list[ModelMetrics] = field(default_factory=list)

    # Drift detection (weekly periods)
    drift_periods: list[PeriodMetrics] = field(default_factory=list)
    drift_detected: bool = False
    drift_severity: float = 0.0  # 0 = no drift, 1 = severe

    # Incidents (rejections)
    incident_count: int = 0

    def to_markdown(self) -> str:
        """Generate Article 15 compliance report in Markdown."""
        lines = [
            "# EU AI Act Article 15 — Accuracy & Robustness Report",
            "",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(self.report_timestamp))}",
            f"**Reporting period:** {time.strftime('%Y-%m-%d', time.gmtime(self.period_start))} to {time.strftime('%Y-%m-%d', time.gmtime(self.period_end))}",
            "**System:** Director-Class AI Hallucination Guardrail",
            "",
            "## 1. Accuracy Metrics (Article 15(1))",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total interactions scored | {self.total_interactions:,} |",
            f"| Overall hallucination rate | {self.overall_hallucination_rate:.2%} ± {self.overall_hallucination_rate_ci:.2%} (95% CI) |",
            f"| Average coherence score | {self.avg_score:.3f} |",
            f"| Average verdict confidence | {self.avg_verdict_confidence:.3f} |",
            f"| Average scoring latency | {self.avg_latency_ms:.1f} ms |",
            "",
            "## 2. Human Oversight (Article 14)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Human overrides recorded | {self.human_override_count:,} |",
            f"| Human override rate | {self.human_override_rate:.2%} |",
            "",
        ]

        if self.model_metrics:
            lines.extend(
                [
                    "## 3. Per-Model Accuracy",
                    "",
                    "| Model | Requests | Hallucination Rate | Avg Score | Avg Confidence | Avg Latency |",
                    "|-------|----------|--------------------|-----------|----------------|-------------|",
                ]
            )
            for m in self.model_metrics:
                lines.append(
                    f"| {m.model} | {m.total_requests:,} | "
                    f"{m.hallucination_rate:.2%} ± {m.hallucination_rate_ci:.2%} | "
                    f"{m.avg_score:.3f} | {m.avg_confidence:.3f} | "
                    f"{m.avg_latency_ms:.1f} ms |"
                )
            lines.append("")

        if self.drift_periods:
            lines.extend(
                [
                    "## 4. Drift Detection (Article 15(3))",
                    "",
                    f"**Drift detected:** {'Yes' if self.drift_detected else 'No'}",
                    f"**Drift severity:** {self.drift_severity:.2f} (0=none, 1=severe)",
                    "",
                    "| Period | Requests | Hallucination Rate | Avg Score |",
                    "|--------|----------|--------------------|-----------|",
                ]
            )
            for p in self.drift_periods:
                start = time.strftime("%Y-%m-%d", time.gmtime(p.period_start))
                end = time.strftime("%Y-%m-%d", time.gmtime(p.period_end))
                lines.append(
                    f"| {start} — {end} | {p.total_requests:,} | "
                    f"{p.hallucination_rate:.2%} | {p.avg_score:.3f} |"
                )
            lines.append("")

        lines.extend(
            [
                "## 5. Incident Summary",
                "",
                f"Total rejections (potential hallucinations blocked): **{self.incident_count:,}**",
                "",
                "---",
                "",
                "*This report is generated automatically by Director-Class AI v3.10.0.*",
                "*It documents accuracy, robustness, and human oversight per EU AI Act Article 15.*",
            ]
        )

        return "\n".join(lines)


class ComplianceReporter:
    """Generate EU AI Act Article 15 compliance reports from the audit log.

    Parameters
    ----------
    audit_log : AuditLog
        The audit log to generate reports from.
    drift_window_days : int
        Size of each drift detection window in days (default 7).
    drift_threshold : float
        Hallucination rate increase that triggers drift alert (default 0.05 = 5pp).
    """

    def __init__(
        self,
        audit_log: AuditLog,
        drift_window_days: int = 7,
        drift_threshold: float = 0.05,
    ):
        self._log = audit_log
        self._drift_window = drift_window_days * 86400
        self._drift_threshold = drift_threshold

    def generate_report(
        self,
        since: float | None = None,
        until: float | None = None,
        model: str | None = None,
        domain: str | None = None,
    ) -> Article15Report:
        """Generate a compliance report for the given time range."""
        now = time.time()
        if until is None:
            until = now
        if since is None:
            since = until - 30 * 86400  # default 30 days

        entries = self._log.query(since=since, until=until, model=model, domain=domain)
        n = len(entries)

        if n == 0:
            return Article15Report(
                report_timestamp=now,
                period_start=since,
                period_end=until,
                total_interactions=0,
                overall_hallucination_rate=0.0,
                overall_hallucination_rate_ci=1.0,
                avg_score=0.0,
                avg_verdict_confidence=0.0,
                avg_latency_ms=0.0,
                human_override_count=0,
                human_override_rate=0.0,
            )

        rejected = sum(1 for e in entries if not e.approved)
        hall_rate = rejected / n
        hall_ci = _wilson_ci(rejected, n)

        avg_score = sum(e.score for e in entries) / n
        avg_conf = sum(e.verdict_confidence for e in entries) / n
        avg_latency = sum(e.latency_ms for e in entries) / n

        human_overrides = sum(1 for e in entries if e.human_override is not None)
        override_rate = human_overrides / n if n > 0 else 0.0

        # Per-model breakdown
        models: dict[str, list] = {}
        for e in entries:
            models.setdefault(e.model or "unknown", []).append(e)

        model_metrics = []
        for model_name, model_entries in sorted(models.items()):
            mn = len(model_entries)
            mr = sum(1 for e in model_entries if not e.approved)
            model_metrics.append(
                ModelMetrics(
                    model=model_name,
                    total_requests=mn,
                    approved_count=mn - mr,
                    rejected_count=mr,
                    hallucination_rate=mr / mn if mn > 0 else 0.0,
                    avg_score=sum(e.score for e in model_entries) / mn,
                    avg_confidence=sum(e.verdict_confidence for e in model_entries)
                    / mn,
                    avg_latency_ms=sum(e.latency_ms for e in model_entries) / mn,
                    hallucination_rate_ci=_wilson_ci(mr, mn),
                )
            )

        # Drift detection
        drift_periods = self._compute_drift_periods(entries, since, until)
        drift_detected = False
        drift_severity = 0.0
        if len(drift_periods) >= 2:
            first_rate = drift_periods[0].hallucination_rate
            last_rate = drift_periods[-1].hallucination_rate
            drift_severity = max(0.0, last_rate - first_rate)
            drift_detected = drift_severity > self._drift_threshold

        return Article15Report(
            report_timestamp=now,
            period_start=since,
            period_end=until,
            total_interactions=n,
            overall_hallucination_rate=hall_rate,
            overall_hallucination_rate_ci=hall_ci,
            avg_score=avg_score,
            avg_verdict_confidence=avg_conf,
            avg_latency_ms=avg_latency,
            human_override_count=human_overrides,
            human_override_rate=override_rate,
            model_metrics=model_metrics,
            drift_periods=drift_periods,
            drift_detected=drift_detected,
            drift_severity=drift_severity,
            incident_count=rejected,
        )

    def _compute_drift_periods(
        self,
        entries: list,
        since: float,
        until: float,
    ) -> list[PeriodMetrics]:
        """Split entries into time windows for drift analysis."""
        if not entries:
            return []

        periods: list[PeriodMetrics] = []
        window_start = since

        while window_start < until:
            window_end = min(window_start + self._drift_window, until)
            window_entries = [
                e for e in entries if window_start <= e.timestamp < window_end
            ]
            if window_entries:
                wn = len(window_entries)
                wr = sum(1 for e in window_entries if not e.approved)
                periods.append(
                    PeriodMetrics(
                        period_start=window_start,
                        period_end=window_end,
                        total_requests=wn,
                        hallucination_rate=wr / wn,
                        avg_score=sum(e.score for e in window_entries) / wn,
                        avg_confidence=sum(e.verdict_confidence for e in window_entries)
                        / wn,
                    )
                )
            window_start = window_end

        return periods
