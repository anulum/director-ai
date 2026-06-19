# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
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
from typing import Any

from .audit_log import AuditLog

__all__ = [
    "Article15Report",
    "Article15TemplateContext",
    "ComplianceReporter",
    "ModelMetrics",
    "PeriodMetrics",
]


def _wilson_ci(successes: int, total: int, z: float = 1.96) -> float:
    """Wilson score CI half-width."""
    if total == 0:
        return 1.0
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return min(spread, center, 1 - center)


def _require_dict(value: object, field_name: str) -> dict[str, Any]:
    """Return a mapping payload field or raise a stable runtime error."""
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dict")
    return value


def _require_list(value: object, field_name: str) -> list[Any]:
    """Return a list payload field or raise a stable runtime error."""
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return value


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


@dataclass(frozen=True)
class Article15TemplateContext:
    """Operator-supplied context for Article 15 technical documentation."""

    system_name: str
    intended_purpose: str
    deployment_context: str
    risk_management_summary: str
    data_governance_summary: str
    robustness_summary: str
    cybersecurity_summary: str
    human_oversight_summary: str
    post_market_monitoring_summary: str
    known_limitations: tuple[str, ...] = ()
    residual_risks: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject missing required Article 15 system-registration fields."""
        required = {
            "system_name": self.system_name,
            "intended_purpose": self.intended_purpose,
            "deployment_context": self.deployment_context,
            "risk_management_summary": self.risk_management_summary,
            "data_governance_summary": self.data_governance_summary,
            "robustness_summary": self.robustness_summary,
            "cybersecurity_summary": self.cybersecurity_summary,
            "human_oversight_summary": self.human_oversight_summary,
            "post_market_monitoring_summary": self.post_market_monitoring_summary,
        }
        for field_name, value in required.items():
            if not value.strip():
                raise ValueError(f"{field_name} is required")

        object.__setattr__(
            self,
            "known_limitations",
            tuple(item.strip() for item in self.known_limitations if item.strip()),
        )
        object.__setattr__(
            self,
            "residual_risks",
            tuple(item.strip() for item in self.residual_risks if item.strip()),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            tuple(item.strip() for item in self.evidence_refs if item.strip()),
        )


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

    def to_article15_template(
        self,
        context: Article15TemplateContext,
    ) -> dict[str, object]:
        """Return tenant-safe EU AI Act Article 15 technical documentation."""
        return {
            "article": "EU AI Act Article 15",
            "generated_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(self.report_timestamp),
            ),
            "reporting_period": {
                "start": time.strftime("%Y-%m-%d", time.gmtime(self.period_start)),
                "end": time.strftime("%Y-%m-%d", time.gmtime(self.period_end)),
            },
            "system": {
                "name": context.system_name,
                "intended_purpose": context.intended_purpose,
                "deployment_context": context.deployment_context,
            },
            "article_15_sections": {
                "accuracy": {
                    "total_interactions": self.total_interactions,
                    "overall_hallucination_rate": self.overall_hallucination_rate,
                    "overall_hallucination_rate_ci": (
                        self.overall_hallucination_rate_ci
                    ),
                    "avg_score": self.avg_score,
                    "avg_verdict_confidence": self.avg_verdict_confidence,
                    "model_metrics": [
                        {
                            "model": metric.model,
                            "total_requests": metric.total_requests,
                            "approved_count": metric.approved_count,
                            "rejected_count": metric.rejected_count,
                            "hallucination_rate": metric.hallucination_rate,
                            "hallucination_rate_ci": metric.hallucination_rate_ci,
                            "avg_score": metric.avg_score,
                            "avg_confidence": metric.avg_confidence,
                            "avg_latency_ms": metric.avg_latency_ms,
                        }
                        for metric in self.model_metrics
                    ],
                },
                "robustness": {
                    "summary": context.robustness_summary,
                    "drift_detected": self.drift_detected,
                    "drift_severity": self.drift_severity,
                    "drift_periods": [
                        {
                            "period_start": time.strftime(
                                "%Y-%m-%d",
                                time.gmtime(period.period_start),
                            ),
                            "period_end": time.strftime(
                                "%Y-%m-%d",
                                time.gmtime(period.period_end),
                            ),
                            "total_requests": period.total_requests,
                            "hallucination_rate": period.hallucination_rate,
                            "avg_score": period.avg_score,
                            "avg_confidence": period.avg_confidence,
                        }
                        for period in self.drift_periods
                    ],
                },
                "cybersecurity": {"summary": context.cybersecurity_summary},
                "risk_management": {
                    "summary": context.risk_management_summary,
                    "incident_count": self.incident_count,
                },
                "data_governance": {"summary": context.data_governance_summary},
                "human_oversight": {
                    "summary": context.human_oversight_summary,
                    "human_override_count": self.human_override_count,
                    "human_override_rate": self.human_override_rate,
                },
                "post_market_monitoring": {
                    "summary": context.post_market_monitoring_summary,
                    "avg_latency_ms": self.avg_latency_ms,
                },
                "residual_risk": {
                    "known_limitations": list(context.known_limitations),
                    "residual_risks": list(context.residual_risks),
                },
            },
            "evidence_refs": list(context.evidence_refs),
            "privacy": {
                "payload_classification": "tenant_safe",
                "raw_interaction_text_included": False,
            },
        }

    def to_article15_markdown(self, context: Article15TemplateContext) -> str:
        """Render the structured Article 15 template as Markdown."""
        payload = self.to_article15_template(context)
        sections = _require_dict(
            payload["article_15_sections"],
            "article_15_sections",
        )
        accuracy = _require_dict(sections["accuracy"], "article_15_sections.accuracy")
        robustness = _require_dict(
            sections["robustness"],
            "article_15_sections.robustness",
        )
        cybersecurity = _require_dict(
            sections["cybersecurity"],
            "article_15_sections.cybersecurity",
        )
        risk = _require_dict(
            sections["risk_management"],
            "article_15_sections.risk_management",
        )
        data = _require_dict(
            sections["data_governance"],
            "article_15_sections.data_governance",
        )
        oversight = _require_dict(
            sections["human_oversight"],
            "article_15_sections.human_oversight",
        )
        monitoring = _require_dict(
            sections["post_market_monitoring"],
            "article_15_sections.post_market_monitoring",
        )
        residual = _require_dict(
            sections["residual_risk"],
            "article_15_sections.residual_risk",
        )

        lines = [
            "# EU AI Act Article 15 Technical Documentation",
            "",
            "## System",
            "",
            f"- Name: {context.system_name}",
            f"- Intended purpose: {context.intended_purpose}",
            f"- Deployment context: {context.deployment_context}",
            "",
            "## Accuracy, Robustness, and Cybersecurity",
            "",
            f"- Total interactions: {accuracy['total_interactions']:,}",
            "- Overall hallucination rate: "
            f"{accuracy['overall_hallucination_rate']:.2%} ± "
            f"{accuracy['overall_hallucination_rate_ci']:.2%}",
            f"- Average score: {accuracy['avg_score']:.3f}",
            f"- Average verdict confidence: {accuracy['avg_verdict_confidence']:.3f}",
            f"- Robustness controls: {robustness['summary']}",
            f"- Drift detected: {'yes' if robustness['drift_detected'] else 'no'}",
            f"- Drift severity: {robustness['drift_severity']:.3f}",
            f"- Cybersecurity controls: {cybersecurity['summary']}",
            "",
            "## Risk Management",
            "",
            f"- Summary: {risk['summary']}",
            f"- Incident count: {risk['incident_count']:,}",
            "",
            "## Data Governance",
            "",
            f"- Summary: {data['summary']}",
            "",
            "## Human Oversight",
            "",
            f"- Summary: {oversight['summary']}",
            f"- Human overrides: {oversight['human_override_count']:,}",
            f"- Human override rate: {oversight['human_override_rate']:.2%}",
            "",
            "## Post-Market Monitoring",
            "",
            f"- Summary: {monitoring['summary']}",
            f"- Average latency: {monitoring['avg_latency_ms']:.1f} ms",
            "",
            "## Residual Risk",
            "",
        ]
        limitations = _require_list(
            residual["known_limitations"],
            "article_15_sections.residual_risk.known_limitations",
        )
        risks = _require_list(
            residual["residual_risks"],
            "article_15_sections.residual_risk.residual_risks",
        )
        lines.extend(f"- Known limitation: {item}" for item in limitations)
        lines.extend(f"- Residual risk: {item}" for item in risks)
        if not limitations and not risks:
            lines.append("- No residual risks supplied in this template context.")
        lines.extend(["", "## Evidence References", ""])
        evidence_refs = _require_list(payload["evidence_refs"], "evidence_refs")
        if evidence_refs:
            lines.extend(f"- {ref}" for ref in evidence_refs)
        else:
            lines.append("- No evidence references supplied.")
        lines.extend(
            [
                "",
                "## Privacy Boundary",
                "",
                "- Payload classification: tenant_safe",
                "- Raw interaction text included: false",
            ]
        )
        return "\n".join(lines)

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
                "*This report is generated automatically by Director-Class AI v3.11.0.*",
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
        models: dict[str, list[Any]] = {}
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
        entries: list[Any],
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
