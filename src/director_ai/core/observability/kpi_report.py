# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Board-Level KPI Presentation
"""Render the board-level KPIs as a status-flagged report.

:func:`director_ai.core.observability.kpis.compute_kpis` produces the raw KPI
numbers; this is the presentation layer over them — it classifies each metric
against operating targets (``ok`` / ``watch`` / ``alert``) and renders a
board-facing Markdown or plain-text summary. It is the human-readable / CLI export
half of the operations dashboard; the data is tenant-safe (aggregates only).
"""

from __future__ import annotations

from dataclasses import dataclass

from .kpis import KpiReport

OK = "ok"
WATCH = "watch"
ALERT = "alert"
NOT_AVAILABLE = "n/a"


@dataclass(frozen=True)
class KpiTargets:
    """Operating targets that classify each KPI into ok / watch / alert."""

    max_false_positive_rate: float = 0.10
    min_halt_precision: float = 0.80
    max_p95_latency_ms: float = 100.0
    watch_fraction: float = 0.8

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_false_positive_rate <= 1.0:
            raise ValueError("max_false_positive_rate must be in [0, 1]")
        if not 0.0 <= self.min_halt_precision <= 1.0:
            raise ValueError("min_halt_precision must be in [0, 1]")
        if self.max_p95_latency_ms <= 0:
            raise ValueError("max_p95_latency_ms must be positive")
        if not 0.0 < self.watch_fraction < 1.0:
            raise ValueError("watch_fraction must be in (0, 1)")


def _status_upper(value: float | None, limit: float, watch_fraction: float) -> str:
    """Status for a metric where lower is better (alert when above ``limit``)."""
    if value is None:
        return NOT_AVAILABLE
    if value > limit:
        return ALERT
    if value > limit * watch_fraction:
        return WATCH
    return OK


def _status_lower(value: float | None, floor: float, watch_fraction: float) -> str:
    """Status for a metric where higher is better (alert when below ``floor``)."""
    if value is None:
        return NOT_AVAILABLE
    if value < floor:
        return ALERT
    # Within the top ``1 - watch_fraction`` band below 1.0 counts as watch.
    if value < floor + (1.0 - floor) * (1.0 - watch_fraction):
        return WATCH
    return OK


def kpi_statuses(
    report: KpiReport, targets: KpiTargets | None = None
) -> dict[str, str]:
    """Classify each KPI against the targets into ok / watch / alert / n/a."""
    t = targets or KpiTargets()
    statuses = {
        "false_positive_rate": _status_upper(
            report.false_positive_rate, t.max_false_positive_rate, t.watch_fraction
        ),
        "halt_precision": _status_lower(
            report.halt_precision, t.min_halt_precision, t.watch_fraction
        ),
        "p95_scoring_latency_ms": _status_upper(
            report.p95_scoring_latency_ms, t.max_p95_latency_ms, t.watch_fraction
        ),
        "tenant_boundary_violations": (
            ALERT if report.tenant_boundary_violations > 0 else OK
        ),
        "security_exception_debt": (
            WATCH if report.security_exception_debt > 0 else OK
        ),
    }
    for domain, rate in report.per_domain_false_positive_rate.items():
        statuses[f"false_positive_rate[{domain}]"] = _status_upper(
            rate, t.max_false_positive_rate, t.watch_fraction
        )
    return statuses


def overall_status(report: KpiReport, targets: KpiTargets | None = None) -> str:
    """Worst per-metric status across the report (alert > watch > ok)."""
    values = set(kpi_statuses(report, targets).values())
    if ALERT in values:
        return ALERT
    if WATCH in values:
        return WATCH
    return OK


def _fmt(value: float | None, *, pct: bool = False, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if pct:
        return f"{value * 100:.2f}%"
    return f"{value:g}{suffix}"


def render_markdown(report: KpiReport, *, targets: KpiTargets | None = None) -> str:
    """Render a board-facing Markdown KPI report with per-metric status."""
    statuses = kpi_statuses(report, targets)
    rows = [
        ("Labelled decisions", str(report.labelled_total), OK),
        ("Halt rate", _fmt(report.halt_rate, pct=True), OK),
        (
            "Halt precision",
            _fmt(report.halt_precision, pct=True),
            statuses["halt_precision"],
        ),
        (
            "False-positive rate",
            _fmt(report.false_positive_rate, pct=True),
            statuses["false_positive_rate"],
        ),
        (
            "p95 scoring latency",
            _fmt(report.p95_scoring_latency_ms, suffix=" ms"),
            statuses["p95_scoring_latency_ms"],
        ),
        (
            "Tenant boundary violations",
            str(report.tenant_boundary_violations),
            statuses["tenant_boundary_violations"],
        ),
        (
            "Unsigned KB writes rejected",
            str(report.unsigned_kb_writes_rejected),
            OK,
        ),
        (
            "Security exception debt",
            str(report.security_exception_debt),
            statuses["security_exception_debt"],
        ),
    ]
    lines = [
        f"# Guardrail KPIs — overall: {overall_status(report, targets).upper()}",
        "",
        "| Metric | Value | Status |",
        "|--------|-------|--------|",
    ]
    lines.extend(f"| {name} | {value} | {status} |" for name, value, status in rows)
    if report.per_domain_false_positive_rate:
        lines.extend(["", "## Per-domain false-positive rate", ""])
        lines.append("| Domain | FPR | Status |")
        lines.append("|--------|-----|--------|")
        for domain in sorted(report.per_domain_false_positive_rate):
            rate = report.per_domain_false_positive_rate[domain]
            lines.append(
                f"| {domain} | {_fmt(rate, pct=True)} | "
                f"{statuses[f'false_positive_rate[{domain}]']} |"
            )
    return "\n".join(lines)


def render_text(report: KpiReport, *, targets: KpiTargets | None = None) -> str:
    """Render a plain-text KPI report with per-metric status."""
    statuses = kpi_statuses(report, targets)
    lines = [f"Guardrail KPIs (overall: {overall_status(report, targets)})"]
    lines.append(f"  labelled_decisions: {report.labelled_total}")
    lines.append(f"  halt_rate: {_fmt(report.halt_rate, pct=True)}")
    lines.append(
        f"  halt_precision: {_fmt(report.halt_precision, pct=True)} "
        f"[{statuses['halt_precision']}]"
    )
    lines.append(
        f"  false_positive_rate: {_fmt(report.false_positive_rate, pct=True)} "
        f"[{statuses['false_positive_rate']}]"
    )
    lines.append(
        f"  p95_scoring_latency: {_fmt(report.p95_scoring_latency_ms, suffix=' ms')} "
        f"[{statuses['p95_scoring_latency_ms']}]"
    )
    lines.append(
        f"  tenant_boundary_violations: {report.tenant_boundary_violations} "
        f"[{statuses['tenant_boundary_violations']}]"
    )
    lines.append(f"  unsigned_kb_writes_rejected: {report.unsigned_kb_writes_rejected}")
    lines.append(
        f"  security_exception_debt: {report.security_exception_debt} "
        f"[{statuses['security_exception_debt']}]"
    )
    return "\n".join(lines)
