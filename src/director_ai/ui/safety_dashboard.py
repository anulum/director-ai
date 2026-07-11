# SPDX-License-Identifier: BUSL-1.1
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - safety operations dashboard

"""Lightweight safety operations dashboard helpers.

The dashboard consumes tenant-safe ``SafetyEvent`` JSONL plus optional
calibration feedback JSONL. It does not need a running server, database,
or Gradio dependency for its core summaries.
"""

from __future__ import annotations

from typing import Any

from ._dashboard_analytics import (
    _drift_alert_rows,
    _evidence_rows,
    _feedback_tune_samples,
    _operations_risk_level,
    _source_rows,
    _summary_markdown,
    _tenant_rows,
    _trust_risk_level,
    _validated_min_window_events,
    _validated_rate,
)
from ._dashboard_analytics import parse_dashboard_records as parse_dashboard_records
from ._dashboard_reports import (
    COMPLIANCE_EXPORT_COLUMNS as COMPLIANCE_EXPORT_COLUMNS,
)
from ._dashboard_reports import DRIFT_ALERT_COLUMNS as DRIFT_ALERT_COLUMNS
from ._dashboard_reports import EVIDENCE_COLUMNS as EVIDENCE_COLUMNS
from ._dashboard_reports import SOURCE_COLUMNS as SOURCE_COLUMNS
from ._dashboard_reports import TENANT_COLUMNS as TENANT_COLUMNS
from ._dashboard_reports import ComplianceExportRef as ComplianceExportRef
from ._dashboard_reports import ComplianceExportStatus as ComplianceExportStatus
from ._dashboard_reports import HaltDashboardRecord as HaltDashboardRecord
from ._dashboard_reports import (
    ObservabilityOperationsReport as ObservabilityOperationsReport,
)
from ._dashboard_reports import TrustConsoleReport as TrustConsoleReport
from ._dashboard_reports import TrustControl as TrustControl
from ._dashboard_reports import TrustControlStatus as TrustControlStatus

RETUNE_MIN_FEEDBACK_SAMPLES = 4


def build_safety_dashboard(
    events_jsonl: str,
    feedback_jsonl: str = "",
    halt_alert_threshold: float = 0.15,
    false_positive_alert_threshold: float = 0.05,
) -> tuple[str, list[list[Any]], list[list[Any]], list[list[Any]], str]:
    """Build summary tables for tenant halt operations.

    Returns
    -------
    tuple
        Markdown summary, tenant table rows, contradiction-source rows,
        recent evidence rows, and a ready-to-run retune command.
    """
    halt_alert_threshold = _validated_rate("halt_alert_threshold", halt_alert_threshold)
    false_positive_alert_threshold = _validated_rate(
        "false_positive_alert_threshold", false_positive_alert_threshold
    )
    records, errors = parse_dashboard_records(events_jsonl, feedback_jsonl)
    tenant_rows = _tenant_rows(
        records,
        halt_alert_threshold=halt_alert_threshold,
        false_positive_alert_threshold=false_positive_alert_threshold,
    )
    source_rows = _source_rows(records)
    evidence_rows = _evidence_rows(records)
    summary = _summary_markdown(records, errors, tenant_rows)
    command = (
        "director-ai tune --dataset recent_feedback.jsonl "
        "--output director-ai-tuned.yaml"
    )
    return summary, tenant_rows, source_rows, evidence_rows, command


def build_observability_operations_report(
    events_jsonl: str,
    feedback_jsonl: str = "",
    *,
    controls: list[TrustControl] | tuple[TrustControl, ...] = (),
    compliance_exports: list[ComplianceExportRef]
    | tuple[ComplianceExportRef, ...] = (),
    title: str = "Director-AI Observability Operations",
    generated_at: str = "",
    halt_alert_threshold: float = 0.15,
    false_positive_alert_threshold: float = 0.05,
    drift_alert_threshold: float = 0.1,
    min_drift_window_events: int = 2,
) -> ObservabilityOperationsReport:
    """Build a tenant-safe halt-forensics and drift operations packet."""
    halt_alert_threshold = _validated_rate("halt_alert_threshold", halt_alert_threshold)
    false_positive_alert_threshold = _validated_rate(
        "false_positive_alert_threshold", false_positive_alert_threshold
    )
    drift_alert_threshold = _validated_rate(
        "drift_alert_threshold", drift_alert_threshold
    )
    min_drift_window_events = _validated_min_window_events(min_drift_window_events)
    records, errors = parse_dashboard_records(events_jsonl, feedback_jsonl)
    tenants = _tenant_rows(
        records,
        halt_alert_threshold=halt_alert_threshold,
        false_positive_alert_threshold=false_positive_alert_threshold,
    )
    sources = _source_rows(records)
    evidence = _evidence_rows(records)
    drift_alerts = _drift_alert_rows(
        records,
        drift_alert_threshold=drift_alert_threshold,
        min_window_events=min_drift_window_events,
    )
    control_tuple = tuple(controls)
    export_tuple = tuple(compliance_exports)
    total = len(records)
    halts = sum(1 for record in records if record.halted)
    false_positives = sum(1 for record in records if record.false_positive)
    tenant_alerts = sum(1 for row in tenants if row[-1] != "ok")
    risk_level = _operations_risk_level(
        tenant_alerts=tenant_alerts,
        drift_alerts=drift_alerts,
        controls=control_tuple,
        compliance_exports=export_tuple,
        halts=halts,
        false_positives=false_positives,
    )
    return ObservabilityOperationsReport(
        title=title,
        generated_at=generated_at or "unspecified",
        summary={
            "total_events": total,
            "tenants": len(tenants),
            "halts": halts,
            "false_positives": false_positives,
            "tenant_alerts": tenant_alerts,
            "drift_alerts": len(drift_alerts),
            "control_failures": sum(
                1 for control in control_tuple if control.status == "failing"
            ),
            "compliance_export_gaps": sum(
                1 for export in export_tuple if export.status in {"missing", "stale"}
            ),
            "risk_level": risk_level,
        },
        tenants=tenants,
        sources=sources,
        recent_evidence=evidence,
        drift_alerts=drift_alerts,
        controls=control_tuple,
        compliance_exports=export_tuple,
        parse_warnings=tuple(errors),
    )


def build_observability_operations_markdown(
    events_jsonl: str,
    feedback_jsonl: str = "",
    halt_alert_threshold: float = 0.15,
    false_positive_alert_threshold: float = 0.05,
    drift_alert_threshold: float = 0.1,
) -> str:
    """Render the default operations report for the dashboard UI."""
    return build_observability_operations_report(
        events_jsonl,
        feedback_jsonl,
        halt_alert_threshold=halt_alert_threshold,
        false_positive_alert_threshold=false_positive_alert_threshold,
        drift_alert_threshold=drift_alert_threshold,
    ).to_markdown()


def build_trust_console_report(
    events_jsonl: str,
    feedback_jsonl: str = "",
    *,
    controls: list[TrustControl] | tuple[TrustControl, ...] = (),
    title: str = "Director-AI Trust Console",
    generated_at: str = "",
    halt_alert_threshold: float = 0.15,
    false_positive_alert_threshold: float = 0.05,
) -> TrustConsoleReport:
    """Build a tenant-safe customer Trust Console report."""
    halt_alert_threshold = _validated_rate("halt_alert_threshold", halt_alert_threshold)
    false_positive_alert_threshold = _validated_rate(
        "false_positive_alert_threshold", false_positive_alert_threshold
    )
    records, errors = parse_dashboard_records(events_jsonl, feedback_jsonl)
    tenants = _tenant_rows(
        records,
        halt_alert_threshold=halt_alert_threshold,
        false_positive_alert_threshold=false_positive_alert_threshold,
    )
    evidence = _evidence_rows(records)
    total = len(records)
    halts = sum(1 for record in records if record.halted)
    false_positives = sum(1 for record in records if record.false_positive)
    alert_count = sum(1 for row in tenants if row[-1] != "ok")
    halt_rate = round(halts / total, 4) if total else 0.0
    fp_denominator = max(halts, false_positives, 1)
    false_positive_rate = round(false_positives / fp_denominator, 4)
    control_tuple = tuple(controls)
    risk_level = _trust_risk_level(
        tenants=tenants,
        controls=control_tuple,
        halt_rate=halt_rate,
        false_positive_rate=false_positive_rate,
    )
    return TrustConsoleReport(
        title=title,
        generated_at=generated_at or "unspecified",
        summary={
            "total_events": total,
            "tenants": len(tenants),
            "halts": halts,
            "halt_rate": halt_rate,
            "false_positives": false_positives,
            "false_positive_rate": false_positive_rate,
            "tenant_alerts": alert_count,
            "control_warnings": sum(
                1 for control in control_tuple if control.status == "warning"
            ),
            "control_failures": sum(
                1 for control in control_tuple if control.status == "failing"
            ),
            "risk_level": risk_level,
        },
        tenants=tenants,
        recent_evidence=evidence,
        controls=control_tuple,
        parse_warnings=tuple(errors),
    )


def build_retune_guidance(
    feedback_jsonl: str,
    profile: str = "tuned",
    base_profile: str = "",
    min_samples: int = RETUNE_MIN_FEEDBACK_SAMPLES,
) -> tuple[str, str]:
    """Build a tuned profile overlay from recent labelled feedback JSONL."""
    samples, errors = _feedback_tune_samples(feedback_jsonl)
    if len(samples) < min_samples:
        lines = [
            "### Retune Guidance",
            f"- Labelled samples: {len(samples)}",
            f"- Required samples: {min_samples}",
            "- Status: collect more labelled feedback with prompt, response, and human verdict.",
        ]
        if errors:
            lines.append("- Parse warnings: " + "; ".join(errors[:5]))
        return "\n".join(lines), ""

    from director_ai.core.calibration.tuner import format_profile_overlay, tune

    result = tune(samples)
    labels = [bool(sample["label"]) for sample in samples]
    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    overlay_profile = profile.strip() or (
        f"{base_profile.strip()}_tuned" if base_profile.strip() else "tuned"
    )
    overlay = format_profile_overlay(
        result,
        profile=overlay_profile,
        base_profile=base_profile.strip(),
    )

    lines = [
        "### Retune Guidance",
        f"- Labelled samples: {len(samples)}",
        f"- Approved labels: {positives}",
        f"- Rejected labels: {negatives}",
        f"- Selected threshold: {result.threshold:.4f}",
        f"- Balanced accuracy: {result.balanced_accuracy:.4f}",
        f"- Confidence: {result.confidence_level}",
    ]
    if positives == 0 or negatives == 0:
        lines.append(
            "- Warning: only one label class present; treat this overlay as provisional."
        )
    if errors:
        lines.append("- Parse warnings: " + "; ".join(errors[:5]))
    return "\n".join(lines), overlay


def launch_safety_dashboard(port: int = 7861, share: bool = False) -> None:
    """Launch the Gradio safety operations dashboard."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            "Safety dashboard requires Gradio. Install with: pip install director-ai[ui]"
        ) from exc

    with gr.Blocks(title="Director-AI Safety Operations") as demo:
        gr.Markdown("# Director-AI Safety Operations")

        events = gr.Textbox(label="SafetyEvent JSONL", lines=14)
        feedback = gr.Textbox(label="Feedback JSONL", lines=8)
        with gr.Row():
            halt_threshold = gr.Slider(
                label="Halt-rate alert threshold",
                minimum=0.0,
                maximum=1.0,
                value=0.15,
                step=0.01,
            )
            fp_threshold = gr.Slider(
                label="False-positive alert threshold",
                minimum=0.0,
                maximum=1.0,
                value=0.05,
                step=0.01,
            )
            drift_threshold = gr.Slider(
                label="Drift alert threshold",
                minimum=0.0,
                maximum=1.0,
                value=0.10,
                step=0.01,
            )

        render = gr.Button("Render Dashboard", variant="primary")
        summary = gr.Markdown()
        tenants = gr.Dataframe(headers=TENANT_COLUMNS, label="Tenant halt rates")
        sources = gr.Dataframe(headers=SOURCE_COLUMNS, label="Contradiction sources")
        evidence = gr.Dataframe(headers=EVIDENCE_COLUMNS, label="Recent halt evidence")
        retune = gr.Code(label="Retune command", language="shell")
        operations = gr.Markdown(label="Observability operations report")

        render.click(
            fn=build_safety_dashboard,
            inputs=[events, feedback, halt_threshold, fp_threshold],
            outputs=[summary, tenants, sources, evidence, retune],
        )
        render.click(
            fn=build_observability_operations_markdown,
            inputs=[
                events,
                feedback,
                halt_threshold,
                fp_threshold,
                drift_threshold,
            ],
            outputs=[operations],
        )

    demo.launch(server_port=port, share=share)
