# SPDX-License-Identifier: AGPL-3.0-or-later
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

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

TENANT_COLUMNS = [
    "tenant_id",
    "events",
    "halts",
    "halt_rate",
    "false_positives",
    "false_positive_rate",
    "alert",
]
SOURCE_COLUMNS = ["source", "halts", "tenants", "last_reason"]
EVIDENCE_COLUMNS = [
    "timestamp",
    "tenant_id",
    "event_id",
    "decision",
    "reason",
    "score",
    "source",
    "action",
]
RETUNE_MIN_FEEDBACK_SAMPLES = 4


@dataclass(frozen=True)
class HaltDashboardRecord:
    """Tenant-safe record used by the safety operations dashboard."""

    tenant_id: str
    event_id: str
    timestamp: str
    decision: str
    reason: str
    halted: bool
    false_positive: bool
    score: float | None
    contradiction_source: str
    action: str


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

    from director_ai.core.training.tuner import format_profile_overlay, tune

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


def parse_dashboard_records(
    events_jsonl: str,
    feedback_jsonl: str = "",
) -> tuple[list[HaltDashboardRecord], list[str]]:
    """Parse dashboard event and feedback JSONL into normalised records."""

    records: list[HaltDashboardRecord] = []
    errors: list[str] = []

    for line_no, item in _jsonl_items(events_jsonl, label="events", errors=errors):
        records.append(_record_from_event(item, fallback_id=f"events:{line_no}"))

    for line_no, item in _jsonl_items(feedback_jsonl, label="feedback", errors=errors):
        record = _record_from_feedback(item, fallback_id=f"feedback:{line_no}")
        if record is not None:
            records.append(record)

    return records, errors


def _feedback_tune_samples(feedback_jsonl: str) -> tuple[list[dict], list[str]]:
    samples: list[dict] = []
    errors: list[str] = []

    for line_no, item in _jsonl_items(feedback_jsonl, label="feedback", errors=errors):
        sample = _feedback_tune_sample(item)
        if sample is None:
            errors.append(
                "feedback:"
                f"{line_no}: expected prompt, response, and human_approved or label"
            )
            continue
        samples.append(sample)

    return samples, errors


def _feedback_tune_sample(item: dict[str, Any]) -> dict[str, Any] | None:
    prompt = _first(item, "prompt", "input", "query", default="")
    response = _first(item, "response", "output", "completion", default="")
    label = _feedback_label(item)
    if not prompt or not response or label is None:
        return None
    return {"prompt": str(prompt), "response": str(response), "label": label}


def _feedback_label(item: dict[str, Any]) -> bool | None:
    human_approved = item.get("human_approved")
    if human_approved is not None:
        return _truthy(human_approved)

    label = item.get("label")
    if isinstance(label, bool):
        return label
    if isinstance(label, int | float):
        return bool(label)
    if isinstance(label, str):
        normalised = label.strip().lower()
        if normalised in {"approved", "approve", "accepted", "correct", "true", "1"}:
            return True
        if normalised in {
            "rejected",
            "reject",
            "blocked",
            "incorrect",
            "false",
            "0",
        }:
            return False
    return None


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

        render = gr.Button("Render Dashboard", variant="primary")
        summary = gr.Markdown()
        tenants = gr.Dataframe(headers=TENANT_COLUMNS, label="Tenant halt rates")
        sources = gr.Dataframe(headers=SOURCE_COLUMNS, label="Contradiction sources")
        evidence = gr.Dataframe(headers=EVIDENCE_COLUMNS, label="Recent halt evidence")
        retune = gr.Code(label="Retune command", language="shell")

        render.click(
            fn=build_safety_dashboard,
            inputs=[events, feedback, halt_threshold, fp_threshold],
            outputs=[summary, tenants, sources, evidence, retune],
        )

    demo.launch(server_port=port, share=share)


def _jsonl_items(
    text: str,
    *,
    label: str,
    errors: list[str],
) -> list[tuple[int, dict[str, Any]]]:
    items: list[tuple[int, dict[str, Any]]] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{label}:{line_no}: {exc.msg}")
            continue
        if not isinstance(parsed, dict):
            errors.append(f"{label}:{line_no}: expected object")
            continue
        items.append((line_no, parsed))
    return items


def _record_from_event(
    item: dict[str, Any],
    *,
    fallback_id: str,
) -> HaltDashboardRecord:
    decision = str(_first(item, "policy_decision", "decision", "state", default=""))
    halted = decision in {"halt", "block", "halted"} or _truthy(item.get("halted"))
    false_positive = (
        _truthy(item.get("false_positive"))
        or str(_first(item, "label", "feedback_label", "outcome", default=""))
        == "false_positive"
    )
    source = _contradiction_source(item)
    return HaltDashboardRecord(
        tenant_id=str(_first_nested(item, "tenant_id", default="default") or "default"),
        event_id=str(_first(item, "event_id", "request_id", default=fallback_id)),
        timestamp=str(_first(item, "timestamp", "created_at", default="")),
        decision=decision or ("halt" if halted else "allow"),
        reason=str(_first(item, "halt_reason", "reason", "halt_cause", default="")),
        halted=halted,
        false_positive=false_positive,
        score=_score(item),
        contradiction_source=source,
        action=str(
            _first(
                item,
                "tenant_safe_explanation",
                "suggested_action",
                "action",
                default="Review recent halt evidence.",
            ),
        ),
    )


def _record_from_feedback(
    item: dict[str, Any],
    *,
    fallback_id: str,
) -> HaltDashboardRecord | None:
    label = str(_first(item, "label", "feedback_label", "outcome", default=""))
    guardrail_approved = item.get("guardrail_approved")
    human_approved = item.get("human_approved")
    false_positive = label == "false_positive" or (
        guardrail_approved is False and human_approved is True
    )
    if not false_positive:
        return None
    return HaltDashboardRecord(
        tenant_id=str(_first_nested(item, "tenant_id", default="default") or "default"),
        event_id=str(_first(item, "event_id", "request_id", default=fallback_id)),
        timestamp=str(_first(item, "timestamp", "created_at", default="")),
        decision="feedback",
        reason=str(_first(item, "reason", "halt_reason", default="false_positive")),
        halted=False,
        false_positive=True,
        score=_score(item),
        contradiction_source=str(
            _first(item, "contradiction_source", "source", default="feedback"),
        ),
        action="Retune from labelled feedback.",
    )


def _tenant_rows(
    records: list[HaltDashboardRecord],
    *,
    halt_alert_threshold: float,
    false_positive_alert_threshold: float,
) -> list[list[Any]]:
    grouped: dict[str, list[HaltDashboardRecord]] = defaultdict(list)
    for record in records:
        grouped[record.tenant_id].append(record)

    rows: list[list[Any]] = []
    for tenant_id in sorted(grouped):
        tenant_records = grouped[tenant_id]
        total = len(tenant_records)
        halts = sum(1 for record in tenant_records if record.halted)
        false_positives = sum(1 for record in tenant_records if record.false_positive)
        halt_rate = halts / total if total else 0.0
        fp_denominator = max(halts, false_positives, 1)
        fp_rate = false_positives / fp_denominator
        alerts: list[str] = []
        if halt_rate >= halt_alert_threshold and total:
            alerts.append("halt-rate")
        if fp_rate >= false_positive_alert_threshold and false_positives:
            alerts.append("false-positive")
        rows.append(
            [
                tenant_id,
                total,
                halts,
                round(halt_rate, 4),
                false_positives,
                round(fp_rate, 4),
                ", ".join(alerts) if alerts else "ok",
            ],
        )
    return rows


def _source_rows(records: list[HaltDashboardRecord]) -> list[list[Any]]:
    counter: Counter[str] = Counter()
    tenants: dict[str, set[str]] = defaultdict(set)
    reasons: dict[str, str] = {}
    for record in records:
        if not record.halted:
            continue
        source = record.contradiction_source or "unknown"
        counter[source] += 1
        tenants[source].add(record.tenant_id)
        if record.reason:
            reasons[source] = record.reason
    return [
        [source, count, len(tenants[source]), reasons.get(source, "")]
        for source, count in counter.most_common()
    ]


def _evidence_rows(records: list[HaltDashboardRecord]) -> list[list[Any]]:
    halted = [record for record in records if record.halted or record.false_positive]
    return [
        [
            record.timestamp,
            record.tenant_id,
            record.event_id,
            record.decision,
            record.reason,
            "" if record.score is None else round(record.score, 4),
            record.contradiction_source or "unknown",
            record.action,
        ]
        for record in halted[-25:]
    ]


def _summary_markdown(
    records: list[HaltDashboardRecord],
    errors: list[str],
    tenant_rows: list[list[Any]],
) -> str:
    total = len(records)
    halts = sum(1 for record in records if record.halted)
    false_positives = sum(1 for record in records if record.false_positive)
    alert_count = sum(1 for row in tenant_rows if row[-1] != "ok")
    lines = [
        "### Safety Operations",
        f"- Events: {total}",
        f"- Halts: {halts}",
        f"- False positives: {false_positives}",
        f"- Tenants with alerts: {alert_count}",
    ]
    if not total:
        lines.append("- Status: load SafetyEvent JSONL or feedback JSONL to begin.")
    if errors:
        lines.append("- Parse warnings: " + "; ".join(errors[:5]))
    return "\n".join(lines)


def _contradiction_source(item: dict[str, Any]) -> str:
    direct = _first(item, "contradiction_source", "source", default="")
    if direct:
        return str(direct)

    attributes = item.get("attributes")
    if isinstance(attributes, dict):
        attr_source = _first(
            attributes,
            "contradiction_source",
            "fact_source",
            "source",
            default="",
        )
        if attr_source:
            return str(attr_source)

    trace = item.get("trace_attribution")
    if isinstance(trace, dict):
        trace_source = _first(trace, "fact_source", "source", default="")
        if trace_source:
            return str(trace_source)

    evidence_refs = item.get("evidence_refs")
    if isinstance(evidence_refs, list) and evidence_refs:
        return str(evidence_refs[0])

    evidence = item.get("halt_evidence") or item.get("halt_evidence_structured")
    if isinstance(evidence, dict):
        return _contradiction_source(evidence)

    chunks = item.get("evidence_chunks")
    if isinstance(chunks, list) and chunks and isinstance(chunks[0], dict):
        return str(_first(chunks[0], "source", "id", default=""))

    return "unknown"


def _score(item: dict[str, Any]) -> float | None:
    value = _first(
        item,
        "observed_score",
        "score",
        "last_score",
        "coherence",
        default=None,
    )
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return default


def _first_nested(mapping: dict[str, Any], key: str, *, default: Any) -> Any:
    direct = mapping.get(key)
    if direct not in (None, ""):
        return direct
    attributes = mapping.get("attributes")
    if isinstance(attributes, dict):
        nested = attributes.get(key)
        if nested not in (None, ""):
            return nested
    return default


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return False
