# SPDX-License-Identifier: BUSL-1.1
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - safety dashboard event analytics

"""Event analytics for the safety operations dashboard.

JSONL parsing of tenant-safe ``SafetyEvent`` and calibration feedback
streams into :class:`~director_ai.ui._dashboard_reports.HaltDashboardRecord`
rows, plus the aggregations the dashboard reports are built from:
tenant/source/evidence tables, drift-alert windows, risk-level
classification, retune sample extraction, and the shared field lookup
and threshold validators. The report models live in
:mod:`._dashboard_reports`; the builders live in :mod:`.safety_dashboard`.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any

from ._dashboard_reports import (
    ComplianceExportRef,
    HaltDashboardRecord,
    TrustControl,
)

__all__ = [
    "_drift_alert_rows",
    "_evidence_rows",
    "_feedback_tune_samples",
    "_operations_risk_level",
    "_source_rows",
    "_summary_markdown",
    "_tenant_rows",
    "_trust_risk_level",
    "_validated_min_window_events",
    "_validated_rate",
    "parse_dashboard_records",
]


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


def _feedback_tune_samples(
    feedback_jsonl: str,
) -> tuple[list[dict[str, object]], list[str]]:
    samples: list[dict[str, object]] = []
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


def _drift_alert_rows(
    records: list[HaltDashboardRecord],
    *,
    drift_alert_threshold: float,
    min_window_events: int,
) -> list[list[Any]]:
    grouped: dict[str, list[HaltDashboardRecord]] = defaultdict(list)
    for record in records:
        if record.decision == "feedback":
            continue
        grouped[record.tenant_id].append(record)

    rows: list[list[Any]] = []
    for tenant_id in sorted(grouped):
        tenant_records = grouped[tenant_id]
        if len(tenant_records) < min_window_events * 2:
            continue
        split_at = len(tenant_records) // 2
        baseline = tenant_records[:split_at]
        current = tenant_records[split_at:]
        # len >= 2*min_window_events above guarantees each half holds at least
        # min_window_events records, so this second window check never trips.
        if (  # pragma: no cover — unreachable: split keeps both windows full
            len(baseline) < min_window_events or len(current) < min_window_events
        ):
            continue
        baseline_rate = _halt_rate(baseline)
        current_rate = _halt_rate(current)
        change = current_rate - baseline_rate
        if change < drift_alert_threshold:
            continue
        severity = _drift_severity(change)
        rows.append(
            [
                tenant_id,
                len(baseline),
                len(current),
                round(baseline_rate, 4),
                round(current_rate, 4),
                round(change, 4),
                severity,
                _drift_recommendation(severity),
            ],
        )
    return rows


def _halt_rate(records: list[HaltDashboardRecord]) -> float:
    return sum(1 for record in records if record.halted) / len(records)


def _drift_severity(rate_change: float) -> str:
    if rate_change >= 0.30:
        return "severe"
    if rate_change >= 0.15:
        return "moderate"
    return "mild"


def _drift_recommendation(severity: str) -> str:
    if severity == "severe":
        return "Freeze rollout, review halt traces, and retune before expansion."
    if severity == "moderate":
        return "Review recent sources and labelled feedback before scaling traffic."
    return "Monitor the next window and collect labelled reviewer feedback."


def _operations_risk_level(
    *,
    tenant_alerts: int,
    drift_alerts: list[list[Any]],
    controls: tuple[TrustControl, ...],
    compliance_exports: tuple[ComplianceExportRef, ...],
    halts: int,
    false_positives: int,
) -> str:
    if any(control.status == "failing" for control in controls):
        return "critical"
    if any(export.status == "missing" for export in compliance_exports):
        return "critical"
    if any(row[6] == "severe" for row in drift_alerts):
        return "critical"
    if any(export.status == "stale" for export in compliance_exports):
        return "attention_required"
    if tenant_alerts or drift_alerts:
        return "attention_required"
    if any(control.status == "warning" for control in controls):
        return "attention_required"
    if halts or false_positives:
        return "monitored"
    return "healthy"


def _trust_risk_level(
    *,
    tenants: list[list[Any]],
    controls: tuple[TrustControl, ...],
    halt_rate: float,
    false_positive_rate: float,
) -> str:
    if any(control.status == "failing" for control in controls):
        return "critical"
    if any(row[-1] != "ok" for row in tenants):
        return "attention_required"
    if any(control.status == "warning" for control in controls):
        return "attention_required"
    if halt_rate > 0.0 or false_positive_rate > 0.0:
        return "monitored"
    return "healthy"


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


def _validated_rate(name: str, value: float) -> float:
    """Return a finite alert-rate threshold in the public dashboard range."""
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def _validated_min_window_events(value: int) -> int:
    """Return a positive per-window event count for drift calculations."""
    if value < 1:
        raise ValueError("min_drift_window_events must be >= 1")
    return value
