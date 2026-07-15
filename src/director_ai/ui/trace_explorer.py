# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — trace explorer

"""Execution-trace explorer for streaming, agent, and swarm traces.

Parses a streaming/agent/swarm trace JSON document into a Markdown summary, an
event table, and a structured detail payload for the configuration wizard's
Gradio surface. Split out of :mod:`director_ai.ui.config_wizard` so trace
inspection is independent of config-YAML generation.
"""

from __future__ import annotations

import json
from typing import Any


def build_trace_explorer(
    trace_json: str,
) -> tuple[str, list[list[Any]], dict[str, Any]]:
    """Build a trace summary, event table, and detail payload for the UI."""
    raw = trace_json.strip()
    if not raw:
        return (
            "Paste a streaming, agent, or swarm trace JSON document.",
            [],
            {"error": "empty input"},
        )

    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        return (
            f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}.",
            [],
            {"error": exc.msg, "line": exc.lineno, "column": exc.colno},
        )

    events = _trace_events(document)
    rows = [_trace_row(index, event) for index, event in enumerate(events)]
    halted = _trace_halted(document, events)
    halt_reason = _trace_halt_reason(document, events)
    counterfactual = _trace_counterfactual(document, events)
    attribution = _trace_attribution(document, events)
    scopes = sorted({row[1] for row in rows if row[1]})

    summary_lines = [
        "### Trace Explorer",
        f"- Events: {len(rows)}",
        f"- Scopes: {', '.join(scopes) if scopes else 'unknown'}",
        f"- Halted: {'yes' if halted else 'no'}",
    ]
    if halt_reason:
        summary_lines.append(f"- Halt reason: {halt_reason}")
    if attribution:
        summary_lines.append(
            "- Attribution: "
            f"{attribution.get('scorer_path', 'unknown scorer')} at token "
            f"{attribution.get('token_offset', 'unknown')}"
        )
    if counterfactual:
        summary_lines.append(
            "- Counterfactual: "
            f"{counterfactual.get('fact_source', 'unknown fact')} needs delta "
            f"{counterfactual.get('required_score_delta', 'unknown')}"
        )

    detail = {
        "event_count": len(rows),
        "halted": halted,
        "halt_reason": halt_reason,
        "scopes": scopes,
        "counterfactual": counterfactual,
        "trace_attribution": attribution,
    }
    return "\n".join(summary_lines), rows, detail


def _trace_events(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, list):
        return [_trace_mapping(event, "trace") for event in document]
    if not isinstance(document, dict):
        return [{"event_type": "value", "value": document, "_scope_hint": "trace"}]

    events: list[dict[str, Any]] = []
    for key, scope in (
        ("events", "streaming"),
        ("trace", "trace"),
        ("safety_events", "safety"),
        ("halt_events", "safety"),
        ("agent_events", "agent"),
        ("swarm_events", "swarm"),
    ):
        value = document.get(key)
        if isinstance(value, list):
            events.extend(_trace_mapping(event, scope) for event in value)

    evidence = document.get("halt_evidence_structured") or document.get(
        "halt_evidence",
    )
    if isinstance(evidence, dict):
        halt_event = dict(evidence)
        halt_event.setdefault("event_type", "halt_evidence")
        halt_event["_scope_hint"] = "streaming"
        events.append(halt_event)

    if not events and any(key in document for key in ("halted", "halt_reason")):
        root_event = dict(document)
        root_event.setdefault("event_type", "session")
        root_event["_scope_hint"] = "streaming"
        events.append(root_event)

    return events


def _trace_mapping(event: Any, scope: str) -> dict[str, Any]:
    if isinstance(event, dict):
        mapped = dict(event)
    else:
        mapped = {"event_type": "value", "value": event}
    mapped.setdefault("_scope_hint", scope)
    return mapped


def _trace_row(index: int, event: dict[str, Any]) -> list[Any]:
    return [
        _first_present(event, ("index", "step", "token_offset"), index),
        _trace_scope(event),
        str(_first_present(event, ("event_type", "type", "name"), "event")),
        _trace_state(event),
        _trace_score(event),
        str(_first_present(event, ("hook_id", "hook", "scorer_path"), "")),
        _trace_event_reason(event),
        _trace_detail(event),
    ]


def _trace_scope(event: dict[str, Any]) -> str:
    explicit = _first_present(event, ("hook_scope", "scope", "source"), "")
    if explicit:
        return str(explicit)
    scope_hint = _first_present(event, ("_scope_hint",), "")
    if scope_hint and scope_hint != "safety":
        return str(scope_hint)
    event_type = str(_first_present(event, ("event_type", "type", "name"), ""))
    if "swarm" in event_type:
        return "swarm"
    if "agent" in event_type or "agent_id" in event:
        return "agent"
    if "token" in event or "coherence" in event:
        return "streaming"
    return "trace"


def _trace_state(event: dict[str, Any]) -> str:
    decision = _first_present(event, ("policy_decision", "decision", "state"), "")
    if decision:
        return str(decision)
    if bool(event.get("halted")):
        return "halted"
    if bool(event.get("warning")):
        return "warning"
    return "passed"


def _trace_score(event: dict[str, Any]) -> str:
    score = _first_present(
        event,
        ("coherence", "score", "last_score", "risk_score", "threshold"),
        "",
    )
    if isinstance(score, int | float):
        return f"{score:.3f}"
    return str(score)


def _trace_event_reason(event: dict[str, Any]) -> str:
    reason = _first_present(
        event,
        ("halt_reason", "reason", "halt_cause", "violation", "suggested_action"),
        "",
    )
    if reason:
        return str(reason)
    evidence = event.get("halt_evidence")
    if isinstance(evidence, dict):
        return str(_first_present(evidence, ("reason", "suggested_action"), ""))
    return ""


def _trace_detail(event: dict[str, Any]) -> str:
    details: list[str] = []
    attribution = _event_attribution(event)
    if attribution:
        token_offset = attribution.get("token_offset")
        scorer_path = attribution.get("scorer_path")
        retrieval_path = attribution.get("retrieval_path")
        if token_offset is not None:
            details.append(f"token={token_offset}")
        if scorer_path:
            details.append(f"scorer={scorer_path}")
        if retrieval_path:
            details.append(f"retrieval={retrieval_path}")

    counterfactual = _event_counterfactual(event)
    if counterfactual:
        fact_source = counterfactual.get("fact_source")
        delta = counterfactual.get("required_score_delta")
        if fact_source:
            details.append(f"fact={fact_source}")
        if delta is not None:
            details.append(f"delta={delta}")

    if not details:
        token = _first_present(event, ("token", "text", "value"), "")
        if token:
            details.append(str(token)[:80])
    return " | ".join(details)


def _trace_halted(document: Any, events: list[dict[str, Any]]) -> bool:
    if isinstance(document, dict) and bool(document.get("halted")):
        return True
    return any(
        bool(event.get("halted")) or _trace_state(event) in {"halted", "halt", "block"}
        for event in events
    )


def _trace_halt_reason(document: Any, events: list[dict[str, Any]]) -> str:
    if isinstance(document, dict):
        reason = _first_present(document, ("halt_reason", "reason", "halt_cause"), "")
        if reason:
            return str(reason)
    for event in events:
        reason = _trace_event_reason(event)
        if reason:
            return reason
    return ""


def _trace_counterfactual(
    document: Any,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    if isinstance(document, dict):
        counterfactual = _event_counterfactual(document)
        if counterfactual:
            return counterfactual
    for event in events:
        counterfactual = _event_counterfactual(event)
        if counterfactual:
            return counterfactual
    return {}


def _trace_attribution(document: Any, events: list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(document, dict):
        attribution = _event_attribution(document)
        if attribution:
            return attribution
    for event in events:
        attribution = _event_attribution(event)
        if attribution:
            return attribution
    return {}


def _event_counterfactual(event: dict[str, Any]) -> dict[str, Any]:
    diagnostic = event.get("counterfactual_diagnostic")
    if isinstance(diagnostic, dict):
        best_change = diagnostic.get("best_change")
        if isinstance(best_change, dict):
            return best_change
        return diagnostic
    evidence = event.get("halt_evidence_structured") or event.get("halt_evidence")
    if isinstance(evidence, dict):
        return _event_counterfactual(evidence)
    return {}


def _event_attribution(event: dict[str, Any]) -> dict[str, Any]:
    attribution = event.get("trace_attribution")
    if isinstance(attribution, dict):
        return attribution
    evidence = event.get("halt_evidence_structured") or event.get("halt_evidence")
    if isinstance(evidence, dict):
        return _event_attribution(evidence)
    return {}


def _first_present(
    mapping: dict[str, Any],
    keys: tuple[str, ...],
    default: Any,
) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return default
