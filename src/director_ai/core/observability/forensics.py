# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — scorer-miss forensics
"""Tenant-safe scorer-miss forensics over guard eval records.

The KPI layer answers whether the guardrail is healthy in aggregate. This module
answers the operator's next question for reviewed decisions: which scorer missed,
why the miss is plausible from the emitted evidence metadata, what knowledge-base
state was visible, which model/version produced the answer, and what action is
worth taking next. Inputs are the same tenant-safe eval attributes emitted by
``director_ai.core.eval_trace`` plus reviewer labels; raw prompts, answers, and
retrieved chunk text are never required or returned.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ForensicsCase",
    "ForensicsReport",
    "build_forensics_report",
    "render_forensics_markdown",
    "render_forensics_text",
]

GROUNDED = "grounded"
HALLUCINATION = "hallucination"
CORRECT_ALLOW = "correct_allow"
CORRECT_HALT = "correct_halt"
FALSE_NEGATIVE = "false_negative"
FALSE_POSITIVE = "false_positive"
UNLABELLED_ALLOW = "unlabelled_allow"
UNLABELLED_HALT = "unlabelled_halt"


@dataclass(frozen=True)
class ForensicsCase:
    """One tenant-safe reviewed guard decision for operator forensics."""

    case_id: str
    outcome: str
    approved: bool
    expected_label: str
    score: float
    threshold: float
    margin: float
    scorer: str
    model: str
    model_revision: str
    domain: str
    knowledge_state: str
    evidence_count: int
    unsupported_claims: int
    reason: str
    recommended_action: str

    def to_dict(self) -> dict[str, str | int | float | bool]:
        """Return a JSON-compatible tenant-safe case payload."""
        return {
            "case_id": self.case_id,
            "outcome": self.outcome,
            "approved": self.approved,
            "expected_label": self.expected_label,
            "score": self.score,
            "threshold": self.threshold,
            "margin": self.margin,
            "scorer": self.scorer,
            "model": self.model,
            "model_revision": self.model_revision,
            "domain": self.domain,
            "knowledge_state": self.knowledge_state,
            "evidence_count": self.evidence_count,
            "unsupported_claims": self.unsupported_claims,
            "reason": self.reason,
            "recommended_action": self.recommended_action,
        }


@dataclass(frozen=True)
class ForensicsReport:
    """Tenant-safe scorer-miss report for a reviewed decision window."""

    total_records: int
    labelled_records: int
    misses_total: int
    false_negatives: int
    false_positives: int
    missed_by_scorer: dict[str, int]
    missed_by_model: dict[str, int]
    missed_by_domain: dict[str, int]
    cases: tuple[ForensicsCase, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report payload."""
        return {
            "total_records": self.total_records,
            "labelled_records": self.labelled_records,
            "misses_total": self.misses_total,
            "false_negatives": self.false_negatives,
            "false_positives": self.false_positives,
            "missed_by_scorer": dict(self.missed_by_scorer),
            "missed_by_model": dict(self.missed_by_model),
            "missed_by_domain": dict(self.missed_by_domain),
            "cases": [case.to_dict() for case in self.cases],
            "privacy": {
                "payload_classification": "tenant_safe",
                "raw_prompt_included": False,
                "raw_response_included": False,
                "raw_evidence_text_included": False,
            },
        }


def build_forensics_report(records: Sequence[Mapping[str, object]]) -> ForensicsReport:
    """Build a scorer-miss report from tenant-safe eval/reviewer records.

    Parameters
    ----------
    records:
        Eval-trace records or JSON objects containing at least approval, score,
        threshold, scorer/model metadata, and optionally a reviewer label. The
        function accepts both ``director.eval.*`` keys and plain aliases such as
        ``approved`` or ``label`` so exports can be joined without rewriting.
    """
    cases = tuple(
        _case_from_record(record, index) for index, record in enumerate(records)
    )
    labelled = [case for case in cases if case.expected_label]
    misses = [
        case for case in cases if case.outcome in {FALSE_NEGATIVE, FALSE_POSITIVE}
    ]
    return ForensicsReport(
        total_records=len(cases),
        labelled_records=len(labelled),
        misses_total=len(misses),
        false_negatives=sum(1 for case in misses if case.outcome == FALSE_NEGATIVE),
        false_positives=sum(1 for case in misses if case.outcome == FALSE_POSITIVE),
        missed_by_scorer=_counter_dict(case.scorer for case in misses),
        missed_by_model=_counter_dict(case.model for case in misses),
        missed_by_domain=_counter_dict(case.domain for case in misses),
        cases=cases,
    )


def render_forensics_markdown(report: ForensicsReport) -> str:
    """Render a Markdown scorer-miss report for operator review."""
    lines = [
        "# Guardrail Forensics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total records | {report.total_records} |",
        f"| Labelled records | {report.labelled_records} |",
        f"| Misses | {report.misses_total} |",
        f"| False negatives | {report.false_negatives} |",
        f"| False positives | {report.false_positives} |",
        "",
        "## Misses by scorer",
        "",
    ]
    lines.extend(_markdown_count_rows(report.missed_by_scorer))
    lines.extend(["", "## Reviewed cases", ""])
    if not report.cases:
        lines.append("No records supplied.")
        return "\n".join(lines)

    lines.append(
        "| Case | Outcome | Score | Threshold | Scorer | Model | KB state | Action |"
    )
    lines.append("|---|---|---:|---:|---|---|---|---|")
    for case in report.cases:
        lines.append(
            "| "
            + " | ".join(
                [
                    case.case_id,
                    case.outcome,
                    f"{case.score:.4f}",
                    f"{case.threshold:.4f}",
                    case.scorer,
                    case.model,
                    case.knowledge_state,
                    case.recommended_action,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_forensics_text(report: ForensicsReport) -> str:
    """Render a plain-text scorer-miss report for CLI output."""
    lines = [
        "Guardrail Forensics",
        f"  total_records: {report.total_records}",
        f"  labelled_records: {report.labelled_records}",
        f"  misses_total: {report.misses_total}",
        f"  false_negatives: {report.false_negatives}",
        f"  false_positives: {report.false_positives}",
    ]
    if report.missed_by_scorer:
        lines.append("  missed_by_scorer:")
        for scorer, count in report.missed_by_scorer.items():
            lines.append(f"    {scorer}: {count}")
    if report.cases:
        lines.append("  cases:")
        for case in report.cases:
            lines.append(
                "    "
                f"{case.case_id}: {case.outcome}; score={case.score:.4f}; "
                f"threshold={case.threshold:.4f}; scorer={case.scorer}; "
                f"action={case.recommended_action}"
            )
    return "\n".join(lines)


def _case_from_record(record: Mapping[str, object], index: int) -> ForensicsCase:
    case_id = _str_value(record, "director.eval.answer_id", "answer_id", "case_id")
    if not case_id:
        case_id = f"record-{index + 1}"
    approved = _approved(record)
    score = _float_value(record, "director.eval.score", "score")
    threshold = _float_value(record, "director.eval.threshold", "threshold")
    label = _label(record)
    evidence_count = _int_value(
        record, "director.eval.evidence_count", "evidence_count"
    )
    unsupported = _int_value(
        record, "director.eval.unsupported_claims", "unsupported_claims"
    )
    scorer = _str_value(record, "director.eval.scorer", "scorer") or "unknown"
    model = _str_value(record, "director.eval.model", "gen_ai.request.model", "model")
    model = model or "unknown"
    model_revision = _str_value(
        record, "director.eval.model_revision", "model_revision"
    )
    domain = _str_value(record, "director.eval.domain", "domain") or "unknown"
    knowledge_state = _knowledge_state(record, evidence_count, unsupported)
    outcome = _outcome(approved=approved, label=label)
    reason = _reason(
        outcome=outcome,
        score=score,
        threshold=threshold,
        evidence_count=evidence_count,
        unsupported_claims=unsupported,
        scorer=scorer,
    )
    action = _recommended_action(
        outcome=outcome,
        knowledge_state=knowledge_state,
        unsupported_claims=unsupported,
    )
    return ForensicsCase(
        case_id=case_id,
        outcome=outcome,
        approved=approved,
        expected_label=label,
        score=round(score, 4),
        threshold=round(threshold, 4),
        margin=round(score - threshold, 4),
        scorer=scorer,
        model=model,
        model_revision=model_revision,
        domain=domain,
        knowledge_state=knowledge_state,
        evidence_count=evidence_count,
        unsupported_claims=unsupported,
        reason=reason,
        recommended_action=action,
    )


def _approved(record: Mapping[str, object]) -> bool:
    value = _first(record, "director.eval.approved", "approved", "guard_approved")
    if isinstance(value, bool):
        return value
    decision = _str_value(record, "director.eval.decision", "decision").lower()
    if decision in {"allow", "approved"}:
        return True
    if decision in {"halt", "blocked", "rejected"}:
        return False
    raise ValueError("record must include approved/guard_approved or a decision")


def _label(record: Mapping[str, object]) -> str:
    value = _str_value(record, "label", "expected_label", "review_label").lower()
    if value in {"grounded", "correct", "true"}:
        return GROUNDED
    if value in {"hallucination", "hallucinated", "incorrect", "false"}:
        return HALLUCINATION
    return ""


def _outcome(*, approved: bool, label: str) -> str:
    if label == HALLUCINATION and approved:
        return FALSE_NEGATIVE
    if label == HALLUCINATION and not approved:
        return CORRECT_HALT
    if label == GROUNDED and approved:
        return CORRECT_ALLOW
    if label == GROUNDED and not approved:
        return FALSE_POSITIVE
    return UNLABELLED_ALLOW if approved else UNLABELLED_HALT


def _knowledge_state(
    record: Mapping[str, object], evidence_count: int, unsupported_claims: int
) -> str:
    version = _str_value(
        record,
        "director.eval.kb_version",
        "director.eval.knowledge_version",
        "kb_version",
        "knowledge_version",
    )
    prefix = f"kb:{version}" if version else "kb:unversioned"
    if evidence_count <= 0:
        return f"{prefix}:no_evidence"
    if unsupported_claims > 0:
        return f"{prefix}:unsupported_claims"
    return f"{prefix}:evidence_present"


def _reason(
    *,
    outcome: str,
    score: float,
    threshold: float,
    evidence_count: int,
    unsupported_claims: int,
    scorer: str,
) -> str:
    if outcome == FALSE_NEGATIVE:
        if evidence_count <= 0:
            return f"{scorer} approved above threshold with no evidence attached"
        if unsupported_claims == 0:
            return f"{scorer} approved above threshold without unsupported claims"
        return f"{scorer} approved despite unsupported-claim metadata"
    if outcome == FALSE_POSITIVE:
        if unsupported_claims > 0:
            return f"{scorer} halted because unsupported claims were present"
        return f"{scorer} halted below threshold without reviewer-confirmed risk"
    if score < threshold:
        return f"{scorer} halted because score was below threshold"
    return f"{scorer} allowed because score met threshold"


def _recommended_action(
    *, outcome: str, knowledge_state: str, unsupported_claims: int
) -> str:
    if outcome == FALSE_NEGATIVE:
        if knowledge_state.endswith(":no_evidence"):
            return "refresh_or_add_governed_facts"
        if unsupported_claims == 0:
            return "add_counterexample_and_recalibrate_scorer"
        return "inspect_claim_attribution_thresholds"
    if outcome == FALSE_POSITIVE:
        if unsupported_claims > 0:
            return "review_retrieval_source_mapping"
        return "lower_false_halt_pressure_or_raise_review_queue"
    return "no_operator_action"


def _counter_dict(values: Iterable[str]) -> dict[str, int]:
    counter = Counter(value or "unknown" for value in values)
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _markdown_count_rows(counts: Mapping[str, int]) -> list[str]:
    if not counts:
        return ["No scorer misses in the labelled window."]
    rows = ["| Scorer | Misses |", "|---|---:|"]
    rows.extend(f"| {key} | {value} |" for key, value in counts.items())
    return rows


def _first(record: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _str_value(record: Mapping[str, object], *keys: str) -> str:
    value = _first(record, *keys)
    return value.strip() if isinstance(value, str) else ""


def _float_value(record: Mapping[str, object], *keys: str) -> float:
    value = _first(record, *keys)
    if isinstance(value, bool):
        raise ValueError(f"{keys[0]} must be numeric")
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"{keys[0]} must be numeric")


def _int_value(record: Mapping[str, object], *keys: str) -> int:
    value = _first(record, *keys)
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError(f"{keys[0]} must be an integer")
    if isinstance(value, int):
        return int(value)
    raise ValueError(f"{keys[0]} must be an integer")
