# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory dataset contract

"""Validation contract for customer-owned guardrail training traces."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..mandatory import mandatory_execution

try:  # pragma: no cover - optional acceleration
    from backfire_kernel import rust_sum_i64

    _RUST_DATASET_CONTRACT = True
except ImportError:  # pragma: no cover - mandatory accelerator guard
    _RUST_DATASET_CONTRACT = True

    def rust_sum_i64(_values: list[int]) -> int:
        raise RuntimeError("backfire_kernel rust_sum_i64 is unavailable")


SCHEMA_VERSION = "1.0.0"

REQUIRED_FIELDS = frozenset(
    {
        "trace_id",
        "customer_id",
        "tenant_id",
        "split",
        "prompt",
        "response",
        "expected_decision",
        "severity",
        "label",
        "source_refs",
        "policy_refs",
        "reviewer_role",
        "observed_at",
    }
)
ALLOWED_DECISIONS = frozenset({"approve", "block", "abstain", "escalate"})
ALLOWED_SEVERITIES = frozenset({"low", "medium", "high", "critical"})
HIGH_RISK_SEVERITIES = frozenset({"high", "critical"})
SECTOR_REQUIRED_METADATA = frozenset(
    {"sector_class", "knowledge_class", "requires_citation", "jurisdiction"}
)


@dataclass(frozen=True)
class CustomerWorkspace:
    """Customer workspace metadata used as the isolation boundary."""

    customer_id: str
    workspace_id: str
    tenant_id: str
    data_classification: str
    allowed_splits: tuple[str, ...]
    regulation_mappings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CustomerTraceFinding:
    """One validation finding for a customer trace dataset."""

    code: str
    severity: str
    message: str
    trace_id: str = ""
    field: str = ""

    def to_dict(self) -> dict[str, str]:
        """Serialise the finding to a stable dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class CustomerDatasetValidationReport:
    """Validation report and manifest preflight for customer datasets."""

    schema_version: str
    ready: bool
    customer_id: str
    workspace_id: str
    tenant_id: str
    row_count: int
    split_counts: dict[str, int]
    severity_counts: dict[str, int]
    high_risk_count: int
    dataset_hash: str
    findings: tuple[CustomerTraceFinding, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a JSON-safe manifest shape."""

        return {
            "schema_version": self.schema_version,
            "ready": self.ready,
            "customer_id": self.customer_id,
            "workspace_id": self.workspace_id,
            "tenant_id": self.tenant_id,
            "row_count": self.row_count,
            "split_counts": dict(sorted(self.split_counts.items())),
            "severity_counts": dict(sorted(self.severity_counts.items())),
            "high_risk_count": self.high_risk_count,
            "dataset_hash": self.dataset_hash,
            "findings": [finding.to_dict() for finding in self.findings],
        }

    def write_json(self, path: Path) -> Path:
        """Write the validation report as deterministic JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


def validate_customer_trace_dataset(
    rows: list[dict[str, Any]],
    workspace: CustomerWorkspace,
    *,
    vertical_profile: str | None = None,
) -> CustomerDatasetValidationReport:
    """Validate customer-owned guardrail traces before training or benchmarking."""

    findings: list[CustomerTraceFinding] = []
    split_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    fingerprints_by_split: dict[str, set[str]] = defaultdict(set)
    trace_ids: set[str] = set()

    for index, row in enumerate(rows):
        trace_id = _string(row.get("trace_id")) or f"row:{index}"
        _validate_required_fields(row, trace_id, findings)
        _validate_workspace_binding(row, workspace, trace_id, findings)
        _validate_trace_id(trace_id, trace_ids, findings)
        _validate_decision(row, trace_id, findings)
        _validate_refs(row, trace_id, findings)
        _validate_secret_redaction(row, trace_id, findings)
        _validate_sector_metadata(row, trace_id, vertical_profile, findings)

        split = _string(row.get("split"))
        if split:
            if split not in workspace.allowed_splits:
                findings.append(
                    CustomerTraceFinding(
                        code="split_not_allowed",
                        severity="error",
                        message=f"split {split!r} is not allowed for workspace",
                        trace_id=trace_id,
                        field="split",
                    )
                )
            split_counts[split] += 1
            fingerprints_by_split[split].add(_semantic_fingerprint(row))

        severity = _string(row.get("severity"))
        if severity:
            if severity not in ALLOWED_SEVERITIES:
                findings.append(
                    CustomerTraceFinding(
                        code="severity_not_allowed",
                        severity="error",
                        message=f"severity {severity!r} is not allowed",
                        trace_id=trace_id,
                        field="severity",
                    )
                )
            severity_counts[severity] += 1

    _validate_required_splits(workspace, split_counts, findings)
    _validate_cross_split_leakage(fingerprints_by_split, findings)
    high_risk_count = _sum_int(
        [severity_counts[severity] for severity in HIGH_RISK_SEVERITIES]
    )
    ready = not any(finding.severity == "error" for finding in findings)
    return CustomerDatasetValidationReport(
        schema_version=SCHEMA_VERSION,
        ready=ready,
        customer_id=workspace.customer_id,
        workspace_id=workspace.workspace_id,
        tenant_id=workspace.tenant_id,
        row_count=len(rows),
        split_counts=dict(sorted(split_counts.items())),
        severity_counts=dict(sorted(severity_counts.items())),
        high_risk_count=high_risk_count,
        dataset_hash=_dataset_hash(rows),
        findings=tuple(findings),
    )


def _validate_required_fields(
    row: dict[str, Any],
    trace_id: str,
    findings: list[CustomerTraceFinding],
) -> None:
    missing = sorted(field for field in REQUIRED_FIELDS if field not in row)
    for field in missing:
        findings.append(
            CustomerTraceFinding(
                code="missing_required_field",
                severity="error",
                message=f"required field {field!r} is missing",
                trace_id=trace_id,
                field=field,
            )
        )


def _validate_workspace_binding(
    row: dict[str, Any],
    workspace: CustomerWorkspace,
    trace_id: str,
    findings: list[CustomerTraceFinding],
) -> None:
    if row.get("customer_id") != workspace.customer_id:
        findings.append(
            CustomerTraceFinding(
                code="customer_id_mismatch",
                severity="error",
                message="row customer_id does not match workspace customer_id",
                trace_id=trace_id,
                field="customer_id",
            )
        )
    if row.get("tenant_id") != workspace.tenant_id:
        findings.append(
            CustomerTraceFinding(
                code="tenant_id_mismatch",
                severity="error",
                message="row tenant_id does not match workspace tenant_id",
                trace_id=trace_id,
                field="tenant_id",
            )
        )


def _validate_trace_id(
    trace_id: str,
    seen: set[str],
    findings: list[CustomerTraceFinding],
) -> None:
    if trace_id in seen:
        findings.append(
            CustomerTraceFinding(
                code="duplicate_trace_id",
                severity="error",
                message="trace_id must be unique inside a customer dataset",
                trace_id=trace_id,
                field="trace_id",
            )
        )
    seen.add(trace_id)


def _validate_decision(
    row: dict[str, Any],
    trace_id: str,
    findings: list[CustomerTraceFinding],
) -> None:
    decision = _string(row.get("expected_decision"))
    if decision and decision not in ALLOWED_DECISIONS:
        findings.append(
            CustomerTraceFinding(
                code="decision_not_allowed",
                severity="error",
                message=f"expected_decision {decision!r} is not allowed",
                trace_id=trace_id,
                field="expected_decision",
            )
        )
    severity = _string(row.get("severity"))
    if severity in HIGH_RISK_SEVERITIES and decision == "approve":
        findings.append(
            CustomerTraceFinding(
                code="high_risk_approve_label",
                severity="warning",
                message="high-risk rows approved by label need reviewer justification",
                trace_id=trace_id,
                field="expected_decision",
            )
        )


def _validate_refs(
    row: dict[str, Any],
    trace_id: str,
    findings: list[CustomerTraceFinding],
) -> None:
    for field in ("source_refs", "policy_refs"):
        value = row.get(field)
        if field in row and not _non_empty_string_list(value):
            findings.append(
                CustomerTraceFinding(
                    code="invalid_reference_list",
                    severity="error",
                    message=f"{field} must be a non-empty list of strings",
                    trace_id=trace_id,
                    field=field,
                )
            )


def _validate_secret_redaction(
    row: dict[str, Any],
    trace_id: str,
    findings: list[CustomerTraceFinding],
) -> None:
    if row.get("contains_secret") is True and row.get("redaction_status") != "redacted":
        findings.append(
            CustomerTraceFinding(
                code="unredacted_secret",
                severity="error",
                message="rows flagged with secrets must be redacted before training",
                trace_id=trace_id,
                field="redaction_status",
            )
        )


def _validate_sector_metadata(
    row: dict[str, Any],
    trace_id: str,
    vertical_profile: str | None,
    findings: list[CustomerTraceFinding],
) -> None:
    if not vertical_profile:
        return
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        missing = set(SECTOR_REQUIRED_METADATA)
    else:
        missing = {
            field
            for field in SECTOR_REQUIRED_METADATA
            if field not in metadata or metadata[field] in ("", None)
        }
    for field in sorted(missing):
        findings.append(
            CustomerTraceFinding(
                code="sector_metadata_missing",
                severity="error",
                message=f"sector profile requires metadata.{field}",
                trace_id=trace_id,
                field=f"metadata.{field}",
            )
        )
    if missing or not isinstance(metadata, dict):
        return

    from .sector_extension import validate_sector_trace_metadata

    findings.extend(
        validate_sector_trace_metadata(
            metadata,
            trace_id=trace_id,
            expected_decision=_string(row.get("expected_decision")),
        )
    )


def _validate_required_splits(
    workspace: CustomerWorkspace,
    split_counts: Counter[str],
    findings: list[CustomerTraceFinding],
) -> None:
    for split in workspace.allowed_splits:
        if split_counts[split] == 0:
            findings.append(
                CustomerTraceFinding(
                    code="split_missing",
                    severity="error",
                    message=f"required split {split!r} has no rows",
                    field="split",
                )
            )


def _validate_cross_split_leakage(
    fingerprints_by_split: dict[str, set[str]],
    findings: list[CustomerTraceFinding],
) -> None:
    splits = sorted(fingerprints_by_split)
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1 :]:
            overlap = fingerprints_by_split[left] & fingerprints_by_split[right]
            if overlap:
                findings.append(
                    CustomerTraceFinding(
                        code="cross_split_duplicate",
                        severity="error",
                        message=f"exact prompt/response duplicate across {left} and {right}",
                        field="split",
                    )
                )


def _dataset_hash(rows: list[dict[str, Any]]) -> str:
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _semantic_fingerprint(row: dict[str, Any]) -> str:
    prompt = _string(row.get("prompt")).strip().lower()
    response = _string(row.get("response")).strip().lower()
    return hashlib.sha256(f"{prompt}\n{response}".encode()).hexdigest()


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _non_empty_string_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )


def _sum_int(values: list[int]) -> int:
    if _RUST_DATASET_CONTRACT:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            return int(rust_sum_i64(values))
    return sum(values)
