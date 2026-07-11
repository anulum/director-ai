# SPDX-License-Identifier: BUSL-1.1
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - safety dashboard report models

"""Tenant-safe report models for the safety operations dashboard.

Column layouts, status vocabularies, and the frozen record/report
dataclasses (halt records, trust controls, compliance-export references,
the Trust Console report, and the observability operations report) with
their JSON and Markdown serialisations. The analytics that populate
these models live in :mod:`._dashboard_analytics`; the builders and the
Gradio app live in :mod:`.safety_dashboard` and :mod:`._dashboard_app`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "COMPLIANCE_EXPORT_COLUMNS",
    "DRIFT_ALERT_COLUMNS",
    "EVIDENCE_COLUMNS",
    "SOURCE_COLUMNS",
    "TENANT_COLUMNS",
    "ComplianceExportRef",
    "ComplianceExportStatus",
    "HaltDashboardRecord",
    "ObservabilityOperationsReport",
    "TrustConsoleReport",
    "TrustControl",
    "TrustControlStatus",
]

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
DRIFT_ALERT_COLUMNS = [
    "tenant_id",
    "baseline_events",
    "current_events",
    "baseline_halt_rate",
    "current_halt_rate",
    "rate_change",
    "severity",
    "recommendation",
]
COMPLIANCE_EXPORT_COLUMNS = [
    "standard",
    "name",
    "status",
    "evidence_ref",
    "updated_at",
]
TrustControlStatus = Literal["passed", "warning", "failing", "not_applicable"]
_VALID_TRUST_CONTROL_STATUSES = frozenset(
    ("passed", "warning", "failing", "not_applicable")
)
ComplianceExportStatus = Literal["available", "missing", "stale", "not_applicable"]
_VALID_COMPLIANCE_EXPORT_STATUSES = frozenset(
    ("available", "missing", "stale", "not_applicable")
)


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


@dataclass(frozen=True)
class TrustControl:
    """Tenant-safe readiness control shown in the Trust Console."""

    control: str
    status: str
    evidence_ref: str
    owner: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        """Normalise and validate the trust-control status."""
        status = self.status.strip().lower()
        if status not in _VALID_TRUST_CONTROL_STATUSES:
            raise ValueError(
                "TrustControl.status must be one of "
                f"{sorted(_VALID_TRUST_CONTROL_STATUSES)}"
            )
        if not self.control.strip():
            raise ValueError("TrustControl.control is required")
        if not self.evidence_ref.strip():
            raise ValueError("TrustControl.evidence_ref is required")
        object.__setattr__(self, "control", self.control.strip())
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "evidence_ref", self.evidence_ref.strip())
        object.__setattr__(self, "owner", self.owner.strip())
        object.__setattr__(self, "updated_at", self.updated_at.strip())

    def to_dict(self) -> dict[str, str]:
        """Return JSON-compatible tenant-safe control metadata."""
        return {
            "control": self.control,
            "status": self.status,
            "evidence_ref": self.evidence_ref,
            "owner": self.owner,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class ComplianceExportRef:
    """Tenant-safe reference to an operator-owned compliance export."""

    standard: str
    name: str
    status: str
    evidence_ref: str
    updated_at: str = ""

    def __post_init__(self) -> None:
        """Normalise and validate the compliance-export status."""
        status = self.status.strip().lower()
        if status not in _VALID_COMPLIANCE_EXPORT_STATUSES:
            raise ValueError(
                "ComplianceExportRef.status must be one of "
                f"{sorted(_VALID_COMPLIANCE_EXPORT_STATUSES)}"
            )
        if not self.standard.strip():
            raise ValueError("ComplianceExportRef.standard is required")
        if not self.name.strip():
            raise ValueError("ComplianceExportRef.name is required")
        if not self.evidence_ref.strip():
            raise ValueError("ComplianceExportRef.evidence_ref is required")
        object.__setattr__(self, "standard", self.standard.strip())
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "evidence_ref", self.evidence_ref.strip())
        object.__setattr__(self, "updated_at", self.updated_at.strip())

    def to_dict(self) -> dict[str, str]:
        """Return JSON-compatible tenant-safe export metadata."""
        return {
            "standard": self.standard,
            "name": self.name,
            "status": self.status,
            "evidence_ref": self.evidence_ref,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class TrustConsoleReport:
    """Tenant-safe Trust Console report for customer-facing review."""

    title: str
    generated_at: str
    summary: dict[str, Any]
    tenants: list[list[Any]]
    recent_evidence: list[list[Any]]
    controls: tuple[TrustControl, ...]
    parse_warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a tenant-safe JSON-compatible report payload."""
        return {
            "title": self.title,
            "generated_at": self.generated_at,
            "summary": dict(self.summary),
            "tenant_columns": list(TENANT_COLUMNS),
            "tenants": self.tenants,
            "evidence_columns": list(EVIDENCE_COLUMNS),
            "recent_evidence": self.recent_evidence,
            "controls": [control.to_dict() for control in self.controls],
            "parse_warnings": list(self.parse_warnings),
            "privacy": {
                "payload_classification": "tenant_safe",
                "raw_event_text_included": False,
                "raw_feedback_text_included": False,
            },
        }

    def to_markdown(self) -> str:
        """Render the Trust Console report as Markdown."""
        lines = [
            f"# {self.title}",
            "",
            f"Generated: {self.generated_at}",
            "",
            "## Summary",
        ]
        for key, value in self.summary.items():
            label = key.replace("_", " ").title()
            lines.append(f"- {label}: {value}")
        if self.parse_warnings:
            lines.append("- Parse Warnings: " + "; ".join(self.parse_warnings[:5]))

        lines.extend(["", "## Readiness Controls", ""])
        if self.controls:
            lines.extend(
                [
                    "| Control | Status | Evidence | Owner | Updated |",
                    "|---|---:|---|---|---|",
                ]
            )
            for control in self.controls:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            control.control,
                            control.status,
                            control.evidence_ref,
                            control.owner,
                            control.updated_at,
                        ]
                    )
                    + " |"
                )
        else:
            lines.append("No readiness controls supplied.")

        lines.extend(["", "## Tenant Operations", ""])
        if self.tenants:
            lines.append("| " + " | ".join(TENANT_COLUMNS) + " |")
            lines.append("|" + "|".join("---" for _ in TENANT_COLUMNS) + "|")
            for row in self.tenants:
                lines.append("| " + " | ".join(str(value) for value in row) + " |")
        else:
            lines.append("No tenant events supplied.")
        return "\n".join(lines)


@dataclass(frozen=True)
class ObservabilityOperationsReport:
    """Tenant-safe report for halt forensics, drift, and compliance review."""

    title: str
    generated_at: str
    summary: dict[str, Any]
    tenants: list[list[Any]]
    sources: list[list[Any]]
    recent_evidence: list[list[Any]]
    drift_alerts: list[list[Any]]
    controls: tuple[TrustControl, ...]
    compliance_exports: tuple[ComplianceExportRef, ...]
    parse_warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a tenant-safe JSON-compatible operations packet."""
        return {
            "title": self.title,
            "generated_at": self.generated_at,
            "summary": dict(self.summary),
            "tenant_columns": list(TENANT_COLUMNS),
            "tenants": self.tenants,
            "source_columns": list(SOURCE_COLUMNS),
            "sources": self.sources,
            "evidence_columns": list(EVIDENCE_COLUMNS),
            "recent_evidence": self.recent_evidence,
            "drift_alert_columns": list(DRIFT_ALERT_COLUMNS),
            "drift_alerts": self.drift_alerts,
            "controls": [control.to_dict() for control in self.controls],
            "compliance_export_columns": list(COMPLIANCE_EXPORT_COLUMNS),
            "compliance_exports": [
                export.to_dict() for export in self.compliance_exports
            ],
            "parse_warnings": list(self.parse_warnings),
            "privacy": {
                "payload_classification": "tenant_safe",
                "raw_event_text_included": False,
                "raw_feedback_text_included": False,
                "raw_compliance_evidence_included": False,
            },
        }

    def to_markdown(self) -> str:
        """Render the operations report as Markdown."""
        lines = [
            f"# {self.title}",
            "",
            f"Generated: {self.generated_at}",
            "",
            "## Summary",
        ]
        for key, value in self.summary.items():
            lines.append(f"- {key.replace('_', ' ').title()}: {value}")
        if self.parse_warnings:
            lines.append("- Parse Warnings: " + "; ".join(self.parse_warnings[:5]))

        lines.extend(["", "## Drift Alerts", ""])
        if self.drift_alerts:
            lines.append("| " + " | ".join(DRIFT_ALERT_COLUMNS) + " |")
            lines.append("|" + "|".join("---" for _ in DRIFT_ALERT_COLUMNS) + "|")
            for row in self.drift_alerts:
                lines.append("| " + " | ".join(str(value) for value in row) + " |")
        else:
            lines.append("No drift alerts for the configured threshold.")

        lines.extend(["", "## Compliance Exports", ""])
        if self.compliance_exports:
            lines.append("| " + " | ".join(COMPLIANCE_EXPORT_COLUMNS) + " |")
            lines.append("|" + "|".join("---" for _ in COMPLIANCE_EXPORT_COLUMNS) + "|")
            for export in self.compliance_exports:
                export_row = export.to_dict()
                lines.append(
                    "| "
                    + " | ".join(
                        str(export_row[column]) for column in COMPLIANCE_EXPORT_COLUMNS
                    )
                    + " |"
                )
        else:
            lines.append("No compliance export references supplied.")

        lines.extend(["", "## Recent Halt Evidence", ""])
        if self.recent_evidence:
            lines.append("| " + " | ".join(EVIDENCE_COLUMNS) + " |")
            lines.append("|" + "|".join("---" for _ in EVIDENCE_COLUMNS) + "|")
            for row in self.recent_evidence[:10]:
                lines.append("| " + " | ".join(str(value) for value in row) + " |")
        else:
            lines.append("No halt evidence supplied.")
        return "\n".join(lines)
