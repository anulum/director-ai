# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — SOC 2 / ISO 27001 readiness

"""Tenant-safe SOC 2 / ISO/IEC 27001 readiness reporting."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast


class ReadinessStatus(StrEnum):
    """Readiness status for a control row."""

    PASS = "passed"  # nosec B105 - readiness status label, not a password.
    WARNING = "warning"
    FAIL = "failing"
    NOT_APPLICABLE = "not_applicable"


_SOC2_CRITERIA = frozenset(
    ("security", "availability", "processing_integrity", "confidentiality", "privacy")
)
_DISCLAIMER = (
    "Readiness evidence only; this is not a SOC 2 report, ISO/IEC 27001 "
    "certification, or auditor opinion."
)


def _status(control: Soc2IsoControl) -> ReadinessStatus:
    return cast(ReadinessStatus, control.status)


@dataclass(frozen=True)
class Soc2IsoControl:
    """One readiness control mapped to SOC 2 and ISO/IEC 27001 references."""

    control_id: str
    title: str
    soc2_criteria: tuple[str, ...]
    iso27001_refs: tuple[str, ...]
    status: ReadinessStatus | str
    evidence_refs: tuple[str, ...]
    owner: str = ""
    updated_at: str = ""
    notes: str = ""
    raw_evidence: str = ""

    def __post_init__(self) -> None:
        control_id = self.control_id.strip().upper()
        if not control_id or not control_id.replace("-", "").isalnum():
            raise ValueError("control_id must contain letters, numbers, or hyphen")
        if not self.title.strip():
            raise ValueError("title is required")
        try:
            status = ReadinessStatus(str(self.status).strip().lower())
        except ValueError as exc:
            raise ValueError(
                f"status must be one of {[item.value for item in ReadinessStatus]}"
            ) from exc

        soc2 = tuple(item.strip().lower() for item in self.soc2_criteria if item)
        invalid_soc2 = sorted(set(soc2) - _SOC2_CRITERIA)
        if invalid_soc2:
            raise ValueError(f"soc2 criteria are invalid: {invalid_soc2}")

        iso_refs = tuple(item.strip() for item in self.iso27001_refs if item)
        if not iso_refs or any(not item.startswith("A.") for item in iso_refs):
            raise ValueError("iso27001_refs must use Annex A references such as A.5.15")

        evidence_refs = tuple(item.strip() for item in self.evidence_refs if item)
        if not evidence_refs:
            raise ValueError("evidence_refs must be non-empty")

        object.__setattr__(self, "control_id", control_id)
        object.__setattr__(self, "title", self.title.strip())
        object.__setattr__(self, "soc2_criteria", soc2)
        object.__setattr__(self, "iso27001_refs", iso_refs)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "evidence_refs", evidence_refs)
        object.__setattr__(self, "owner", self.owner.strip())
        object.__setattr__(self, "updated_at", self.updated_at.strip())
        object.__setattr__(self, "notes", self.notes.strip())

    def to_dict(self) -> dict[str, Any]:
        """Return tenant-safe JSON-compatible control metadata."""

        return {
            "control_id": self.control_id,
            "title": self.title,
            "soc2_criteria": list(self.soc2_criteria),
            "iso27001_refs": list(self.iso27001_refs),
            "status": _status(self).value,
            "evidence_refs": list(self.evidence_refs),
            "owner": self.owner,
            "updated_at": self.updated_at,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class Soc2IsoReadinessReport:
    """Tenant-safe SOC 2 / ISO/IEC 27001 readiness report."""

    generated_at: str
    controls: tuple[Soc2IsoControl, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a tenant-safe JSON-compatible readiness payload."""

        return {
            "frameworks": ["SOC 2 Trust Services Criteria", "ISO/IEC 27001:2022"],
            "generated_at": self.generated_at,
            "summary": self.summary(),
            "controls": [control.to_dict() for control in self.controls],
            "disclaimer": _DISCLAIMER,
            "privacy": {
                "payload_classification": "tenant_safe",
                "raw_security_evidence_included": False,
                "certification_claimed": False,
            },
        }

    def summary(self) -> dict[str, int | float | str]:
        """Return aggregate readiness counts and risk level."""

        total = len(self.controls)
        passed = sum(
            1 for control in self.controls if _status(control) is ReadinessStatus.PASS
        )
        warnings = sum(
            1
            for control in self.controls
            if _status(control) is ReadinessStatus.WARNING
        )
        failures = sum(
            1 for control in self.controls if _status(control) is ReadinessStatus.FAIL
        )
        not_applicable = sum(
            1
            for control in self.controls
            if _status(control) is ReadinessStatus.NOT_APPLICABLE
        )
        applicable = max(total - not_applicable, 1)
        readiness_score = round(passed / applicable, 4)
        return {
            "total_controls": total,
            "passed": passed,
            "warnings": warnings,
            "failures": failures,
            "not_applicable": not_applicable,
            "readiness_score": readiness_score,
            "risk_level": _risk_level(failures=failures, warnings=warnings),
        }

    def to_markdown(self) -> str:
        """Render the readiness report as Markdown."""

        summary = self.summary()
        lines = [
            "# SOC 2 / ISO 27001 Readiness",
            "",
            f"Generated: {self.generated_at}",
            "",
            "## Summary",
        ]
        for key, value in summary.items():
            lines.append(f"- {key.replace('_', ' ').title()}: {value}")
        lines.extend(["", f"> {_DISCLAIMER}", "", "## Controls", ""])
        lines.extend(
            [
                "| ID | Control | Status | SOC 2 | ISO 27001 | Evidence |",
                "|---|---|---:|---|---|---|",
            ]
        )
        for control in self.controls:
            lines.append(
                "| "
                + " | ".join(
                    [
                        control.control_id,
                        control.title,
                        _status(control).value,
                        ", ".join(control.soc2_criteria),
                        ", ".join(control.iso27001_refs),
                        ", ".join(control.evidence_refs),
                    ]
                )
                + " |"
            )
        return "\n".join(lines)

    def to_trust_controls(self):
        """Convert readiness rows into Trust Console controls."""

        from director_ai.ui.safety_dashboard import TrustControl

        return [
            TrustControl(
                control=f"SOC2/ISO {control.control_id}: {control.title}",
                status=_status(control).value,
                evidence_ref=", ".join(control.evidence_refs),
                owner=control.owner,
                updated_at=control.updated_at,
            )
            for control in self.controls
        ]


def build_soc2_iso_readiness_report(
    *,
    controls: tuple[Soc2IsoControl, ...] | list[Soc2IsoControl] | None = None,
    generated_at: str = "",
) -> Soc2IsoReadinessReport:
    """Build a tenant-safe SOC 2 / ISO/IEC 27001 readiness report."""

    return Soc2IsoReadinessReport(
        generated_at=generated_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        controls=tuple(controls)
        if controls is not None
        else default_readiness_controls(),
    )


def default_readiness_controls() -> tuple[Soc2IsoControl, ...]:
    """Return Director-AI's default product-readiness control catalogue."""

    return (
        Soc2IsoControl(
            control_id="SEC-01",
            title="Tenant authentication and access isolation",
            soc2_criteria=("security", "confidentiality"),
            iso27001_refs=("A.5.15", "A.8.3"),
            status=ReadinessStatus.PASS,
            evidence_refs=("tests/test_server_auth.py", "tests/test_enterprise.py"),
        ),
        Soc2IsoControl(
            control_id="PRIV-01",
            title="PII redaction and tenant-safe audit metadata",
            soc2_criteria=("privacy", "confidentiality"),
            iso27001_refs=("A.8.11", "A.8.12"),
            status=ReadinessStatus.PASS,
            evidence_refs=("tests/test_pii_redactor.py", "docs/BENCHMARKS.md"),
        ),
        Soc2IsoControl(
            control_id="MON-01",
            title="Safety monitoring dashboard and alert evidence",
            soc2_criteria=("security", "availability"),
            iso27001_refs=("A.8.16",),
            status=ReadinessStatus.PASS,
            evidence_refs=("tests/test_safety_dashboard.py", "deploy/observability"),
        ),
        Soc2IsoControl(
            control_id="INC-01",
            title="Incident and human review audit trail",
            soc2_criteria=("security", "processing_integrity"),
            iso27001_refs=("A.5.24", "A.5.26"),
            status=ReadinessStatus.WARNING,
            evidence_refs=(
                "tests/test_human_review.py",
                "docs-site/api/human-review.md",
            ),
            notes="Operator incident-response ownership must be supplied per deployment.",
        ),
        Soc2IsoControl(
            control_id="VULN-01",
            title="Dependency and security scanning evidence",
            soc2_criteria=("security",),
            iso27001_refs=("A.8.8",),
            status=ReadinessStatus.WARNING,
            evidence_refs=("docs/internal/AUDIT_INDEX.md", ".github/workflows"),
            notes="External auditor evidence pack remains deployment-specific.",
        ),
        Soc2IsoControl(
            control_id="CHANGE-01",
            title="Change management and release verification",
            soc2_criteria=("security", "processing_integrity"),
            iso27001_refs=("A.8.32",),
            status=ReadinessStatus.WARNING,
            evidence_refs=("docs/PRODUCTION_CHECKLIST.md", ".githooks/pre-commit"),
            notes="Customer deployment change approvals are outside repository scope.",
        ),
    )


def _risk_level(*, failures: int, warnings: int) -> str:
    if failures:
        return "critical"
    if warnings:
        return "attention_required"
    return "ready"
