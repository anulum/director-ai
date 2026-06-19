# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — SOC 2 / ISO 27001 / HIPAA readiness

"""Tenant-safe compliance readiness reporting.

The module builds operator-reviewable evidence packets for customer security
reviews. It deliberately records evidence references and deployment obligations
without serialising raw audit rows, customer text, PHI, secrets, or screenshots.
The reports are readiness aids only; they are not certifications, legal advice,
auditor opinions, or proof that a deployment satisfies a regulation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from director_ai.ui.safety_dashboard import TrustControl


class ReadinessStatus(StrEnum):
    """Readiness status for a control row."""

    # Readiness status label, not a password or credential.
    PASS = "passed"  # nosec B105
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
_HIPAA_DISCLAIMER = (
    "HIPAA documentation readiness evidence only; this is not legal advice, a "
    "business associate agreement, OCR determination, or proof that a deployment "
    "is HIPAA compliant."
)
_HIPAA_REF_PREFIX = "45 CFR 164."
_DEFAULT_PHI_SUMMARY = (
    "Director-AI can redact configured identifiers, avoid raw prompt/response "
    "text in default audit exports, bind API keys to tenants, and emit "
    "tenant-safe evidence references. The deployment operator remains "
    "responsible for determining whether ePHI is processed, executing required "
    "agreements, configuring storage/network controls, training workforce users, "
    "and preserving breach-response procedures."
)
_SOC2_TYPE_I_PATH = (
    "Define system boundaries, in-scope services, data flows, sub-processors, "
    "and production environments.",
    "Assign owners for security, availability, processing-integrity, "
    "confidentiality, privacy, incident response, vulnerability management, "
    "change management, and vendor management.",
    "Attach dated evidence references for every readiness control, including "
    "CI/security scans, access reviews, audit-log verification, deployment "
    "runbooks, monitoring, backup, and incident procedures.",
    "Review all warning or failing controls, either remediate them or document "
    "auditor-approved compensating controls.",
    "Freeze the Type I observation point only after an independent auditor or "
    "authorised internal exception confirms the evidence packet is complete.",
)


def _status(control: Soc2IsoControl) -> ReadinessStatus:
    return cast(ReadinessStatus, control.status)


@dataclass(frozen=True)
class Soc2IsoControl:
    """One readiness control mapped to SOC 2, ISO/IEC 27001, and HIPAA.

    Parameters
    ----------
    control_id:
        Stable uppercase identifier used in readiness reports and trust-console
        rows.
    title:
        Human-readable control statement.
    soc2_criteria:
        SOC 2 Trust Services Criteria categories touched by the control.
    iso27001_refs:
        ISO/IEC 27001:2022 Annex A-style references, such as ``A.5.15``.
    status:
        Current readiness status. String values are normalised into
        :class:`ReadinessStatus`.
    evidence_refs:
        Tenant-safe pointers to tests, docs, tickets, dashboards, or evidence
        stores. Raw evidence is intentionally excluded from serialisation.
    hipaa_security_refs:
        Optional HIPAA Security Rule references from 45 CFR Part 164 Subpart C.

    Notes
    -----
    The class stores references, not compliance claims. HIPAA applicability and
    control sufficiency remain deployment- and counsel-owned decisions.
    """

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
    hipaa_security_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalise and validate the control id, status, and framework references."""
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

        hipaa_refs = tuple(item.strip() for item in self.hipaa_security_refs if item)
        if hipaa_refs and any(
            not item.startswith(_HIPAA_REF_PREFIX) for item in hipaa_refs
        ):
            raise ValueError("hipaa_security_refs must use 45 CFR Part 164 references")

        object.__setattr__(self, "control_id", control_id)
        object.__setattr__(self, "title", self.title.strip())
        object.__setattr__(self, "soc2_criteria", soc2)
        object.__setattr__(self, "iso27001_refs", iso_refs)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "evidence_refs", evidence_refs)
        object.__setattr__(self, "owner", self.owner.strip())
        object.__setattr__(self, "updated_at", self.updated_at.strip())
        object.__setattr__(self, "notes", self.notes.strip())
        object.__setattr__(self, "hipaa_security_refs", hipaa_refs)

    def to_dict(self) -> dict[str, Any]:
        """Return tenant-safe JSON-compatible control metadata.

        Returns
        -------
        dict[str, Any]
            Control metadata with evidence references and framework mappings.
            ``raw_evidence`` is never included.
        """
        return {
            "control_id": self.control_id,
            "title": self.title,
            "soc2_criteria": list(self.soc2_criteria),
            "iso27001_refs": list(self.iso27001_refs),
            "hipaa_security_refs": list(self.hipaa_security_refs),
            "status": _status(self).value,
            "evidence_refs": list(self.evidence_refs),
            "owner": self.owner,
            "updated_at": self.updated_at,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class Soc2IsoReadinessReport:
    """Tenant-safe SOC 2 / ISO/IEC 27001 readiness report.

    Parameters
    ----------
    generated_at:
        UTC timestamp for the report payload.
    controls:
        Readiness controls included in the report.
    """

    generated_at: str
    controls: tuple[Soc2IsoControl, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a tenant-safe JSON-compatible readiness payload."""
        return {
            "frameworks": [
                "SOC 2 Trust Services Criteria",
                "ISO/IEC 27001:2022",
                "HIPAA Security Rule readiness crosswalk",
            ],
            "generated_at": self.generated_at,
            "summary": self.summary(),
            "soc2_type_i_path": list(_SOC2_TYPE_I_PATH),
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
        """Render the readiness report as Markdown.

        Returns
        -------
        str
            Markdown table suitable for a tenant-safe security review packet.
        """
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
        lines.extend(["", "## SOC 2 Type I Path", ""])
        lines.extend(f"{idx}. {step}" for idx, step in enumerate(_SOC2_TYPE_I_PATH, 1))
        lines.extend(["", f"> {_DISCLAIMER}", "", "## Controls", ""])
        lines.extend(
            [
                "| ID | Control | Status | SOC 2 | ISO 27001 | HIPAA Security Rule | Evidence |",
                "|---|---|---:|---|---|---|---|",
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
                        ", ".join(control.hipaa_security_refs),
                        ", ".join(control.evidence_refs),
                    ]
                )
                + " |"
            )
        return "\n".join(lines)

    def to_trust_controls(self) -> list[TrustControl]:
        """Convert readiness rows into Trust Console controls.

        Returns
        -------
        list[director_ai.ui.safety_dashboard.TrustControl]
            Tenant-safe controls for dashboard and procurement exports.
        """
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


@dataclass(frozen=True)
class HipaaDeploymentObligation:
    """One operator-owned HIPAA documentation obligation.

    Parameters
    ----------
    obligation_id:
        Stable uppercase identifier for the obligation row.
    title:
        Obligation statement to review during deployment approval.
    hipaa_security_refs:
        HIPAA Security Rule references from 45 CFR Part 164 Subpart C.
    status:
        Current readiness status.
    evidence_refs:
        Tenant-safe evidence references. Raw PHI and raw security evidence must
        stay in the operator-controlled evidence store.
    operator_action:
        Concrete action the deployment operator must complete or attest to.
    """

    obligation_id: str
    title: str
    hipaa_security_refs: tuple[str, ...]
    status: ReadinessStatus | str
    evidence_refs: tuple[str, ...]
    operator_action: str
    owner: str = ""
    updated_at: str = ""
    notes: str = ""
    raw_evidence: str = ""

    def __post_init__(self) -> None:
        """Normalise and validate the obligation id and HIPAA references."""
        obligation_id = self.obligation_id.strip().upper()
        if not obligation_id or not obligation_id.replace("-", "").isalnum():
            raise ValueError("obligation_id must contain letters, numbers, or hyphen")
        if not self.title.strip():
            raise ValueError("title is required")
        try:
            status = ReadinessStatus(str(self.status).strip().lower())
        except ValueError as exc:
            raise ValueError(
                f"status must be one of {[item.value for item in ReadinessStatus]}"
            ) from exc

        hipaa_refs = tuple(item.strip() for item in self.hipaa_security_refs if item)
        if not hipaa_refs or any(
            not item.startswith(_HIPAA_REF_PREFIX) for item in hipaa_refs
        ):
            raise ValueError("hipaa_security_refs must use 45 CFR Part 164 references")

        evidence_refs = tuple(item.strip() for item in self.evidence_refs if item)
        if not evidence_refs:
            raise ValueError("evidence_refs must be non-empty")
        operator_action = self.operator_action.strip()
        if not operator_action:
            raise ValueError("operator_action is required")

        object.__setattr__(self, "obligation_id", obligation_id)
        object.__setattr__(self, "title", self.title.strip())
        object.__setattr__(self, "hipaa_security_refs", hipaa_refs)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "evidence_refs", evidence_refs)
        object.__setattr__(self, "operator_action", operator_action)
        object.__setattr__(self, "owner", self.owner.strip())
        object.__setattr__(self, "updated_at", self.updated_at.strip())
        object.__setattr__(self, "notes", self.notes.strip())

    def to_dict(self) -> dict[str, Any]:
        """Return tenant-safe obligation metadata.

        Returns
        -------
        dict[str, Any]
            Obligation row with references and operator action. ``raw_evidence``
            is intentionally excluded.
        """
        return {
            "obligation_id": self.obligation_id,
            "title": self.title,
            "hipaa_security_refs": list(self.hipaa_security_refs),
            "status": _obligation_status(self).value,
            "evidence_refs": list(self.evidence_refs),
            "operator_action": self.operator_action,
            "owner": self.owner,
            "updated_at": self.updated_at,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class HipaaDocumentationPacket:
    """Tenant-safe HIPAA documentation readiness packet.

    Parameters
    ----------
    generated_at:
        UTC timestamp for the packet.
    readiness_report:
        SOC 2 / ISO readiness report with optional HIPAA crosswalk references.
    obligations:
        Operator-owned HIPAA obligations that cannot be inferred from product
        tests alone.
    phi_handling_summary:
        Deployment-facing summary of how Director-AI can limit PHI exposure and
        what remains operator-owned.
    baa_required:
        Whether the packet should flag business-associate-agreement review as a
        required deployment decision.
    """

    generated_at: str
    readiness_report: Soc2IsoReadinessReport
    obligations: tuple[HipaaDeploymentObligation, ...]
    phi_handling_summary: str
    baa_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a tenant-safe HIPAA documentation packet.

        Returns
        -------
        dict[str, Any]
            JSON-compatible packet for procurement, security, or deployment
            approval reviews. Raw PHI, raw interactions, and raw security
            evidence are never included.
        """
        return {
            "framework": "HIPAA Security Rule documentation readiness",
            "generated_at": self.generated_at,
            "summary": self.summary(),
            "phi_handling_summary": self.phi_handling_summary,
            "baa_required": self.baa_required,
            "soc2_iso_readiness": self.readiness_report.to_dict(),
            "obligations": [obligation.to_dict() for obligation in self.obligations],
            "disclaimer": _HIPAA_DISCLAIMER,
            "privacy": {
                "payload_classification": "tenant_safe",
                "raw_phi_included": False,
                "raw_interaction_text_included": False,
                "raw_security_evidence_included": False,
                "hipaa_compliance_claimed": False,
            },
        }

    def summary(self) -> dict[str, int | float | str | bool]:
        """Return aggregate HIPAA documentation readiness counts."""
        total = len(self.obligations)
        passed = sum(
            1
            for obligation in self.obligations
            if _obligation_status(obligation) is ReadinessStatus.PASS
        )
        warnings = sum(
            1
            for obligation in self.obligations
            if _obligation_status(obligation) is ReadinessStatus.WARNING
        )
        failures = sum(
            1
            for obligation in self.obligations
            if _obligation_status(obligation) is ReadinessStatus.FAIL
        )
        not_applicable = sum(
            1
            for obligation in self.obligations
            if _obligation_status(obligation) is ReadinessStatus.NOT_APPLICABLE
        )
        applicable = max(total - not_applicable, 1)
        readiness_score = round(passed / applicable, 4)
        return {
            "total_obligations": total,
            "passed": passed,
            "warnings": warnings,
            "failures": failures,
            "not_applicable": not_applicable,
            "readiness_score": readiness_score,
            "risk_level": _risk_level(failures=failures, warnings=warnings),
            "baa_required": self.baa_required,
        }

    def to_markdown(self) -> str:
        """Render the HIPAA documentation packet as Markdown."""
        summary = self.summary()
        lines = [
            "# HIPAA Documentation Readiness",
            "",
            f"Generated: {self.generated_at}",
            "",
            "## Summary",
        ]
        for key, value in summary.items():
            lines.append(f"- {key.replace('_', ' ').title()}: {value}")
        lines.extend(
            [
                "",
                "## PHI Handling Summary",
                "",
                self.phi_handling_summary,
                "",
                "## Operator Obligations",
                "",
                "| ID | Obligation | Status | HIPAA Security Rule | Evidence | Operator Action |",
                "|---|---|---:|---|---|---|",
            ]
        )
        for obligation in self.obligations:
            lines.append(
                "| "
                + " | ".join(
                    [
                        obligation.obligation_id,
                        obligation.title,
                        _obligation_status(obligation).value,
                        ", ".join(obligation.hipaa_security_refs),
                        ", ".join(obligation.evidence_refs),
                        obligation.operator_action,
                    ]
                )
                + " |"
            )
        lines.extend(["", f"> {_HIPAA_DISCLAIMER}"])
        return "\n".join(lines)


def _obligation_status(obligation: HipaaDeploymentObligation) -> ReadinessStatus:
    return cast(ReadinessStatus, obligation.status)


def build_soc2_iso_readiness_report(
    *,
    controls: tuple[Soc2IsoControl, ...] | list[Soc2IsoControl] | None = None,
    generated_at: str = "",
) -> Soc2IsoReadinessReport:
    """Build a tenant-safe SOC 2 / ISO/IEC 27001 readiness report.

    Parameters
    ----------
    controls:
        Optional operator-supplied control catalogue. Defaults to the built-in
        Director-AI readiness controls.
    generated_at:
        Optional UTC timestamp. If omitted, the current UTC time is used.

    Returns
    -------
    Soc2IsoReadinessReport
        Tenant-safe readiness report with SOC 2 Type I path guidance.
    """
    return Soc2IsoReadinessReport(
        generated_at=generated_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        controls=tuple(controls)
        if controls is not None
        else default_readiness_controls(),
    )


def default_readiness_controls() -> tuple[Soc2IsoControl, ...]:
    """Return Director-AI's default product-readiness control catalogue.

    Returns
    -------
    tuple[Soc2IsoControl, ...]
        Controls mapped to SOC 2, ISO/IEC 27001, and HIPAA Security Rule
        references where the product evidence directly supports the mapping.
    """
    return (
        Soc2IsoControl(
            control_id="SEC-01",
            title="Tenant authentication and access isolation",
            soc2_criteria=("security", "confidentiality"),
            iso27001_refs=("A.5.15", "A.8.3"),
            hipaa_security_refs=("45 CFR 164.308(a)(4)", "45 CFR 164.312(a)(1)"),
            status=ReadinessStatus.PASS,
            evidence_refs=("tests/test_server_auth.py", "tests/test_enterprise.py"),
        ),
        Soc2IsoControl(
            control_id="PRIV-01",
            title="PII redaction and tenant-safe audit metadata",
            soc2_criteria=("privacy", "confidentiality"),
            iso27001_refs=("A.8.11", "A.8.12"),
            hipaa_security_refs=("45 CFR 164.312(a)(1)", "45 CFR 164.312(e)(1)"),
            status=ReadinessStatus.PASS,
            evidence_refs=("tests/test_pii_redactor.py", "docs/BENCHMARKS.md"),
        ),
        Soc2IsoControl(
            control_id="MON-01",
            title="Safety monitoring dashboard and alert evidence",
            soc2_criteria=("security", "availability"),
            iso27001_refs=("A.8.16",),
            hipaa_security_refs=("45 CFR 164.308(a)(1)(ii)(D)",),
            status=ReadinessStatus.PASS,
            evidence_refs=("tests/test_safety_dashboard.py", "deploy/observability"),
        ),
        Soc2IsoControl(
            control_id="INC-01",
            title="Incident and human review audit trail",
            soc2_criteria=("security", "processing_integrity"),
            iso27001_refs=("A.5.24", "A.5.26"),
            hipaa_security_refs=("45 CFR 164.308(a)(6)",),
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
            hipaa_security_refs=("45 CFR 164.308(a)(1)(ii)(B)",),
            status=ReadinessStatus.WARNING,
            evidence_refs=("docs/internal/AUDIT_INDEX.md", ".github/workflows"),
            notes="External auditor evidence pack remains deployment-specific.",
        ),
        Soc2IsoControl(
            control_id="CHANGE-01",
            title="Change management and release verification",
            soc2_criteria=("security", "processing_integrity"),
            iso27001_refs=("A.8.32",),
            hipaa_security_refs=("45 CFR 164.308(a)(8)",),
            status=ReadinessStatus.WARNING,
            evidence_refs=("docs/PRODUCTION_CHECKLIST.md", ".githooks/pre-commit"),
            notes="Customer deployment change approvals are outside repository scope.",
        ),
    )


def build_hipaa_documentation_packet(
    *,
    readiness_report: Soc2IsoReadinessReport | None = None,
    obligations: tuple[HipaaDeploymentObligation, ...]
    | list[HipaaDeploymentObligation]
    | None = None,
    generated_at: str = "",
    phi_handling_summary: str = "",
    baa_required: bool = True,
) -> HipaaDocumentationPacket:
    """Build a tenant-safe HIPAA documentation readiness packet.

    Parameters
    ----------
    readiness_report:
        Optional SOC 2 / ISO readiness report to embed. If omitted, the default
        readiness catalogue is generated with the same timestamp.
    obligations:
        Optional deployment obligations. If omitted, the built-in HIPAA
        operator-obligation catalogue is used.
    generated_at:
        Optional UTC timestamp. If omitted, the current UTC time is used.
    phi_handling_summary:
        Optional operator-edited PHI handling summary. Defaults to the bounded
        product/operator responsibility statement.
    baa_required:
        Whether business-associate-agreement review should be marked required.

    Returns
    -------
    HipaaDocumentationPacket
        Tenant-safe HIPAA documentation readiness packet.
    """
    timestamp = generated_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report = readiness_report or build_soc2_iso_readiness_report(generated_at=timestamp)
    return HipaaDocumentationPacket(
        generated_at=timestamp,
        readiness_report=report,
        obligations=tuple(obligations)
        if obligations is not None
        else default_hipaa_obligations(),
        phi_handling_summary=phi_handling_summary or _DEFAULT_PHI_SUMMARY,
        baa_required=baa_required,
    )


def default_hipaa_obligations() -> tuple[HipaaDeploymentObligation, ...]:
    """Return deployment-owned HIPAA documentation obligations.

    Returns
    -------
    tuple[HipaaDeploymentObligation, ...]
        Operator obligations that complement product readiness evidence and
        must be reviewed before processing ePHI.
    """
    return (
        HipaaDeploymentObligation(
            obligation_id="HIPAA-RA-01",
            title="Risk analysis and risk management record",
            hipaa_security_refs=(
                "45 CFR 164.308(a)(1)(ii)(A)",
                "45 CFR 164.308(a)(1)(ii)(B)",
            ),
            status=ReadinessStatus.WARNING,
            evidence_refs=(
                "docs-site/guide/compliance-reporting.md",
                "docs/PRODUCTION_CHECKLIST.md",
            ),
            operator_action=(
                "Attach deployment-specific ePHI data-flow inventory, threat "
                "analysis, risk treatment decisions, and review cadence."
            ),
            notes="Director-AI can supply evidence references; the risk analysis is deployment-owned.",
        ),
        HipaaDeploymentObligation(
            obligation_id="HIPAA-BAA-01",
            title="Business associate agreement and sub-processor review",
            hipaa_security_refs=("45 CFR 164.308(b)(1)",),
            status=ReadinessStatus.WARNING,
            evidence_refs=("docs-site/privacy.md", "docs-site/licensing.md"),
            operator_action=(
                "Determine covered-entity or business-associate role, execute "
                "required agreements, and document sub-processors before ePHI use."
            ),
        ),
        HipaaDeploymentObligation(
            obligation_id="HIPAA-AUD-01",
            title="Audit controls and activity-review procedure",
            hipaa_security_refs=(
                "45 CFR 164.308(a)(1)(ii)(D)",
                "45 CFR 164.312(b)",
            ),
            status=ReadinessStatus.PASS,
            evidence_refs=(
                "tests/test_audit_log.py",
                "tests/test_audit_chain.py",
                "docs-site/api/enterprise.md#audit-logging",
            ),
            operator_action=(
                "Enable tamper-evident audit logging, define review frequency, "
                "and preserve reviewer sign-off outside the tenant-safe packet."
            ),
        ),
        HipaaDeploymentObligation(
            obligation_id="HIPAA-ACCESS-01",
            title="Unique user access and minimum-necessary configuration",
            hipaa_security_refs=(
                "45 CFR 164.308(a)(4)",
                "45 CFR 164.312(a)(1)",
                "45 CFR 164.312(a)(2)(i)",
            ),
            status=ReadinessStatus.WARNING,
            evidence_refs=(
                "tests/test_server_auth.py",
                "docs-site/deployment/production.md",
            ),
            operator_action=(
                "Bind API keys to tenants, enforce least-privilege operators, "
                "and document identity-provider controls for human users."
            ),
        ),
        HipaaDeploymentObligation(
            obligation_id="HIPAA-INC-01",
            title="Security incident and breach-response procedure",
            hipaa_security_refs=("45 CFR 164.308(a)(6)",),
            status=ReadinessStatus.WARNING,
            evidence_refs=(
                "docs-site/deployment/runbooks.md",
                "docs-site/deployment/monitoring.md",
            ),
            operator_action=(
                "Attach incident playbooks, breach-notification ownership, "
                "escalation contacts, and evidence-retention locations."
            ),
        ),
        HipaaDeploymentObligation(
            obligation_id="HIPAA-BACKUP-01",
            title="Contingency plan, backup, and recovery evidence",
            hipaa_security_refs=("45 CFR 164.308(a)(7)",),
            status=ReadinessStatus.WARNING,
            evidence_refs=(
                "docs-site/deployment/repository-backups.md",
                "docs-site/deployment/production.md",
            ),
            operator_action=(
                "Document backup scope, restore tests, disaster-recovery roles, "
                "and ePHI retention controls for the deployed environment."
            ),
        ),
    )


def _risk_level(*, failures: int, warnings: int) -> str:
    if failures:
        return "critical"
    if warnings:
        return "attention_required"
    return "ready"
