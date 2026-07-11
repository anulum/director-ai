# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — SOC 2 / ISO 27001 readiness tests

from __future__ import annotations

import json

import pytest

from director_ai.compliance import (
    HipaaDeploymentObligation,
    ReadinessStatus,
    Soc2IsoControl,
    build_hipaa_documentation_packet,
    build_soc2_iso_readiness_report,
)

pytestmark = pytest.mark.enterprise


def test_readiness_report_scores_controls_and_exports_trust_controls() -> None:
    report = build_soc2_iso_readiness_report(
        controls=[
            Soc2IsoControl(
                control_id="SEC-01",
                title="Tenant access control",
                soc2_criteria=("security", "confidentiality"),
                iso27001_refs=("A.5.15", "A.8.3"),
                status=ReadinessStatus.PASS,
                evidence_refs=("tests/test_server_auth.py", "SECURITY.md"),
                owner="security",
                updated_at="2026-05-17",
            ),
            Soc2IsoControl(
                control_id="MON-01",
                title="Continuous monitoring",
                soc2_criteria=("availability",),
                iso27001_refs=("A.8.16",),
                status=ReadinessStatus.WARNING,
                evidence_refs=("deploy/observability/grafana-dashboard.json",),
            ),
        ],
        generated_at="2026-05-17T13:00:00Z",
    )

    payload = report.to_dict()
    trust_controls = report.to_trust_controls()

    assert payload["summary"] == {
        "total_controls": 2,
        "passed": 1,
        "warnings": 1,
        "failures": 0,
        "not_applicable": 0,
        "readiness_score": 0.5,
        "risk_level": "attention_required",
    }
    assert payload["privacy"] == {
        "payload_classification": "tenant_safe",
        "raw_security_evidence_included": False,
        "certification_claimed": False,
    }
    assert payload["soc2_type_i_path"]
    assert payload["controls"][0]["hipaa_security_refs"] == []
    assert trust_controls[0].control == "SOC2/ISO SEC-01: Tenant access control"
    assert trust_controls[0].status == "passed"
    assert trust_controls[1].status == "warning"


def test_default_catalogue_covers_product_readiness_surfaces() -> None:
    report = build_soc2_iso_readiness_report(generated_at="2026-05-17T13:00:00Z")
    payload = report.to_dict()
    control_ids = {control["control_id"] for control in payload["controls"]}

    assert {
        "SEC-01",
        "PRIV-01",
        "MON-01",
        "INC-01",
        "VULN-01",
        "CHANGE-01",
    }.issubset(control_ids)
    hipaa_refs = {
        ref for control in payload["controls"] for ref in control["hipaa_security_refs"]
    }
    assert {
        "45 CFR 164.308(a)(1)(ii)(B)",
        "45 CFR 164.308(a)(1)(ii)(D)",
        "45 CFR 164.312(a)(1)",
    }.issubset(hipaa_refs)
    assert payload["summary"]["risk_level"] == "attention_required"
    assert payload["disclaimer"] == (
        "Readiness evidence only; this is not a SOC 2 report, ISO/IEC 27001 "
        "certification, or auditor opinion."
    )


def test_readiness_markdown_renders_summary_and_control_rows() -> None:
    report = build_soc2_iso_readiness_report(
        controls=[
            Soc2IsoControl(
                control_id="SEC-02",
                title="Credential rotation",
                soc2_criteria=("security",),
                iso27001_refs=("A.5.17",),
                status=ReadinessStatus.FAIL,
                evidence_refs=("SECURITY.md#rotation",),
            ),
            Soc2IsoControl(
                control_id="NA-01",
                title="Legacy data centre controls",
                soc2_criteria=("availability",),
                iso27001_refs=("A.5.30",),
                status=ReadinessStatus.NOT_APPLICABLE,
                evidence_refs=("docs/PRODUCTION_CHECKLIST.md",),
            ),
        ],
        generated_at="2026-05-17T13:00:00Z",
    )

    summary = report.summary()
    markdown = report.to_markdown()

    assert summary["risk_level"] == "critical"
    assert summary["readiness_score"] == 0.0
    assert "# SOC 2 / ISO 27001 Readiness" in markdown
    assert "## SOC 2 Type I Path" in markdown
    assert "- Failures: 1" in markdown
    assert (
        "| SEC-02 | Credential rotation | failing | security | A.5.17 |  |" in markdown
    )
    assert "| NA-01 | Legacy data centre controls | not_applicable |" in markdown


def test_report_is_tenant_safe_and_does_not_serialize_raw_evidence() -> None:
    report = build_soc2_iso_readiness_report(
        controls=[
            Soc2IsoControl(
                control_id="PRIV-01",
                title="PII redaction",
                soc2_criteria=("privacy", "confidentiality"),
                iso27001_refs=("A.8.11",),
                status="passed",
                evidence_refs=("docs/BENCHMARKS.md#pii-redaction",),
                raw_evidence="Customer email jane@example.com was redacted.",
            ),
        ],
    )

    encoded = json.dumps(report.to_dict(), sort_keys=True)

    assert "jane@example.com" not in encoded
    assert "raw_evidence" not in encoded
    assert "docs/BENCHMARKS.md#pii-redaction" in encoded


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"control_id": "bad id"}, "control_id"),
        ({"title": "  "}, "title"),
        ({"soc2_criteria": ("unknown",)}, "soc2"),
        ({"iso27001_refs": ("5.15",)}, "iso27001"),
        ({"hipaa_security_refs": ("164.312(a)(1)",)}, "hipaa"),
        ({"evidence_refs": ()}, "evidence_refs"),
        ({"status": "maybe"}, "status"),
    ],
)
def test_control_validation_rejects_invalid_operator_input(
    kwargs: dict[str, object],
    message: str,
) -> None:
    base = {
        "control_id": "SEC-01",
        "title": "Tenant access control",
        "soc2_criteria": ("security",),
        "iso27001_refs": ("A.5.15",),
        "status": "passed",
        "evidence_refs": ("SECURITY.md",),
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=message):
        Soc2IsoControl(**base)


def test_hipaa_documentation_packet_is_tenant_safe_and_bounded() -> None:
    packet = build_hipaa_documentation_packet(
        generated_at="2026-06-18T08:00:00Z",
        obligations=[
            HipaaDeploymentObligation(
                obligation_id="HIPAA-AUD-01",
                title="Audit controls and activity review",
                hipaa_security_refs=(
                    "45 CFR 164.308(a)(1)(ii)(D)",
                    "45 CFR 164.312(b)",
                ),
                status=ReadinessStatus.PASS,
                evidence_refs=("tests/test_audit_chain.py",),
                operator_action="Enable audit review and retain reviewer sign-off.",
                raw_evidence="Patient name Jane Example appeared in a trace.",
            ),
            HipaaDeploymentObligation(
                obligation_id="HIPAA-BAA-01",
                title="Business associate agreement review",
                hipaa_security_refs=("45 CFR 164.308(b)(1)",),
                status=ReadinessStatus.WARNING,
                evidence_refs=("docs-site/privacy.md",),
                operator_action="Complete role analysis before processing ePHI.",
            ),
        ],
        phi_handling_summary="No raw PHI is included in this packet.",
    )

    payload = packet.to_dict()
    markdown = packet.to_markdown()
    encoded = json.dumps(payload, sort_keys=True)

    assert payload["summary"] == {
        "total_obligations": 2,
        "passed": 1,
        "warnings": 1,
        "failures": 0,
        "not_applicable": 0,
        "readiness_score": 0.5,
        "risk_level": "attention_required",
        "baa_required": True,
    }
    assert payload["privacy"] == {
        "payload_classification": "tenant_safe",
        "raw_phi_included": False,
        "raw_interaction_text_included": False,
        "raw_security_evidence_included": False,
        "hipaa_compliance_claimed": False,
    }
    assert "Jane Example" not in encoded
    assert "raw_evidence" not in encoded
    assert "not legal advice" in payload["disclaimer"]
    assert "# HIPAA Documentation Readiness" in markdown
    assert "45 CFR 164.312(b)" in markdown


def test_default_hipaa_packet_records_operator_owned_obligations() -> None:
    packet = build_hipaa_documentation_packet(
        generated_at="2026-06-18T08:00:00Z",
        baa_required=False,
    )
    payload = packet.to_dict()
    obligation_ids = {
        obligation["obligation_id"] for obligation in payload["obligations"]
    }

    assert {
        "HIPAA-RA-01",
        "HIPAA-BAA-01",
        "HIPAA-AUD-01",
        "HIPAA-ACCESS-01",
        "HIPAA-INC-01",
        "HIPAA-BACKUP-01",
    }.issubset(obligation_ids)
    assert payload["summary"]["baa_required"] is False
    assert payload["summary"]["risk_level"] == "attention_required"
    assert payload["soc2_iso_readiness"]["privacy"]["certification_claimed"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"obligation_id": "bad id"}, "obligation_id"),
        ({"title": " "}, "title"),
        ({"hipaa_security_refs": ()}, "hipaa"),
        ({"hipaa_security_refs": ("164.308(a)(1)",)}, "hipaa"),
        ({"evidence_refs": ()}, "evidence_refs"),
        ({"operator_action": " "}, "operator_action"),
        ({"status": "unknown"}, "status"),
    ],
)
def test_hipaa_obligation_validation_rejects_invalid_operator_input(
    kwargs: dict[str, object],
    message: str,
) -> None:
    base = {
        "obligation_id": "HIPAA-AUD-01",
        "title": "Audit controls",
        "hipaa_security_refs": ("45 CFR 164.312(b)",),
        "status": "passed",
        "evidence_refs": ("tests/test_audit_chain.py",),
        "operator_action": "Enable audit review.",
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=message):
        HipaaDeploymentObligation(**base)


def test_builders_return_the_readiness_report_and_hipaa_packet_contracts() -> None:
    from director_ai.compliance.readiness import (
        HipaaDocumentationPacket,
        Soc2IsoReadinessReport,
    )

    packet = build_hipaa_documentation_packet(generated_at="2026-07-12T00:00:00Z")

    assert isinstance(packet, HipaaDocumentationPacket)
    assert packet.generated_at == "2026-07-12T00:00:00Z"
    assert packet.baa_required is True
    assert packet.obligations
    assert isinstance(packet.readiness_report, Soc2IsoReadinessReport)
    assert packet.readiness_report.generated_at == packet.generated_at
