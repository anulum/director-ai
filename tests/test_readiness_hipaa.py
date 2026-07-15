# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — HIPAA documentation readiness tests

from __future__ import annotations

import json

import pytest

from director_ai.compliance import (
    HipaaDeploymentObligation,
    ReadinessStatus,
    build_hipaa_documentation_packet,
)

pytestmark = pytest.mark.enterprise


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
