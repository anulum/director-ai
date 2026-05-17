# SPDX-License-Identifier: AGPL-3.0-or-later
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
    ReadinessStatus,
    Soc2IsoControl,
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
    assert payload["summary"]["risk_level"] == "attention_required"
    assert payload["disclaimer"] == (
        "Readiness evidence only; this is not a SOC 2 report, ISO/IEC 27001 "
        "certification, or auditor opinion."
    )


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
        ({"soc2_criteria": ("unknown",)}, "soc2"),
        ({"iso27001_refs": ("5.15",)}, "iso27001"),
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
