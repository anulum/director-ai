# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory risk register tests

from __future__ import annotations

import json
from pathlib import Path

from director_ai.core.customer_model_factory.evidence_pack import (
    CustomerEvidencePackManifest,
)
from director_ai.core.customer_model_factory.monitoring_manifest import (
    CustomerMonitoringManifest,
    MonitoringMetrics,
    MonitoringThresholds,
)
from director_ai.core.customer_model_factory.risk_register import (
    CustomerRiskException,
    build_risk_register,
)

ROOT = Path(__file__).resolve().parents[1]


def _evidence_pack() -> CustomerEvidencePackManifest:
    return CustomerEvidencePackManifest(
        schema_version="1.0.0",
        package_id="evidence-customer-alpha-20260518",
        ready=True,
        customer_id="customer-alpha",
        workspace_id="customer-alpha-prod",
        tenant_id="customer-alpha-tenant",
        deployment_id="customer-alpha-prod-20260518",
        environment="production",
        classification="restricted",
        export_uri="gs://customer-artifacts/customer-alpha/evidence/pack",
        external_callbacks_allowed=False,
        callback_endpoints=(),
        artefacts={"deployment_hash": "b" * 64},
        control_evidence={"human_escalation": ("requires_escalation",)},
        findings=(),
        evidence_hash="a" * 64,
    )


def _monitoring_manifest() -> CustomerMonitoringManifest:
    return CustomerMonitoringManifest(
        schema_version="1.0.0",
        monitoring_id="monitor-customer-alpha-20260518",
        ready=True,
        health_status="within_control",
        customer_id="customer-alpha",
        workspace_id="customer-alpha-prod",
        tenant_id="customer-alpha-tenant",
        runtime_id="runtime-customer-alpha-20260518",
        deployment_id="customer-alpha-prod-20260518",
        evidence_hash="a" * 64,
        decision_log_uri="gs://customer-artifacts/customer-alpha/audit/decision-log.jsonl",
        review_queue_uri="gs://customer-artifacts/customer-alpha/review/fp.jsonl",
        incident_queue_uri="gs://customer-artifacts/customer-alpha/incidents/fn.jsonl",
        observation_window="2026-05-18T00:00:00Z/2026-05-18T12:00:00Z",
        monitored_at="2026-05-18T12:05:00Z",
        metrics=MonitoringMetrics(
            total_decisions=1200,
            input_drift_score=0.04,
            source_corpus_drift_score=0.03,
            false_positive_review_count=2,
            false_negative_incident_count=0,
            abstention_rate=0.05,
            escalation_rate=0.09,
            latency_p95_ms=48.0,
            cost_per_1k_decisions=0.12,
        ),
        thresholds=MonitoringThresholds(
            max_input_drift_score=0.20,
            max_source_corpus_drift_score=0.15,
            max_false_negative_incidents=0,
            max_abstention_rate=0.20,
            max_escalation_rate=0.30,
            max_latency_p95_ms=250.0,
            max_cost_per_1k_decisions=1.50,
        ),
        package_version={"runtime_id": "runtime-customer-alpha-20260518"},
        retraining_recommended=False,
        recommendations=(),
        findings=(),
        monitoring_hash="d" * 64,
    )


def _risk(**overrides: object) -> CustomerRiskException:
    payload = {
        "risk_id": "risk-customer-alpha-001",
        "status": "accepted",
        "severity": "medium",
        "owner": "customer-alpha-risk-owner",
        "accepted_at": "2026-05-18",
        "expires_at": "2026-06-18",
        "compensating_controls": ("manual review of escalated cases",),
        "linked_artifact_hashes": {
            "evidence_hash": "a" * 64,
            "monitoring_hash": "d" * 64,
        },
        "rationale": "Temporary acceptance during customer pilot.",
    }
    payload.update(overrides)
    return CustomerRiskException(**payload)


def test_risk_register_ready_when_accepted_risks_have_owner_expiry_and_controls():
    register = build_risk_register(
        register_id="risk_register_customer_alpha-20260518",
        evidence_pack=_evidence_pack(),
        monitoring_manifest=_monitoring_manifest(),
        risks=(_risk(),),
        generated_at="2026-05-18",
    )

    assert register.ready is True
    assert register.findings == ()
    assert register.customer_id == "customer-alpha"
    assert register.tenant_id == "customer-alpha-tenant"
    assert len(register.register_hash) == 64


def test_risk_register_blocks_expired_accepted_risks():
    register = build_risk_register(
        register_id="risk_register_customer_alpha-expired",
        evidence_pack=_evidence_pack(),
        monitoring_manifest=_monitoring_manifest(),
        risks=(_risk(expires_at="2026-05-01"),),
        generated_at="2026-05-18",
    )

    assert register.ready is False
    assert any(
        finding["code"] == "accepted_risk_expired" for finding in register.findings
    )


def test_risk_register_blocks_missing_owner_or_controls():
    register = build_risk_register(
        register_id="risk_register_customer_alpha-incomplete",
        evidence_pack=_evidence_pack(),
        monitoring_manifest=_monitoring_manifest(),
        risks=(_risk(owner="", compensating_controls=()),),
        generated_at="2026-05-18",
    )

    assert register.ready is False
    assert {finding["code"] for finding in register.findings} >= {
        "accepted_risk_owner_missing",
        "accepted_risk_controls_missing",
    }


def test_risk_register_blocks_unlinked_or_mismatched_artifact_hashes():
    register = build_risk_register(
        register_id="risk_register_customer_alpha-unlinked",
        evidence_pack=_evidence_pack(),
        monitoring_manifest=_monitoring_manifest(),
        risks=(_risk(linked_artifact_hashes={"evidence_hash": "wrong"}),),
        generated_at="2026-05-18",
    )

    assert register.ready is False
    assert {finding["code"] for finding in register.findings} >= {
        "risk_evidence_hash_mismatch",
        "risk_monitoring_hash_mismatch",
    }


def test_risk_register_blocks_missing_identity_not_ready_and_boundary_mismatches():
    evidence_pack = CustomerEvidencePackManifest(
        **{
            **_evidence_pack().to_dict(),
            "ready": False,
            "findings": [{"code": "evidence_pack_missing"}],
        }
    )
    monitoring = CustomerMonitoringManifest(
        **{
            **_monitoring_manifest().to_dict(),
            "ready": False,
            "customer_id": "wrong-customer",
            "workspace_id": "wrong-workspace",
            "tenant_id": "wrong-tenant",
            "deployment_id": "wrong-deployment",
            "evidence_hash": "f" * 64,
            "findings": [{"code": "decision_log_missing"}],
        }
    )

    register = build_risk_register(
        register_id=" ",
        evidence_pack=evidence_pack,
        monitoring_manifest=monitoring,
        risks=(),
        generated_at=" ",
    )

    assert register.ready is False
    assert {finding["code"] for finding in register.findings} >= {
        "register_id_missing",
        "generated_at_missing",
        "evidence_pack_not_ready",
        "monitoring_manifest_not_ready",
        "customer_boundary_mismatch",
        "workspace_boundary_mismatch",
        "tenant_boundary_mismatch",
        "deployment_boundary_mismatch",
        "evidence_hash_mismatch",
    }


def test_risk_register_rejects_unknown_status_and_missing_expiry():
    rejected = _risk(
        status="rejected", owner="", expires_at="", compensating_controls=()
    )
    unsupported = _risk(status="waived")
    accepted_missing_expiry = _risk(expires_at="")

    register = build_risk_register(
        register_id="risk_register_customer_alpha-statuses",
        evidence_pack=_evidence_pack(),
        monitoring_manifest=_monitoring_manifest(),
        risks=(rejected, unsupported, accepted_missing_expiry),
        generated_at="2026-05-18",
    )

    assert register.ready is False
    assert {finding["code"] for finding in register.findings} >= {
        "risk_status_unknown",
        "accepted_risk_expiry_missing",
    }
    assert "accepted_risk_owner_missing" not in {
        finding["code"] for finding in register.findings
    }


def test_risk_exception_round_trips_from_dict():
    risk = _risk()

    restored = CustomerRiskException.from_dict(risk.to_dict())

    assert restored == risk


def test_risk_register_serialises_and_round_trips(tmp_path: Path):
    register = build_risk_register(
        register_id="risk_register_customer_alpha-20260518",
        evidence_pack=_evidence_pack(),
        monitoring_manifest=_monitoring_manifest(),
        risks=(_risk(),),
        generated_at="2026-05-18",
    )

    output = register.write_json(tmp_path / "risk_register.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload == register.to_dict()
    assert payload["register_hash"] == register.register_hash


def test_risk_register_schema_is_machine_readable():
    schema_path = ROOT / "schemas" / "customer-model-factory-risk-register.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == "DIRECTOR-AI Customer Model Factory Risk Register"
    assert set(schema["required"]) >= {
        "register_id",
        "customer_id",
        "tenant_id",
        "risks",
        "findings",
        "register_hash",
    }
