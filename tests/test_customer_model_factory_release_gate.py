# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory release gate tests

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
from director_ai.core.customer_model_factory.release_gate import (
    build_release_gate_manifest,
)
from director_ai.core.customer_model_factory.risk_register import (
    CustomerRiskRegister,
)
from director_ai.core.customer_model_factory.runtime_package import (
    CustomerRuntimePackage,
)

ROOT = Path(__file__).resolve().parents[1]


def _runtime_package() -> CustomerRuntimePackage:
    return CustomerRuntimePackage(
        schema_version="1.0.0",
        runtime_id="runtime-bank-alpha-20260518",
        ready=True,
        customer_id="bank-alpha",
        workspace_id="bank-alpha-prod",
        tenant_id="bank-alpha-tenant",
        deployment_id="bank-alpha-prod-20260518",
        evidence_hash="a" * 64,
        runtime_mode="offline_private",
        runtime_config={
            "customer_id": "bank-alpha",
            "workspace_id": "bank-alpha-prod",
            "tenant_id": "bank-alpha-tenant",
            "deployment_id": "bank-alpha-prod-20260518",
            "deployment_hash": "b" * 64,
            "evidence_hash": "a" * 64,
            "selected_model_artifact_uri": "gs://customer-artifacts/bank-alpha/models/cmf-bank-alpha",
            "audit_log_uri": "gs://customer-artifacts/bank-alpha/audit/decision-log.jsonl",
            "evidence_pack_uri": "gs://customer-artifacts/bank-alpha/evidence/pack",
            "telemetry_mode": "customer_controlled",
            "external_callbacks_allowed": False,
        },
        findings=(),
        runtime_hash="c" * 64,
    )


def _evidence_pack() -> CustomerEvidencePackManifest:
    return CustomerEvidencePackManifest(
        schema_version="1.0.0",
        package_id="evidence-bank-alpha-20260518",
        ready=True,
        customer_id="bank-alpha",
        workspace_id="bank-alpha-prod",
        tenant_id="bank-alpha-tenant",
        deployment_id="bank-alpha-prod-20260518",
        environment="production",
        classification="restricted",
        export_uri="gs://customer-artifacts/bank-alpha/evidence/pack",
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
        monitoring_id="monitor-bank-alpha-20260518",
        ready=True,
        health_status="within_control",
        customer_id="bank-alpha",
        workspace_id="bank-alpha-prod",
        tenant_id="bank-alpha-tenant",
        runtime_id="runtime-bank-alpha-20260518",
        deployment_id="bank-alpha-prod-20260518",
        evidence_hash="a" * 64,
        decision_log_uri="gs://customer-artifacts/bank-alpha/audit/decision-log.jsonl",
        review_queue_uri="gs://customer-artifacts/bank-alpha/review/fp.jsonl",
        incident_queue_uri="gs://customer-artifacts/bank-alpha/incidents/fn.jsonl",
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
        package_version={"runtime_id": "runtime-bank-alpha-20260518"},
        retraining_recommended=False,
        recommendations=(),
        findings=(),
        monitoring_hash="d" * 64,
    )


def _risk_register() -> CustomerRiskRegister:
    return CustomerRiskRegister(
        schema_version="1.0.0",
        register_id="risk_register_bank_alpha-20260518",
        ready=True,
        customer_id="bank-alpha",
        workspace_id="bank-alpha-prod",
        tenant_id="bank-alpha-tenant",
        deployment_id="bank-alpha-prod-20260518",
        evidence_hash="a" * 64,
        monitoring_hash="d" * 64,
        generated_at="2026-05-18",
        risks=(),
        findings=(),
        register_hash="e" * 64,
    )


def test_release_gate_allows_promotion_when_all_artifacts_are_ready():
    gate = build_release_gate_manifest(
        release_id="release-bank-alpha-20260518",
        enterprise_ready=True,
        enterprise_blocking_debt_ids=(),
        runtime_package=_runtime_package(),
        evidence_pack=_evidence_pack(),
        monitoring_manifest=_monitoring_manifest(),
        risk_register=_risk_register(),
        generated_at="2026-05-18T18:35:00Z",
    )

    assert gate.ready is True
    assert gate.promotion_allowed is True
    assert gate.blockers == ()
    assert gate.customer_id == "bank-alpha"
    assert gate.artifact_hashes["runtime_hash"] == "c" * 64
    assert gate.artifact_hashes["evidence_hash"] == "a" * 64
    assert gate.artifact_hashes["monitoring_hash"] == "d" * 64
    assert gate.artifact_hashes["risk_register_hash"] == "e" * 64
    assert len(gate.release_hash) == 64


def test_release_gate_blocks_enterprise_trust_debt():
    gate = build_release_gate_manifest(
        release_id="release-bank-alpha-blocked",
        enterprise_ready=False,
        enterprise_blocking_debt_ids=("TRUST-DEBT-0002",),
        runtime_package=_runtime_package(),
        evidence_pack=_evidence_pack(),
        monitoring_manifest=_monitoring_manifest(),
        risk_register=_risk_register(),
        generated_at="2026-05-18T18:35:00Z",
    )

    assert gate.ready is False
    assert gate.promotion_allowed is False
    assert any(
        blocker["code"] == "enterprise_trust_not_ready" for blocker in gate.blockers
    )


def test_release_gate_blocks_not_ready_required_artifacts():
    risk_register = _risk_register()
    risk_register = CustomerRiskRegister(
        **{
            **risk_register.to_dict(),
            "ready": False,
            "findings": [{"code": "accepted_risk_expired"}],
        },
    )

    gate = build_release_gate_manifest(
        release_id="release-bank-alpha-risk-blocked",
        enterprise_ready=True,
        enterprise_blocking_debt_ids=(),
        runtime_package=_runtime_package(),
        evidence_pack=_evidence_pack(),
        monitoring_manifest=_monitoring_manifest(),
        risk_register=risk_register,
        generated_at="2026-05-18T18:35:00Z",
    )

    assert gate.ready is False
    assert any(
        blocker["code"] == "risk_register_not_ready" for blocker in gate.blockers
    )


def test_release_gate_blocks_customer_boundary_mismatch():
    evidence = _evidence_pack()
    evidence = CustomerEvidencePackManifest(
        **{**evidence.to_dict(), "tenant_id": "wrong-tenant"},
    )

    gate = build_release_gate_manifest(
        release_id="release-bank-alpha-mismatch",
        enterprise_ready=True,
        enterprise_blocking_debt_ids=(),
        runtime_package=_runtime_package(),
        evidence_pack=evidence,
        monitoring_manifest=_monitoring_manifest(),
        risk_register=_risk_register(),
        generated_at="2026-05-18T18:35:00Z",
    )

    assert gate.ready is False
    assert any(
        blocker["code"] == "tenant_boundary_mismatch" for blocker in gate.blockers
    )


def test_release_gate_serialises_deterministically(tmp_path: Path):
    gate = build_release_gate_manifest(
        release_id="release-bank-alpha-20260518",
        enterprise_ready=True,
        enterprise_blocking_debt_ids=(),
        runtime_package=_runtime_package(),
        evidence_pack=_evidence_pack(),
        monitoring_manifest=_monitoring_manifest(),
        risk_register=_risk_register(),
        generated_at="2026-05-18T18:35:00Z",
    )

    output = gate.write_json(tmp_path / "release_gate.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload == gate.to_dict()
    assert payload["release_hash"] == gate.release_hash


def test_release_gate_schema_is_machine_readable():
    schema_path = ROOT / "schemas" / "customer-model-factory-release-gate.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == "DIRECTOR-AI Customer Model Factory Release Gate"
    assert set(schema["required"]) >= {
        "release_id",
        "promotion_allowed",
        "artifact_hashes",
        "blockers",
        "release_hash",
    }
