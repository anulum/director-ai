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
    DeploymentHardeningEvidence,
    ObservabilityOperationsEvidence,
    ProvenanceLineageEvidence,
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
        runtime_id="runtime-customer-alpha-20260518",
        ready=True,
        customer_id="customer-alpha",
        workspace_id="customer-alpha-prod",
        tenant_id="customer-alpha-tenant",
        deployment_id="customer-alpha-prod-20260518",
        evidence_hash="a" * 64,
        runtime_mode="offline_private",
        runtime_config={
            "customer_id": "customer-alpha",
            "workspace_id": "customer-alpha-prod",
            "tenant_id": "customer-alpha-tenant",
            "deployment_id": "customer-alpha-prod-20260518",
            "deployment_hash": "b" * 64,
            "evidence_hash": "a" * 64,
            "selected_model_artifact_uri": "gs://customer-artifacts/customer-alpha/models/cmf-customer-alpha",
            "audit_log_uri": "gs://customer-artifacts/customer-alpha/audit/decision-log.jsonl",
            "evidence_pack_uri": "gs://customer-artifacts/customer-alpha/evidence/pack",
            "telemetry_mode": "customer_controlled",
            "external_callbacks_allowed": False,
        },
        findings=(),
        runtime_hash="c" * 64,
    )


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


def _risk_register() -> CustomerRiskRegister:
    return CustomerRiskRegister(
        schema_version="1.0.0",
        register_id="risk_register_customer_alpha-20260518",
        ready=True,
        customer_id="customer-alpha",
        workspace_id="customer-alpha-prod",
        tenant_id="customer-alpha-tenant",
        deployment_id="customer-alpha-prod-20260518",
        evidence_hash="a" * 64,
        monitoring_hash="d" * 64,
        generated_at="2026-05-18",
        risks=(),
        findings=(),
        register_hash="e" * 64,
    )


def _deployment_hardening_evidence() -> DeploymentHardeningEvidence:
    return DeploymentHardeningEvidence(
        ready=True,
        environment="staging",
        observation_window="2026-05-18T00:00:00Z/2026-05-18T12:00:00Z",
        telemetry_uri="gs://customer-artifacts/customer-alpha/telemetry/r17.jsonl",
        sustained_load_packet_uri="gs://customer-artifacts/customer-alpha/evidence/sustained-load.json",
        operator_signoff_uri="gs://customer-artifacts/customer-alpha/signoff/r17.json",
        async_ordering_passed=True,
        tenant_poisoning_passed=True,
        evidence_hash="f" * 64,
    )


def _observability_operations_evidence() -> ObservabilityOperationsEvidence:
    return ObservabilityOperationsEvidence(
        ready=True,
        environment="staging",
        operations_packet_uri="gs://customer-artifacts/customer-alpha/observability/operations-packet.json",
        dashboard_evidence_uri="gs://customer-artifacts/customer-alpha/observability/dashboard-evidence.json",
        operator_signoff_uri="gs://customer-artifacts/customer-alpha/signoff/r8.json",
        controls_passed=True,
        compliance_exports_available=True,
        drift_reviewed=True,
        evidence_hash="1" * 64,
    )


def _provenance_lineage_evidence() -> ProvenanceLineageEvidence:
    return ProvenanceLineageEvidence(
        ready=True,
        environment="staging",
        feedback_loop_run_uri="gs://customer-artifacts/customer-alpha/provenance/live-feedback-run.json",
        signed_lineage_packet_uri="gs://customer-artifacts/customer-alpha/provenance/signed-lineage.json",
        tenant_kb_snapshot_uri="gs://customer-artifacts/customer-alpha/provenance/tenant-kb-snapshot.json",
        operator_signoff_uri="gs://customer-artifacts/customer-alpha/signoff/r9.json",
        lineage_matches_deployed_facts=True,
        protected_claim_conflicts_resolved=True,
        evidence_hash="2" * 64,
    )


def test_release_gate_allows_promotion_when_all_artifacts_are_ready():
    gate = build_release_gate_manifest(
        release_id="release-customer-alpha-20260518",
        enterprise_ready=True,
        enterprise_blocking_debt_ids=(),
        runtime_package=_runtime_package(),
        evidence_pack=_evidence_pack(),
        monitoring_manifest=_monitoring_manifest(),
        risk_register=_risk_register(),
        observability_operations_evidence=_observability_operations_evidence(),
        provenance_lineage_evidence=_provenance_lineage_evidence(),
        deployment_hardening_evidence=_deployment_hardening_evidence(),
        generated_at="2026-05-18T18:35:00Z",
    )

    assert gate.ready is True
    assert gate.promotion_allowed is True
    assert gate.blockers == ()
    assert gate.customer_id == "customer-alpha"
    assert gate.artifact_hashes["runtime_hash"] == "c" * 64
    assert gate.artifact_hashes["evidence_hash"] == "a" * 64
    assert gate.artifact_hashes["monitoring_hash"] == "d" * 64
    assert gate.artifact_hashes["risk_register_hash"] == "e" * 64
    assert gate.observability_operations_evidence.drift_reviewed is True
    assert gate.provenance_lineage_evidence.lineage_matches_deployed_facts is True
    assert gate.deployment_hardening_evidence.tenant_poisoning_passed is True
    assert len(gate.release_hash) == 64


def test_release_gate_blocks_enterprise_trust_debt():
    gate = build_release_gate_manifest(
        release_id="release-customer-alpha-blocked",
        enterprise_ready=False,
        enterprise_blocking_debt_ids=("TRUST-DEBT-0002",),
        runtime_package=_runtime_package(),
        evidence_pack=_evidence_pack(),
        monitoring_manifest=_monitoring_manifest(),
        risk_register=_risk_register(),
        observability_operations_evidence=_observability_operations_evidence(),
        provenance_lineage_evidence=_provenance_lineage_evidence(),
        deployment_hardening_evidence=_deployment_hardening_evidence(),
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
        release_id="release-customer-alpha-risk-blocked",
        enterprise_ready=True,
        enterprise_blocking_debt_ids=(),
        runtime_package=_runtime_package(),
        evidence_pack=_evidence_pack(),
        monitoring_manifest=_monitoring_manifest(),
        risk_register=risk_register,
        observability_operations_evidence=_observability_operations_evidence(),
        provenance_lineage_evidence=_provenance_lineage_evidence(),
        deployment_hardening_evidence=_deployment_hardening_evidence(),
        generated_at="2026-05-18T18:35:00Z",
    )

    assert gate.ready is False
    assert any(
        blocker["code"] == "risk_register_not_ready" for blocker in gate.blockers
    )


def test_release_gate_blocks_missing_release_identity_and_all_not_ready_artifacts():
    runtime = CustomerRuntimePackage(
        **{
            **_runtime_package().to_dict(),
            "ready": False,
            "findings": [{"code": "runtime_mode_unknown"}],
        }
    )
    evidence = CustomerEvidencePackManifest(
        **{
            **_evidence_pack().to_dict(),
            "ready": False,
            "findings": [{"code": "audit_log_missing"}],
        }
    )
    monitoring = CustomerMonitoringManifest(
        **{
            **_monitoring_manifest().to_dict(),
            "ready": False,
            "findings": [{"code": "review_queue_missing"}],
        }
    )
    risk_register = CustomerRiskRegister(
        **{
            **_risk_register().to_dict(),
            "ready": False,
            "findings": [{"code": "accepted_risk_expired"}],
        }
    )

    gate = build_release_gate_manifest(
        release_id=" ",
        enterprise_ready=True,
        enterprise_blocking_debt_ids=(),
        runtime_package=runtime,
        evidence_pack=evidence,
        monitoring_manifest=monitoring,
        risk_register=risk_register,
        observability_operations_evidence=_observability_operations_evidence(),
        provenance_lineage_evidence=_provenance_lineage_evidence(),
        deployment_hardening_evidence=_deployment_hardening_evidence(),
        generated_at=" ",
    )

    assert gate.ready is False
    assert {blocker["code"] for blocker in gate.blockers} >= {
        "release_id_missing",
        "generated_at_missing",
        "runtime_package_not_ready",
        "evidence_pack_not_ready",
        "monitoring_manifest_not_ready",
        "risk_register_not_ready",
    }


def test_release_gate_blocks_missing_observability_operations_evidence():
    evidence = ObservabilityOperationsEvidence(
        ready=False,
        environment="local",
        operations_packet_uri="",
        dashboard_evidence_uri="",
        operator_signoff_uri="",
        controls_passed=False,
        compliance_exports_available=False,
        drift_reviewed=False,
        evidence_hash="not-a-sha",
    )

    gate = build_release_gate_manifest(
        release_id="release-customer-alpha-r8-blocked",
        enterprise_ready=True,
        enterprise_blocking_debt_ids=(),
        runtime_package=_runtime_package(),
        evidence_pack=_evidence_pack(),
        monitoring_manifest=_monitoring_manifest(),
        risk_register=_risk_register(),
        observability_operations_evidence=evidence,
        provenance_lineage_evidence=_provenance_lineage_evidence(),
        deployment_hardening_evidence=_deployment_hardening_evidence(),
        generated_at="2026-05-18T18:35:00Z",
    )

    assert gate.ready is False
    assert {blocker["code"] for blocker in gate.blockers} >= {
        "observability_operations_not_ready",
        "observability_environment_invalid",
        "observability_operations_packet_missing",
        "observability_dashboard_evidence_missing",
        "observability_operator_signoff_missing",
        "observability_controls_failed",
        "observability_compliance_exports_missing",
        "observability_drift_not_reviewed",
        "observability_evidence_hash_invalid",
    }


def test_observability_operations_evidence_round_trips_from_json_safe_dict():
    evidence = _observability_operations_evidence()

    restored = ObservabilityOperationsEvidence.from_dict(evidence.to_dict())

    assert restored == evidence


def test_release_gate_blocks_missing_provenance_lineage_evidence():
    evidence = ProvenanceLineageEvidence(
        ready=False,
        environment="local",
        feedback_loop_run_uri="",
        signed_lineage_packet_uri="",
        tenant_kb_snapshot_uri="",
        operator_signoff_uri="",
        lineage_matches_deployed_facts=False,
        protected_claim_conflicts_resolved=False,
        evidence_hash="not-a-sha",
    )

    gate = build_release_gate_manifest(
        release_id="release-customer-alpha-r9-blocked",
        enterprise_ready=True,
        enterprise_blocking_debt_ids=(),
        runtime_package=_runtime_package(),
        evidence_pack=_evidence_pack(),
        monitoring_manifest=_monitoring_manifest(),
        risk_register=_risk_register(),
        observability_operations_evidence=_observability_operations_evidence(),
        provenance_lineage_evidence=evidence,
        deployment_hardening_evidence=_deployment_hardening_evidence(),
        generated_at="2026-05-18T18:35:00Z",
    )

    assert gate.ready is False
    assert {blocker["code"] for blocker in gate.blockers} >= {
        "provenance_lineage_not_ready",
        "provenance_lineage_environment_invalid",
        "provenance_feedback_loop_run_missing",
        "provenance_signed_lineage_packet_missing",
        "provenance_tenant_kb_snapshot_missing",
        "provenance_operator_signoff_missing",
        "provenance_lineage_mismatch",
        "provenance_conflicts_unresolved",
        "provenance_evidence_hash_invalid",
    }


def test_provenance_lineage_evidence_round_trips_from_json_safe_dict():
    evidence = _provenance_lineage_evidence()

    restored = ProvenanceLineageEvidence.from_dict(evidence.to_dict())

    assert restored == evidence


def test_release_gate_blocks_missing_deployment_hardening_evidence():
    evidence = DeploymentHardeningEvidence(
        ready=False,
        environment="local",
        observation_window="",
        telemetry_uri="",
        sustained_load_packet_uri="",
        operator_signoff_uri="",
        async_ordering_passed=False,
        tenant_poisoning_passed=False,
        evidence_hash="not-a-sha",
    )

    gate = build_release_gate_manifest(
        release_id="release-customer-alpha-r17-blocked",
        enterprise_ready=True,
        enterprise_blocking_debt_ids=(),
        runtime_package=_runtime_package(),
        evidence_pack=_evidence_pack(),
        monitoring_manifest=_monitoring_manifest(),
        risk_register=_risk_register(),
        observability_operations_evidence=_observability_operations_evidence(),
        provenance_lineage_evidence=_provenance_lineage_evidence(),
        deployment_hardening_evidence=evidence,
        generated_at="2026-05-18T18:35:00Z",
    )

    assert gate.ready is False
    assert {blocker["code"] for blocker in gate.blockers} >= {
        "deployment_hardening_not_ready",
        "deployment_hardening_environment_invalid",
        "deployment_observation_window_missing",
        "deployment_telemetry_uri_missing",
        "deployment_sustained_load_packet_missing",
        "deployment_operator_signoff_missing",
        "deployment_async_ordering_failed",
        "deployment_tenant_poisoning_failed",
        "deployment_hardening_hash_invalid",
    }


def test_deployment_hardening_evidence_round_trips_from_json_safe_dict():
    evidence = _deployment_hardening_evidence()

    restored = DeploymentHardeningEvidence.from_dict(evidence.to_dict())

    assert restored == evidence


def test_release_gate_blocks_customer_boundary_mismatch():
    evidence = _evidence_pack()
    evidence = CustomerEvidencePackManifest(
        **{**evidence.to_dict(), "tenant_id": "wrong-tenant"},
    )

    gate = build_release_gate_manifest(
        release_id="release-customer-alpha-mismatch",
        enterprise_ready=True,
        enterprise_blocking_debt_ids=(),
        runtime_package=_runtime_package(),
        evidence_pack=evidence,
        monitoring_manifest=_monitoring_manifest(),
        risk_register=_risk_register(),
        observability_operations_evidence=_observability_operations_evidence(),
        provenance_lineage_evidence=_provenance_lineage_evidence(),
        deployment_hardening_evidence=_deployment_hardening_evidence(),
        generated_at="2026-05-18T18:35:00Z",
    )

    assert gate.ready is False
    assert any(
        blocker["code"] == "tenant_boundary_mismatch" for blocker in gate.blockers
    )


def test_release_gate_blocks_cross_artifact_boundary_and_hash_mismatches():
    evidence = CustomerEvidencePackManifest(
        **{
            **_evidence_pack().to_dict(),
            "customer_id": "wrong-customer",
            "workspace_id": "wrong-workspace",
            "deployment_id": "wrong-deployment",
            "evidence_hash": "f" * 64,
        }
    )
    monitoring = CustomerMonitoringManifest(
        **{
            **_monitoring_manifest().to_dict(),
            "evidence_hash": "g" * 64,
            "monitoring_hash": "h" * 64,
        }
    )
    risk_register = CustomerRiskRegister(
        **{
            **_risk_register().to_dict(),
            "evidence_hash": "i" * 64,
            "monitoring_hash": "j" * 64,
        }
    )

    gate = build_release_gate_manifest(
        release_id="release-customer-alpha-cross-boundary",
        enterprise_ready=True,
        enterprise_blocking_debt_ids=(),
        runtime_package=_runtime_package(),
        evidence_pack=evidence,
        monitoring_manifest=monitoring,
        risk_register=risk_register,
        observability_operations_evidence=_observability_operations_evidence(),
        provenance_lineage_evidence=_provenance_lineage_evidence(),
        deployment_hardening_evidence=_deployment_hardening_evidence(),
        generated_at="2026-05-18T18:35:00Z",
    )

    assert gate.ready is False
    assert {blocker["code"] for blocker in gate.blockers} >= {
        "customer_boundary_mismatch",
        "workspace_boundary_mismatch",
        "deployment_boundary_mismatch",
        "runtime_evidence_hash_mismatch",
        "monitoring_evidence_hash_mismatch",
        "risk_evidence_hash_mismatch",
        "risk_monitoring_hash_mismatch",
    }


def test_release_gate_serialises_deterministically(tmp_path: Path):
    gate = build_release_gate_manifest(
        release_id="release-customer-alpha-20260518",
        enterprise_ready=True,
        enterprise_blocking_debt_ids=(),
        runtime_package=_runtime_package(),
        evidence_pack=_evidence_pack(),
        monitoring_manifest=_monitoring_manifest(),
        risk_register=_risk_register(),
        observability_operations_evidence=_observability_operations_evidence(),
        provenance_lineage_evidence=_provenance_lineage_evidence(),
        deployment_hardening_evidence=_deployment_hardening_evidence(),
        generated_at="2026-05-18T18:35:00Z",
    )

    output = gate.write_json(tmp_path / "release_gate.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload == gate.to_dict()
    assert payload["observability_operations_evidence"][
        "operations_packet_uri"
    ].endswith("/observability/operations-packet.json")
    assert payload["provenance_lineage_evidence"][
        "signed_lineage_packet_uri"
    ].endswith("/provenance/signed-lineage.json")
    assert payload["deployment_hardening_evidence"]["telemetry_uri"].endswith(
        "/telemetry/r17.jsonl"
    )
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
        "observability_operations_evidence",
        "provenance_lineage_evidence",
        "deployment_hardening_evidence",
        "blockers",
        "release_hash",
    }
