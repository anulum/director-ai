# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory release CLI tests

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
    AutoRedteamDefenceEvidence,
    ConformalRoutingEvidence,
    DeploymentHardeningEvidence,
    EdgeMobileEvidence,
    FederatedPrivacyEvidence,
    MultimodalTemporalEvidence,
    ObservabilityOperationsEvidence,
    ProvenanceLineageEvidence,
    TrajectoryRollbackEvidence,
)
from director_ai.core.customer_model_factory.risk_register import CustomerRiskRegister
from director_ai.core.customer_model_factory.runtime_package import (
    CustomerRuntimePackage,
)
from tools.assemble_customer_model_factory_release import main as assemble_main


def _write_inputs(tmp_path: Path, *, enterprise_ready: bool = True) -> dict[str, Path]:
    runtime = CustomerRuntimePackage(
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
    evidence = CustomerEvidencePackManifest(
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
    monitoring = CustomerMonitoringManifest(
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
    risk = CustomerRiskRegister(
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
    hardening = DeploymentHardeningEvidence(
        ready=True,
        environment="production",
        observation_window="2026-05-18T00:00:00Z/2026-05-18T12:00:00Z",
        telemetry_uri="gs://customer-artifacts/customer-alpha/telemetry/r17.jsonl",
        sustained_load_packet_uri="gs://customer-artifacts/customer-alpha/evidence/sustained-load.json",
        operator_signoff_uri="gs://customer-artifacts/customer-alpha/signoff/r17.json",
        async_ordering_passed=True,
        tenant_poisoning_passed=True,
        evidence_hash="f" * 64,
    )
    observability = ObservabilityOperationsEvidence(
        ready=True,
        environment="production",
        operations_packet_uri="gs://customer-artifacts/customer-alpha/observability/operations-packet.json",
        dashboard_evidence_uri="gs://customer-artifacts/customer-alpha/observability/dashboard-evidence.json",
        operator_signoff_uri="gs://customer-artifacts/customer-alpha/signoff/r8.json",
        controls_passed=True,
        compliance_exports_available=True,
        drift_reviewed=True,
        evidence_hash="1" * 64,
    )
    provenance = ProvenanceLineageEvidence(
        ready=True,
        environment="production",
        feedback_loop_run_uri="gs://customer-artifacts/customer-alpha/provenance/live-feedback-run.json",
        signed_lineage_packet_uri="gs://customer-artifacts/customer-alpha/provenance/signed-lineage.json",
        tenant_kb_snapshot_uri="gs://customer-artifacts/customer-alpha/provenance/tenant-kb-snapshot.json",
        operator_signoff_uri="gs://customer-artifacts/customer-alpha/signoff/r9.json",
        lineage_matches_deployed_facts=True,
        protected_claim_conflicts_resolved=True,
        evidence_hash="2" * 64,
    )
    conformal = ConformalRoutingEvidence(
        ready=True,
        environment="production",
        domain_calibration_packet_uri="gs://customer-artifacts/customer-alpha/conformal/domain-calibration.json",
        deployment_routing_packet_uri="gs://customer-artifacts/customer-alpha/conformal/deployment-routing.json",
        escalation_route="human_review",
        operator_signoff_uri="gs://customer-artifacts/customer-alpha/signoff/r10.json",
        target_coverage=0.95,
        empirical_coverage=0.97,
        calibration_sample_count=240,
        escalation_route_verified=True,
        reject_to_human_available=True,
        evidence_hash="3" * 64,
    )
    trajectory = TrajectoryRollbackEvidence(
        ready=True,
        environment="production",
        simulation_evidence_uri="gs://customer-artifacts/customer-alpha/trajectory/simulation-evidence.json",
        live_undo_backend_uri="gs://customer-artifacts/customer-alpha/trajectory/live-undo-backend.json",
        adversarial_stress_packet_uri="gs://customer-artifacts/customer-alpha/trajectory/adversarial-stress.json",
        incident_change_record_uri="gs://customer-artifacts/customer-alpha/incidents/change-record-r11.json",
        operator_signoff_uri="gs://customer-artifacts/customer-alpha/signoff/r11.json",
        rollback_hook_verified=True,
        idempotency_verified=True,
        tenant_safe_audit_verified=True,
        evidence_hash="4" * 64,
    )
    multimodal = MultimodalTemporalEvidence(
        ready=True,
        environment="production",
        vision_nli_benchmark_uri="gs://customer-artifacts/customer-alpha/multimodal/vision-nli-benchmark.json",
        video_frame_validation_uri="gs://customer-artifacts/customer-alpha/multimodal/video-frame-validation.json",
        modality_coverage_uri="gs://customer-artifacts/customer-alpha/multimodal/modality-coverage.json",
        operator_signoff_uri="gs://customer-artifacts/customer-alpha/signoff/r12.json",
        image_guard_verified=True,
        audio_guard_verified=True,
        video_temporal_verified=True,
        caption_grounding_verified=True,
        deployment_modalities_covered=True,
        evidence_hash="5" * 64,
    )
    federated = FederatedPrivacyEvidence(
        ready=True,
        environment="production",
        external_federation_run_uri="gs://customer-artifacts/customer-alpha/federated/external-federation-run.json",
        malicious_secure_review_uri="gs://customer-artifacts/customer-alpha/federated/malicious-secure-review.json",
        poisoning_resilience_packet_uri="gs://customer-artifacts/customer-alpha/federated/poisoning-resilience.json",
        privacy_budget_ledger_uri="gs://customer-artifacts/customer-alpha/federated/privacy-budget-ledger.json",
        operator_signoff_uri="gs://customer-artifacts/customer-alpha/signoff/r13.json",
        dp_aggregation_verified=True,
        cohort_gate_verified=True,
        secret_sharing_verified=True,
        contribution_caps_verified=True,
        evidence_hash="6" * 64,
    )
    edge_mobile = EdgeMobileEvidence(
        ready=True,
        environment="production",
        edge_runtime_evidence_uri="gs://customer-artifacts/customer-alpha/edge/edge-mobile-evidence.json",
        quantised_model_artifact_uri="gs://customer-artifacts/customer-alpha/edge/models/tiny-nli-int8.onnx",
        wasm_package_evidence_uri="gs://customer-artifacts/customer-alpha/edge/wasm-release-package.json",
        browser_worker_smoke_uri="gs://customer-artifacts/customer-alpha/edge/browser-worker-smoke.json",
        mobile_smoke_evidence_uri="gs://customer-artifacts/customer-alpha/edge/mobile-device-smoke.json",
        package_publish_evidence_uri="gs://customer-artifacts/customer-alpha/edge/package-publish.json",
        latency_profile_uri="gs://customer-artifacts/customer-alpha/edge/latency-profile.json",
        operator_signoff_uri="gs://customer-artifacts/customer-alpha/signoff/r14.json",
        quantised_model_verified=True,
        wasm_package_verified=True,
        browser_worker_smoke_passed=True,
        mobile_or_embedded_smoke_passed=True,
        latency_budget_met=True,
        package_publish_verified=True,
        evidence_hash="7" * 64,
    )
    auto_redteam = AutoRedteamDefenceEvidence(
        ready=True,
        environment="production",
        nightly_run_uri="gs://customer-artifacts/customer-alpha/redteam/nightly-run.json",
        defence_update_packet_uri="gs://customer-artifacts/customer-alpha/redteam/defence-update-packet.json",
        registry_snapshot_uri="gs://customer-artifacts/customer-alpha/redteam/registry-snapshot.json",
        external_adversarial_corpus_uri="gs://customer-artifacts/customer-alpha/redteam/external-adversarial-corpus.json",
        patch_integration_signoff_uri="gs://customer-artifacts/customer-alpha/redteam/patch-integration-signoff.json",
        rollback_plan_uri="gs://customer-artifacts/customer-alpha/redteam/rollback-plan.json",
        operator_signoff_uri="gs://customer-artifacts/customer-alpha/signoff/r15.json",
        repeated_cycles_verified=True,
        detection_uplift_verified=True,
        registry_promotions_verified=True,
        tenant_safe_reports_verified=True,
        rollback_plan_verified=True,
        evidence_hash="8" * 64,
    )
    paths = {
        "runtime": tmp_path / "runtime.json",
        "evidence": tmp_path / "evidence.json",
        "monitoring": tmp_path / "monitoring.json",
        "risk": tmp_path / "risk.json",
        "observability": tmp_path / "observability_operations.json",
        "provenance": tmp_path / "provenance_lineage.json",
        "conformal": tmp_path / "conformal_routing.json",
        "trajectory": tmp_path / "trajectory_rollback.json",
        "multimodal": tmp_path / "multimodal_temporal.json",
        "federated": tmp_path / "federated_privacy.json",
        "edge_mobile": tmp_path / "edge_mobile.json",
        "auto_redteam": tmp_path / "auto_redteam_defence.json",
        "hardening": tmp_path / "deployment_hardening.json",
        "enterprise": tmp_path / "enterprise.json",
        "output": tmp_path / "release_gate.json",
    }
    runtime.write_json(paths["runtime"])
    evidence.write_json(paths["evidence"])
    monitoring.write_json(paths["monitoring"])
    risk.write_json(paths["risk"])
    paths["hardening"].write_text(
        json.dumps(hardening.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    paths["observability"].write_text(
        json.dumps(observability.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    paths["provenance"].write_text(
        json.dumps(provenance.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    paths["conformal"].write_text(
        json.dumps(conformal.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    paths["trajectory"].write_text(
        json.dumps(trajectory.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    paths["multimodal"].write_text(
        json.dumps(multimodal.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    paths["federated"].write_text(
        json.dumps(federated.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    paths["edge_mobile"].write_text(
        json.dumps(edge_mobile.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    paths["auto_redteam"].write_text(
        json.dumps(auto_redteam.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    paths["enterprise"].write_text(
        json.dumps(
            {
                "ready": enterprise_ready,
                "blocking_debt_ids": [] if enterprise_ready else ["TRUST-DEBT-0002"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return paths


def test_release_cli_writes_ready_release_gate(tmp_path: Path):
    paths = _write_inputs(tmp_path)

    exit_code = assemble_main(
        [
            "--release-id",
            "release-customer-alpha-20260518",
            "--generated-at",
            "2026-05-18T18:45:00Z",
            "--enterprise-readiness",
            str(paths["enterprise"]),
            "--runtime-package",
            str(paths["runtime"]),
            "--evidence-pack",
            str(paths["evidence"]),
            "--monitoring-manifest",
            str(paths["monitoring"]),
            "--risk-register",
            str(paths["risk"]),
            "--observability-operations-evidence",
            str(paths["observability"]),
            "--provenance-lineage-evidence",
            str(paths["provenance"]),
            "--conformal-routing-evidence",
            str(paths["conformal"]),
            "--trajectory-rollback-evidence",
            str(paths["trajectory"]),
            "--multimodal-temporal-evidence",
            str(paths["multimodal"]),
            "--federated-privacy-evidence",
            str(paths["federated"]),
            "--edge-mobile-evidence",
            str(paths["edge_mobile"]),
            "--auto-redteam-defence-evidence",
            str(paths["auto_redteam"]),
            "--deployment-hardening-evidence",
            str(paths["hardening"]),
            "--output",
            str(paths["output"]),
        ]
    )
    payload = json.loads(paths["output"].read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["promotion_allowed"] is True
    assert payload["blockers"] == []
    assert payload["artifact_hashes"]["risk_register_hash"] == "e" * 64
    assert payload["observability_operations_evidence"]["environment"] == "production"
    assert payload["provenance_lineage_evidence"]["environment"] == "production"
    assert payload["conformal_routing_evidence"]["environment"] == "production"
    assert payload["trajectory_rollback_evidence"]["environment"] == "production"
    assert payload["multimodal_temporal_evidence"]["environment"] == "production"
    assert payload["federated_privacy_evidence"]["environment"] == "production"
    assert payload["edge_mobile_evidence"]["environment"] == "production"
    assert payload["auto_redteam_defence_evidence"]["environment"] == "production"
    assert payload["deployment_hardening_evidence"]["environment"] == "production"


def test_release_cli_fails_closed_when_enterprise_readiness_blocks(tmp_path: Path):
    paths = _write_inputs(tmp_path, enterprise_ready=False)

    exit_code = assemble_main(
        [
            "--release-id",
            "release-customer-alpha-blocked",
            "--generated-at",
            "2026-05-18T18:45:00Z",
            "--enterprise-readiness",
            str(paths["enterprise"]),
            "--runtime-package",
            str(paths["runtime"]),
            "--evidence-pack",
            str(paths["evidence"]),
            "--monitoring-manifest",
            str(paths["monitoring"]),
            "--risk-register",
            str(paths["risk"]),
            "--observability-operations-evidence",
            str(paths["observability"]),
            "--provenance-lineage-evidence",
            str(paths["provenance"]),
            "--conformal-routing-evidence",
            str(paths["conformal"]),
            "--trajectory-rollback-evidence",
            str(paths["trajectory"]),
            "--multimodal-temporal-evidence",
            str(paths["multimodal"]),
            "--federated-privacy-evidence",
            str(paths["federated"]),
            "--edge-mobile-evidence",
            str(paths["edge_mobile"]),
            "--auto-redteam-defence-evidence",
            str(paths["auto_redteam"]),
            "--deployment-hardening-evidence",
            str(paths["hardening"]),
            "--output",
            str(paths["output"]),
        ]
    )
    payload = json.loads(paths["output"].read_text(encoding="utf-8"))

    assert exit_code == 1
    assert payload["promotion_allowed"] is False
    assert any(
        blocker["code"] == "enterprise_trust_not_ready"
        for blocker in payload["blockers"]
    )
