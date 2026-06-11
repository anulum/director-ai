# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory release gate

"""Final promotion gate for Customer Model Factory packages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evidence_pack import CustomerEvidencePackManifest
from .monitoring_manifest import CustomerMonitoringManifest
from .risk_register import CustomerRiskRegister
from .runtime_package import CustomerRuntimePackage

SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class DeploymentHardeningEvidence:
    """Sustained deployment-hardening evidence attached to a release gate."""

    ready: bool
    environment: str
    observation_window: str
    telemetry_uri: str
    sustained_load_packet_uri: str
    operator_signoff_uri: str
    async_ordering_passed: bool
    tenant_poisoning_passed: bool
    evidence_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise deployment-hardening evidence to stable JSON-safe data."""

        return {
            "ready": self.ready,
            "environment": self.environment,
            "observation_window": self.observation_window,
            "telemetry_uri": self.telemetry_uri,
            "sustained_load_packet_uri": self.sustained_load_packet_uri,
            "operator_signoff_uri": self.operator_signoff_uri,
            "async_ordering_passed": self.async_ordering_passed,
            "tenant_poisoning_passed": self.tenant_poisoning_passed,
            "evidence_hash": self.evidence_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DeploymentHardeningEvidence:
        """Rebuild deployment-hardening evidence from JSON-safe data."""

        return cls(
            ready=bool(payload["ready"]),
            environment=str(payload["environment"]),
            observation_window=str(payload["observation_window"]),
            telemetry_uri=str(payload["telemetry_uri"]),
            sustained_load_packet_uri=str(payload["sustained_load_packet_uri"]),
            operator_signoff_uri=str(payload["operator_signoff_uri"]),
            async_ordering_passed=bool(payload["async_ordering_passed"]),
            tenant_poisoning_passed=bool(payload["tenant_poisoning_passed"]),
            evidence_hash=str(payload["evidence_hash"]),
        )


@dataclass(frozen=True)
class ObservabilityOperationsEvidence:
    """Operations-dashboard evidence attached to a release gate."""

    ready: bool
    environment: str
    operations_packet_uri: str
    dashboard_evidence_uri: str
    operator_signoff_uri: str
    controls_passed: bool
    compliance_exports_available: bool
    drift_reviewed: bool
    evidence_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise observability operations evidence to JSON-safe data."""

        return {
            "ready": self.ready,
            "environment": self.environment,
            "operations_packet_uri": self.operations_packet_uri,
            "dashboard_evidence_uri": self.dashboard_evidence_uri,
            "operator_signoff_uri": self.operator_signoff_uri,
            "controls_passed": self.controls_passed,
            "compliance_exports_available": self.compliance_exports_available,
            "drift_reviewed": self.drift_reviewed,
            "evidence_hash": self.evidence_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ObservabilityOperationsEvidence:
        """Rebuild observability operations evidence from JSON-safe data."""

        return cls(
            ready=bool(payload["ready"]),
            environment=str(payload["environment"]),
            operations_packet_uri=str(payload["operations_packet_uri"]),
            dashboard_evidence_uri=str(payload["dashboard_evidence_uri"]),
            operator_signoff_uri=str(payload["operator_signoff_uri"]),
            controls_passed=bool(payload["controls_passed"]),
            compliance_exports_available=bool(payload["compliance_exports_available"]),
            drift_reviewed=bool(payload["drift_reviewed"]),
            evidence_hash=str(payload["evidence_hash"]),
        )


@dataclass(frozen=True)
class ProvenanceLineageEvidence:
    """Online KB provenance and lineage evidence attached to a release gate."""

    ready: bool
    environment: str
    feedback_loop_run_uri: str
    signed_lineage_packet_uri: str
    tenant_kb_snapshot_uri: str
    operator_signoff_uri: str
    lineage_matches_deployed_facts: bool
    protected_claim_conflicts_resolved: bool
    evidence_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise KB provenance-lineage evidence to JSON-safe data."""

        return {
            "ready": self.ready,
            "environment": self.environment,
            "feedback_loop_run_uri": self.feedback_loop_run_uri,
            "signed_lineage_packet_uri": self.signed_lineage_packet_uri,
            "tenant_kb_snapshot_uri": self.tenant_kb_snapshot_uri,
            "operator_signoff_uri": self.operator_signoff_uri,
            "lineage_matches_deployed_facts": self.lineage_matches_deployed_facts,
            "protected_claim_conflicts_resolved": (
                self.protected_claim_conflicts_resolved
            ),
            "evidence_hash": self.evidence_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProvenanceLineageEvidence:
        """Rebuild KB provenance-lineage evidence from JSON-safe data."""

        return cls(
            ready=bool(payload["ready"]),
            environment=str(payload["environment"]),
            feedback_loop_run_uri=str(payload["feedback_loop_run_uri"]),
            signed_lineage_packet_uri=str(payload["signed_lineage_packet_uri"]),
            tenant_kb_snapshot_uri=str(payload["tenant_kb_snapshot_uri"]),
            operator_signoff_uri=str(payload["operator_signoff_uri"]),
            lineage_matches_deployed_facts=bool(
                payload["lineage_matches_deployed_facts"]
            ),
            protected_claim_conflicts_resolved=bool(
                payload["protected_claim_conflicts_resolved"]
            ),
            evidence_hash=str(payload["evidence_hash"]),
        )


@dataclass(frozen=True)
class ConformalRoutingEvidence:
    """Conformal calibration and escalation evidence attached to a release gate."""

    ready: bool
    environment: str
    domain_calibration_packet_uri: str
    deployment_routing_packet_uri: str
    escalation_route: str
    operator_signoff_uri: str
    target_coverage: float
    empirical_coverage: float
    calibration_sample_count: int
    escalation_route_verified: bool
    reject_to_human_available: bool
    evidence_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise conformal routing evidence to JSON-safe data."""

        return {
            "ready": self.ready,
            "environment": self.environment,
            "domain_calibration_packet_uri": self.domain_calibration_packet_uri,
            "deployment_routing_packet_uri": self.deployment_routing_packet_uri,
            "escalation_route": self.escalation_route,
            "operator_signoff_uri": self.operator_signoff_uri,
            "target_coverage": self.target_coverage,
            "empirical_coverage": self.empirical_coverage,
            "calibration_sample_count": self.calibration_sample_count,
            "escalation_route_verified": self.escalation_route_verified,
            "reject_to_human_available": self.reject_to_human_available,
            "evidence_hash": self.evidence_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConformalRoutingEvidence:
        """Rebuild conformal routing evidence from JSON-safe data."""

        return cls(
            ready=bool(payload["ready"]),
            environment=str(payload["environment"]),
            domain_calibration_packet_uri=str(payload["domain_calibration_packet_uri"]),
            deployment_routing_packet_uri=str(payload["deployment_routing_packet_uri"]),
            escalation_route=str(payload["escalation_route"]),
            operator_signoff_uri=str(payload["operator_signoff_uri"]),
            target_coverage=float(payload["target_coverage"]),
            empirical_coverage=float(payload["empirical_coverage"]),
            calibration_sample_count=int(payload["calibration_sample_count"]),
            escalation_route_verified=bool(payload["escalation_route_verified"]),
            reject_to_human_available=bool(payload["reject_to_human_available"]),
            evidence_hash=str(payload["evidence_hash"]),
        )


@dataclass(frozen=True)
class TrajectoryRollbackEvidence:
    """Trajectory simulation and rollback evidence attached to a release gate."""

    ready: bool
    environment: str
    simulation_evidence_uri: str
    live_undo_backend_uri: str
    adversarial_stress_packet_uri: str
    incident_change_record_uri: str
    operator_signoff_uri: str
    rollback_hook_verified: bool
    idempotency_verified: bool
    tenant_safe_audit_verified: bool
    evidence_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise trajectory rollback evidence to JSON-safe data."""

        return {
            "ready": self.ready,
            "environment": self.environment,
            "simulation_evidence_uri": self.simulation_evidence_uri,
            "live_undo_backend_uri": self.live_undo_backend_uri,
            "adversarial_stress_packet_uri": self.adversarial_stress_packet_uri,
            "incident_change_record_uri": self.incident_change_record_uri,
            "operator_signoff_uri": self.operator_signoff_uri,
            "rollback_hook_verified": self.rollback_hook_verified,
            "idempotency_verified": self.idempotency_verified,
            "tenant_safe_audit_verified": self.tenant_safe_audit_verified,
            "evidence_hash": self.evidence_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrajectoryRollbackEvidence:
        """Rebuild trajectory rollback evidence from JSON-safe data."""

        return cls(
            ready=bool(payload["ready"]),
            environment=str(payload["environment"]),
            simulation_evidence_uri=str(payload["simulation_evidence_uri"]),
            live_undo_backend_uri=str(payload["live_undo_backend_uri"]),
            adversarial_stress_packet_uri=str(payload["adversarial_stress_packet_uri"]),
            incident_change_record_uri=str(payload["incident_change_record_uri"]),
            operator_signoff_uri=str(payload["operator_signoff_uri"]),
            rollback_hook_verified=bool(payload["rollback_hook_verified"]),
            idempotency_verified=bool(payload["idempotency_verified"]),
            tenant_safe_audit_verified=bool(payload["tenant_safe_audit_verified"]),
            evidence_hash=str(payload["evidence_hash"]),
        )


@dataclass(frozen=True)
class MultimodalTemporalEvidence:
    """Multimodal and temporal consistency evidence attached to a release gate."""

    ready: bool
    environment: str
    vision_nli_benchmark_uri: str
    video_frame_validation_uri: str
    modality_coverage_uri: str
    operator_signoff_uri: str
    image_guard_verified: bool
    audio_guard_verified: bool
    video_temporal_verified: bool
    caption_grounding_verified: bool
    deployment_modalities_covered: bool
    evidence_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise multimodal temporal evidence to JSON-safe data."""

        return {
            "ready": self.ready,
            "environment": self.environment,
            "vision_nli_benchmark_uri": self.vision_nli_benchmark_uri,
            "video_frame_validation_uri": self.video_frame_validation_uri,
            "modality_coverage_uri": self.modality_coverage_uri,
            "operator_signoff_uri": self.operator_signoff_uri,
            "image_guard_verified": self.image_guard_verified,
            "audio_guard_verified": self.audio_guard_verified,
            "video_temporal_verified": self.video_temporal_verified,
            "caption_grounding_verified": self.caption_grounding_verified,
            "deployment_modalities_covered": self.deployment_modalities_covered,
            "evidence_hash": self.evidence_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MultimodalTemporalEvidence:
        """Rebuild multimodal temporal evidence from JSON-safe data."""

        return cls(
            ready=bool(payload["ready"]),
            environment=str(payload["environment"]),
            vision_nli_benchmark_uri=str(payload["vision_nli_benchmark_uri"]),
            video_frame_validation_uri=str(payload["video_frame_validation_uri"]),
            modality_coverage_uri=str(payload["modality_coverage_uri"]),
            operator_signoff_uri=str(payload["operator_signoff_uri"]),
            image_guard_verified=bool(payload["image_guard_verified"]),
            audio_guard_verified=bool(payload["audio_guard_verified"]),
            video_temporal_verified=bool(payload["video_temporal_verified"]),
            caption_grounding_verified=bool(payload["caption_grounding_verified"]),
            deployment_modalities_covered=bool(
                payload["deployment_modalities_covered"]
            ),
            evidence_hash=str(payload["evidence_hash"]),
        )


@dataclass(frozen=True)
class FederatedPrivacyEvidence:
    """Federated privacy evidence attached to a release gate."""

    ready: bool
    environment: str
    external_federation_run_uri: str
    malicious_secure_review_uri: str
    poisoning_resilience_packet_uri: str
    privacy_budget_ledger_uri: str
    operator_signoff_uri: str
    dp_aggregation_verified: bool
    cohort_gate_verified: bool
    secret_sharing_verified: bool
    contribution_caps_verified: bool
    evidence_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise federated privacy evidence to JSON-safe data."""

        return {
            "ready": self.ready,
            "environment": self.environment,
            "external_federation_run_uri": self.external_federation_run_uri,
            "malicious_secure_review_uri": self.malicious_secure_review_uri,
            "poisoning_resilience_packet_uri": self.poisoning_resilience_packet_uri,
            "privacy_budget_ledger_uri": self.privacy_budget_ledger_uri,
            "operator_signoff_uri": self.operator_signoff_uri,
            "dp_aggregation_verified": self.dp_aggregation_verified,
            "cohort_gate_verified": self.cohort_gate_verified,
            "secret_sharing_verified": self.secret_sharing_verified,
            "contribution_caps_verified": self.contribution_caps_verified,
            "evidence_hash": self.evidence_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FederatedPrivacyEvidence:
        """Rebuild federated privacy evidence from JSON-safe data."""

        return cls(
            ready=bool(payload["ready"]),
            environment=str(payload["environment"]),
            external_federation_run_uri=str(payload["external_federation_run_uri"]),
            malicious_secure_review_uri=str(payload["malicious_secure_review_uri"]),
            poisoning_resilience_packet_uri=str(
                payload["poisoning_resilience_packet_uri"]
            ),
            privacy_budget_ledger_uri=str(payload["privacy_budget_ledger_uri"]),
            operator_signoff_uri=str(payload["operator_signoff_uri"]),
            dp_aggregation_verified=bool(payload["dp_aggregation_verified"]),
            cohort_gate_verified=bool(payload["cohort_gate_verified"]),
            secret_sharing_verified=bool(payload["secret_sharing_verified"]),
            contribution_caps_verified=bool(payload["contribution_caps_verified"]),
            evidence_hash=str(payload["evidence_hash"]),
        )


@dataclass(frozen=True)
class EdgeMobileEvidence:
    """Edge/mobile runtime evidence attached to a release gate."""

    ready: bool
    environment: str
    edge_runtime_evidence_uri: str
    quantised_model_artifact_uri: str
    wasm_package_evidence_uri: str
    browser_worker_smoke_uri: str
    mobile_smoke_evidence_uri: str
    package_publish_evidence_uri: str
    latency_profile_uri: str
    operator_signoff_uri: str
    quantised_model_verified: bool
    wasm_package_verified: bool
    browser_worker_smoke_passed: bool
    mobile_or_embedded_smoke_passed: bool
    latency_budget_met: bool
    package_publish_verified: bool
    evidence_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise edge/mobile evidence to JSON-safe data."""

        return {
            "ready": self.ready,
            "environment": self.environment,
            "edge_runtime_evidence_uri": self.edge_runtime_evidence_uri,
            "quantised_model_artifact_uri": self.quantised_model_artifact_uri,
            "wasm_package_evidence_uri": self.wasm_package_evidence_uri,
            "browser_worker_smoke_uri": self.browser_worker_smoke_uri,
            "mobile_smoke_evidence_uri": self.mobile_smoke_evidence_uri,
            "package_publish_evidence_uri": self.package_publish_evidence_uri,
            "latency_profile_uri": self.latency_profile_uri,
            "operator_signoff_uri": self.operator_signoff_uri,
            "quantised_model_verified": self.quantised_model_verified,
            "wasm_package_verified": self.wasm_package_verified,
            "browser_worker_smoke_passed": self.browser_worker_smoke_passed,
            "mobile_or_embedded_smoke_passed": (self.mobile_or_embedded_smoke_passed),
            "latency_budget_met": self.latency_budget_met,
            "package_publish_verified": self.package_publish_verified,
            "evidence_hash": self.evidence_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EdgeMobileEvidence:
        """Rebuild edge/mobile evidence from JSON-safe data."""

        return cls(
            ready=bool(payload["ready"]),
            environment=str(payload["environment"]),
            edge_runtime_evidence_uri=str(payload["edge_runtime_evidence_uri"]),
            quantised_model_artifact_uri=str(payload["quantised_model_artifact_uri"]),
            wasm_package_evidence_uri=str(payload["wasm_package_evidence_uri"]),
            browser_worker_smoke_uri=str(payload["browser_worker_smoke_uri"]),
            mobile_smoke_evidence_uri=str(payload["mobile_smoke_evidence_uri"]),
            package_publish_evidence_uri=str(payload["package_publish_evidence_uri"]),
            latency_profile_uri=str(payload["latency_profile_uri"]),
            operator_signoff_uri=str(payload["operator_signoff_uri"]),
            quantised_model_verified=bool(payload["quantised_model_verified"]),
            wasm_package_verified=bool(payload["wasm_package_verified"]),
            browser_worker_smoke_passed=bool(payload["browser_worker_smoke_passed"]),
            mobile_or_embedded_smoke_passed=bool(
                payload["mobile_or_embedded_smoke_passed"]
            ),
            latency_budget_met=bool(payload["latency_budget_met"]),
            package_publish_verified=bool(payload["package_publish_verified"]),
            evidence_hash=str(payload["evidence_hash"]),
        )


@dataclass(frozen=True)
class AutoRedteamDefenceEvidence:
    """Auto-redteam and defence-genome evidence attached to a release gate."""

    ready: bool
    environment: str
    nightly_run_uri: str
    defence_update_packet_uri: str
    registry_snapshot_uri: str
    external_adversarial_corpus_uri: str
    patch_integration_signoff_uri: str
    rollback_plan_uri: str
    operator_signoff_uri: str
    repeated_cycles_verified: bool
    detection_uplift_verified: bool
    registry_promotions_verified: bool
    tenant_safe_reports_verified: bool
    rollback_plan_verified: bool
    evidence_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise auto-redteam defence evidence to JSON-safe data."""

        return {
            "ready": self.ready,
            "environment": self.environment,
            "nightly_run_uri": self.nightly_run_uri,
            "defence_update_packet_uri": self.defence_update_packet_uri,
            "registry_snapshot_uri": self.registry_snapshot_uri,
            "external_adversarial_corpus_uri": self.external_adversarial_corpus_uri,
            "patch_integration_signoff_uri": self.patch_integration_signoff_uri,
            "rollback_plan_uri": self.rollback_plan_uri,
            "operator_signoff_uri": self.operator_signoff_uri,
            "repeated_cycles_verified": self.repeated_cycles_verified,
            "detection_uplift_verified": self.detection_uplift_verified,
            "registry_promotions_verified": self.registry_promotions_verified,
            "tenant_safe_reports_verified": self.tenant_safe_reports_verified,
            "rollback_plan_verified": self.rollback_plan_verified,
            "evidence_hash": self.evidence_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AutoRedteamDefenceEvidence:
        """Rebuild auto-redteam defence evidence from JSON-safe data."""

        return cls(
            ready=bool(payload["ready"]),
            environment=str(payload["environment"]),
            nightly_run_uri=str(payload["nightly_run_uri"]),
            defence_update_packet_uri=str(payload["defence_update_packet_uri"]),
            registry_snapshot_uri=str(payload["registry_snapshot_uri"]),
            external_adversarial_corpus_uri=str(
                payload["external_adversarial_corpus_uri"]
            ),
            patch_integration_signoff_uri=str(payload["patch_integration_signoff_uri"]),
            rollback_plan_uri=str(payload["rollback_plan_uri"]),
            operator_signoff_uri=str(payload["operator_signoff_uri"]),
            repeated_cycles_verified=bool(payload["repeated_cycles_verified"]),
            detection_uplift_verified=bool(payload["detection_uplift_verified"]),
            registry_promotions_verified=bool(payload["registry_promotions_verified"]),
            tenant_safe_reports_verified=bool(payload["tenant_safe_reports_verified"]),
            rollback_plan_verified=bool(payload["rollback_plan_verified"]),
            evidence_hash=str(payload["evidence_hash"]),
        )


@dataclass(frozen=True)
class FormalSymbolicEvidence:
    """Formal and symbolic verification evidence attached to a release gate."""

    ready: bool
    environment: str
    formal_symbolic_packet_uri: str
    external_lean_proof_uri: str
    z3_release_packet_uri: str
    domain_contracts_uri: str
    code_contract_packet_uri: str
    operator_signoff_uri: str
    dpll_formula_guard_verified: bool
    lean_external_run_verified: bool
    z3_actual_run_verified: bool
    code_contract_ordering_verified: bool
    tenant_safe_serialisation_verified: bool
    domain_contracts_verified: bool
    evidence_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise formal-symbolic evidence to JSON-safe data."""

        return {
            "ready": self.ready,
            "environment": self.environment,
            "formal_symbolic_packet_uri": self.formal_symbolic_packet_uri,
            "external_lean_proof_uri": self.external_lean_proof_uri,
            "z3_release_packet_uri": self.z3_release_packet_uri,
            "domain_contracts_uri": self.domain_contracts_uri,
            "code_contract_packet_uri": self.code_contract_packet_uri,
            "operator_signoff_uri": self.operator_signoff_uri,
            "dpll_formula_guard_verified": self.dpll_formula_guard_verified,
            "lean_external_run_verified": self.lean_external_run_verified,
            "z3_actual_run_verified": self.z3_actual_run_verified,
            "code_contract_ordering_verified": (self.code_contract_ordering_verified),
            "tenant_safe_serialisation_verified": (
                self.tenant_safe_serialisation_verified
            ),
            "domain_contracts_verified": self.domain_contracts_verified,
            "evidence_hash": self.evidence_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FormalSymbolicEvidence:
        """Rebuild formal-symbolic evidence from JSON-safe data."""

        return cls(
            ready=bool(payload["ready"]),
            environment=str(payload["environment"]),
            formal_symbolic_packet_uri=str(payload["formal_symbolic_packet_uri"]),
            external_lean_proof_uri=str(payload["external_lean_proof_uri"]),
            z3_release_packet_uri=str(payload["z3_release_packet_uri"]),
            domain_contracts_uri=str(payload["domain_contracts_uri"]),
            code_contract_packet_uri=str(payload["code_contract_packet_uri"]),
            operator_signoff_uri=str(payload["operator_signoff_uri"]),
            dpll_formula_guard_verified=bool(payload["dpll_formula_guard_verified"]),
            lean_external_run_verified=bool(payload["lean_external_run_verified"]),
            z3_actual_run_verified=bool(payload["z3_actual_run_verified"]),
            code_contract_ordering_verified=bool(
                payload["code_contract_ordering_verified"]
            ),
            tenant_safe_serialisation_verified=bool(
                payload["tenant_safe_serialisation_verified"]
            ),
            domain_contracts_verified=bool(payload["domain_contracts_verified"]),
            evidence_hash=str(payload["evidence_hash"]),
        )


@dataclass(frozen=True)
class CustomerReleaseGateManifest:
    """Release-promotion gate across factory readiness artefacts."""

    schema_version: str
    release_id: str
    ready: bool
    promotion_allowed: bool
    customer_id: str
    workspace_id: str
    tenant_id: str
    deployment_id: str
    generated_at: str
    enterprise_ready: bool
    enterprise_blocking_debt_ids: tuple[str, ...]
    artifact_hashes: dict[str, str]
    observability_operations_evidence: ObservabilityOperationsEvidence
    provenance_lineage_evidence: ProvenanceLineageEvidence
    conformal_routing_evidence: ConformalRoutingEvidence
    trajectory_rollback_evidence: TrajectoryRollbackEvidence
    multimodal_temporal_evidence: MultimodalTemporalEvidence
    federated_privacy_evidence: FederatedPrivacyEvidence
    edge_mobile_evidence: EdgeMobileEvidence
    auto_redteam_defence_evidence: AutoRedteamDefenceEvidence
    formal_symbolic_evidence: FormalSymbolicEvidence
    deployment_hardening_evidence: DeploymentHardeningEvidence
    blockers: tuple[dict[str, str], ...]
    release_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise the release gate to stable JSON-safe data."""

        return {
            "schema_version": self.schema_version,
            "release_id": self.release_id,
            "ready": self.ready,
            "promotion_allowed": self.promotion_allowed,
            "customer_id": self.customer_id,
            "workspace_id": self.workspace_id,
            "tenant_id": self.tenant_id,
            "deployment_id": self.deployment_id,
            "generated_at": self.generated_at,
            "enterprise_ready": self.enterprise_ready,
            "enterprise_blocking_debt_ids": list(self.enterprise_blocking_debt_ids),
            "artifact_hashes": dict(sorted(self.artifact_hashes.items())),
            "observability_operations_evidence": (
                self.observability_operations_evidence.to_dict()
            ),
            "provenance_lineage_evidence": (self.provenance_lineage_evidence.to_dict()),
            "conformal_routing_evidence": self.conformal_routing_evidence.to_dict(),
            "trajectory_rollback_evidence": (
                self.trajectory_rollback_evidence.to_dict()
            ),
            "multimodal_temporal_evidence": (
                self.multimodal_temporal_evidence.to_dict()
            ),
            "federated_privacy_evidence": (self.federated_privacy_evidence.to_dict()),
            "edge_mobile_evidence": self.edge_mobile_evidence.to_dict(),
            "auto_redteam_defence_evidence": (
                self.auto_redteam_defence_evidence.to_dict()
            ),
            "formal_symbolic_evidence": self.formal_symbolic_evidence.to_dict(),
            "deployment_hardening_evidence": (
                self.deployment_hardening_evidence.to_dict()
            ),
            "blockers": [dict(blocker) for blocker in self.blockers],
            "release_hash": self.release_hash,
        }

    def write_json(self, path: Path) -> Path:
        """Write the release gate as deterministic JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


def build_release_gate_manifest(
    *,
    release_id: str,
    enterprise_ready: bool,
    enterprise_blocking_debt_ids: tuple[str, ...],
    runtime_package: CustomerRuntimePackage,
    evidence_pack: CustomerEvidencePackManifest,
    monitoring_manifest: CustomerMonitoringManifest,
    risk_register: CustomerRiskRegister,
    observability_operations_evidence: ObservabilityOperationsEvidence,
    provenance_lineage_evidence: ProvenanceLineageEvidence,
    conformal_routing_evidence: ConformalRoutingEvidence,
    trajectory_rollback_evidence: TrajectoryRollbackEvidence,
    multimodal_temporal_evidence: MultimodalTemporalEvidence,
    federated_privacy_evidence: FederatedPrivacyEvidence,
    edge_mobile_evidence: EdgeMobileEvidence,
    auto_redteam_defence_evidence: AutoRedteamDefenceEvidence,
    formal_symbolic_evidence: FormalSymbolicEvidence,
    deployment_hardening_evidence: DeploymentHardeningEvidence,
    generated_at: str,
) -> CustomerReleaseGateManifest:
    """Build the final Customer Model Factory release-promotion gate."""

    artifact_hashes = _artifact_hashes(
        runtime_package=runtime_package,
        evidence_pack=evidence_pack,
        monitoring_manifest=monitoring_manifest,
        risk_register=risk_register,
    )
    blockers = _collect_blockers(
        release_id=release_id,
        enterprise_ready=enterprise_ready,
        enterprise_blocking_debt_ids=enterprise_blocking_debt_ids,
        runtime_package=runtime_package,
        evidence_pack=evidence_pack,
        monitoring_manifest=monitoring_manifest,
        risk_register=risk_register,
        observability_operations_evidence=observability_operations_evidence,
        provenance_lineage_evidence=provenance_lineage_evidence,
        conformal_routing_evidence=conformal_routing_evidence,
        trajectory_rollback_evidence=trajectory_rollback_evidence,
        multimodal_temporal_evidence=multimodal_temporal_evidence,
        federated_privacy_evidence=federated_privacy_evidence,
        edge_mobile_evidence=edge_mobile_evidence,
        auto_redteam_defence_evidence=auto_redteam_defence_evidence,
        formal_symbolic_evidence=formal_symbolic_evidence,
        deployment_hardening_evidence=deployment_hardening_evidence,
        generated_at=generated_at,
    )
    release_hash = _stable_hash(
        {
            "release_id": release_id,
            "customer_id": runtime_package.customer_id,
            "workspace_id": runtime_package.workspace_id,
            "tenant_id": runtime_package.tenant_id,
            "deployment_id": runtime_package.deployment_id,
            "generated_at": generated_at,
            "enterprise_ready": enterprise_ready,
            "enterprise_blocking_debt_ids": enterprise_blocking_debt_ids,
            "artifact_hashes": artifact_hashes,
            "observability_operations_evidence": (
                observability_operations_evidence.to_dict()
            ),
            "provenance_lineage_evidence": provenance_lineage_evidence.to_dict(),
            "conformal_routing_evidence": conformal_routing_evidence.to_dict(),
            "trajectory_rollback_evidence": trajectory_rollback_evidence.to_dict(),
            "multimodal_temporal_evidence": multimodal_temporal_evidence.to_dict(),
            "federated_privacy_evidence": federated_privacy_evidence.to_dict(),
            "edge_mobile_evidence": edge_mobile_evidence.to_dict(),
            "auto_redteam_defence_evidence": auto_redteam_defence_evidence.to_dict(),
            "formal_symbolic_evidence": formal_symbolic_evidence.to_dict(),
            "deployment_hardening_evidence": deployment_hardening_evidence.to_dict(),
            "blockers": blockers,
        }
    )
    ready = not blockers
    return CustomerReleaseGateManifest(
        schema_version=SCHEMA_VERSION,
        release_id=release_id,
        ready=ready,
        promotion_allowed=ready,
        customer_id=runtime_package.customer_id,
        workspace_id=runtime_package.workspace_id,
        tenant_id=runtime_package.tenant_id,
        deployment_id=runtime_package.deployment_id,
        generated_at=generated_at,
        enterprise_ready=enterprise_ready,
        enterprise_blocking_debt_ids=enterprise_blocking_debt_ids,
        artifact_hashes=artifact_hashes,
        observability_operations_evidence=observability_operations_evidence,
        provenance_lineage_evidence=provenance_lineage_evidence,
        conformal_routing_evidence=conformal_routing_evidence,
        trajectory_rollback_evidence=trajectory_rollback_evidence,
        multimodal_temporal_evidence=multimodal_temporal_evidence,
        federated_privacy_evidence=federated_privacy_evidence,
        edge_mobile_evidence=edge_mobile_evidence,
        auto_redteam_defence_evidence=auto_redteam_defence_evidence,
        formal_symbolic_evidence=formal_symbolic_evidence,
        deployment_hardening_evidence=deployment_hardening_evidence,
        blockers=tuple(blockers),
        release_hash=release_hash,
    )


def _artifact_hashes(
    *,
    runtime_package: CustomerRuntimePackage,
    evidence_pack: CustomerEvidencePackManifest,
    monitoring_manifest: CustomerMonitoringManifest,
    risk_register: CustomerRiskRegister,
) -> dict[str, str]:
    return {
        "runtime_hash": runtime_package.runtime_hash,
        "evidence_hash": evidence_pack.evidence_hash,
        "monitoring_hash": monitoring_manifest.monitoring_hash,
        "risk_register_hash": risk_register.register_hash,
    }


def _collect_blockers(
    *,
    release_id: str,
    enterprise_ready: bool,
    enterprise_blocking_debt_ids: tuple[str, ...],
    runtime_package: CustomerRuntimePackage,
    evidence_pack: CustomerEvidencePackManifest,
    monitoring_manifest: CustomerMonitoringManifest,
    risk_register: CustomerRiskRegister,
    observability_operations_evidence: ObservabilityOperationsEvidence,
    provenance_lineage_evidence: ProvenanceLineageEvidence,
    conformal_routing_evidence: ConformalRoutingEvidence,
    trajectory_rollback_evidence: TrajectoryRollbackEvidence,
    multimodal_temporal_evidence: MultimodalTemporalEvidence,
    federated_privacy_evidence: FederatedPrivacyEvidence,
    edge_mobile_evidence: EdgeMobileEvidence,
    auto_redteam_defence_evidence: AutoRedteamDefenceEvidence,
    formal_symbolic_evidence: FormalSymbolicEvidence,
    deployment_hardening_evidence: DeploymentHardeningEvidence,
    generated_at: str,
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    if not release_id.strip():
        blockers.append(_blocker("release_id_missing", "release_id is required"))
    if not generated_at.strip():
        blockers.append(_blocker("generated_at_missing", "generated_at is required"))
    if not enterprise_ready or enterprise_blocking_debt_ids:
        blockers.append(
            _blocker(
                "enterprise_trust_not_ready",
                "enterprise-trust readiness has blocking debt",
                debt_ids=",".join(enterprise_blocking_debt_ids),
            )
        )
    _extend_readiness_blockers(
        runtime_package, evidence_pack, monitoring_manifest, risk_register, blockers
    )
    _extend_observability_operations_blockers(
        observability_operations_evidence, blockers
    )
    _extend_provenance_lineage_blockers(provenance_lineage_evidence, blockers)
    _extend_conformal_routing_blockers(conformal_routing_evidence, blockers)
    _extend_trajectory_rollback_blockers(trajectory_rollback_evidence, blockers)
    _extend_multimodal_temporal_blockers(multimodal_temporal_evidence, blockers)
    _extend_federated_privacy_blockers(federated_privacy_evidence, blockers)
    _extend_edge_mobile_blockers(edge_mobile_evidence, blockers)
    _extend_auto_redteam_defence_blockers(auto_redteam_defence_evidence, blockers)
    _extend_formal_symbolic_blockers(formal_symbolic_evidence, blockers)
    _extend_deployment_hardening_blockers(deployment_hardening_evidence, blockers)
    _extend_boundary_blockers(
        runtime_package, evidence_pack, monitoring_manifest, risk_register, blockers
    )
    return blockers


def _extend_readiness_blockers(
    runtime_package: CustomerRuntimePackage,
    evidence_pack: CustomerEvidencePackManifest,
    monitoring_manifest: CustomerMonitoringManifest,
    risk_register: CustomerRiskRegister,
    blockers: list[dict[str, str]],
) -> None:
    if not runtime_package.ready:
        blockers.append(
            _blocker("runtime_package_not_ready", "runtime package is not ready")
        )
    if not evidence_pack.ready:
        blockers.append(
            _blocker("evidence_pack_not_ready", "evidence pack is not ready")
        )
    if not monitoring_manifest.ready:
        blockers.append(
            _blocker(
                "monitoring_manifest_not_ready", "monitoring manifest is not ready"
            )
        )
    if not risk_register.ready:
        blockers.append(
            _blocker("risk_register_not_ready", "risk register is not ready")
        )


def _extend_observability_operations_blockers(
    evidence: ObservabilityOperationsEvidence,
    blockers: list[dict[str, str]],
) -> None:
    if not evidence.ready:
        blockers.append(
            _blocker(
                "observability_operations_not_ready",
                "observability operations evidence is not ready",
            )
        )
    if evidence.environment.strip().lower() not in {"staging", "production"}:
        blockers.append(
            _blocker(
                "observability_environment_invalid",
                "observability operations evidence must come from staging or production",
            )
        )
    for field, code in (
        ("operations_packet_uri", "observability_operations_packet_missing"),
        ("dashboard_evidence_uri", "observability_dashboard_evidence_missing"),
        ("operator_signoff_uri", "observability_operator_signoff_missing"),
    ):
        if not getattr(evidence, field).strip():
            blockers.append(_blocker(code, f"{field} is required"))
    if not evidence.controls_passed:
        blockers.append(
            _blocker(
                "observability_controls_failed",
                "observability readiness controls did not pass",
            )
        )
    if not evidence.compliance_exports_available:
        blockers.append(
            _blocker(
                "observability_compliance_exports_missing",
                "observability compliance exports are missing or stale",
            )
        )
    if not evidence.drift_reviewed:
        blockers.append(
            _blocker(
                "observability_drift_not_reviewed",
                "observability drift alerts were not reviewed",
            )
        )
    if not _is_sha256(evidence.evidence_hash):
        blockers.append(
            _blocker(
                "observability_evidence_hash_invalid",
                "observability evidence_hash must be a sha256 hex digest",
            )
        )


def _extend_provenance_lineage_blockers(
    evidence: ProvenanceLineageEvidence,
    blockers: list[dict[str, str]],
) -> None:
    if not evidence.ready:
        blockers.append(
            _blocker(
                "provenance_lineage_not_ready",
                "KB provenance lineage evidence is not ready",
            )
        )
    if evidence.environment.strip().lower() not in {"staging", "production"}:
        blockers.append(
            _blocker(
                "provenance_lineage_environment_invalid",
                "KB provenance lineage evidence must come from staging or production",
            )
        )
    for field, code in (
        ("feedback_loop_run_uri", "provenance_feedback_loop_run_missing"),
        ("signed_lineage_packet_uri", "provenance_signed_lineage_packet_missing"),
        ("tenant_kb_snapshot_uri", "provenance_tenant_kb_snapshot_missing"),
        ("operator_signoff_uri", "provenance_operator_signoff_missing"),
    ):
        if not getattr(evidence, field).strip():
            blockers.append(_blocker(code, f"{field} is required"))
    if not evidence.lineage_matches_deployed_facts:
        blockers.append(
            _blocker(
                "provenance_lineage_mismatch",
                "signed lineage packet does not match deployed tenant facts",
            )
        )
    if not evidence.protected_claim_conflicts_resolved:
        blockers.append(
            _blocker(
                "provenance_conflicts_unresolved",
                "protected-claim conflicts are not resolved",
            )
        )
    if not _is_sha256(evidence.evidence_hash):
        blockers.append(
            _blocker(
                "provenance_evidence_hash_invalid",
                "provenance evidence_hash must be a sha256 hex digest",
            )
        )


def _extend_conformal_routing_blockers(
    evidence: ConformalRoutingEvidence,
    blockers: list[dict[str, str]],
) -> None:
    if not evidence.ready:
        blockers.append(
            _blocker(
                "conformal_routing_not_ready",
                "conformal routing evidence is not ready",
            )
        )
    if evidence.environment.strip().lower() not in {"staging", "production"}:
        blockers.append(
            _blocker(
                "conformal_routing_environment_invalid",
                "conformal routing evidence must come from staging or production",
            )
        )
    for field, code in (
        ("domain_calibration_packet_uri", "conformal_calibration_packet_missing"),
        ("deployment_routing_packet_uri", "conformal_routing_packet_missing"),
        ("escalation_route", "conformal_escalation_route_missing"),
        ("operator_signoff_uri", "conformal_operator_signoff_missing"),
    ):
        if not getattr(evidence, field).strip():
            blockers.append(_blocker(code, f"{field} is required"))
    if not 0.0 < evidence.target_coverage <= 1.0:
        blockers.append(
            _blocker(
                "conformal_target_coverage_invalid",
                "conformal target_coverage must be between 0 and 1",
            )
        )
    if not 0.0 <= evidence.empirical_coverage <= 1.0:
        blockers.append(
            _blocker(
                "conformal_empirical_coverage_invalid",
                "conformal empirical_coverage must be between 0 and 1",
            )
        )
    if (
        0.0 < evidence.target_coverage <= 1.0
        and 0.0 <= evidence.empirical_coverage <= 1.0
        and evidence.empirical_coverage < evidence.target_coverage
    ):
        blockers.append(
            _blocker(
                "conformal_coverage_below_target",
                "empirical conformal coverage is below the target",
            )
        )
    if evidence.calibration_sample_count <= 0:
        blockers.append(
            _blocker(
                "conformal_calibration_samples_missing",
                "conformal calibration_sample_count must be positive",
            )
        )
    if not evidence.escalation_route_verified:
        blockers.append(
            _blocker(
                "conformal_escalation_route_unverified",
                "conformal escalation route is not verified",
            )
        )
    if not evidence.reject_to_human_available:
        blockers.append(
            _blocker(
                "conformal_reject_to_human_unavailable",
                "reject-to-human route is not available",
            )
        )
    if not _is_sha256(evidence.evidence_hash):
        blockers.append(
            _blocker(
                "conformal_evidence_hash_invalid",
                "conformal evidence_hash must be a sha256 hex digest",
            )
        )


def _extend_trajectory_rollback_blockers(
    evidence: TrajectoryRollbackEvidence,
    blockers: list[dict[str, str]],
) -> None:
    if not evidence.ready:
        blockers.append(
            _blocker(
                "trajectory_rollback_not_ready",
                "trajectory rollback evidence is not ready",
            )
        )
    if evidence.environment.strip().lower() not in {"staging", "production"}:
        blockers.append(
            _blocker(
                "trajectory_rollback_environment_invalid",
                "trajectory rollback evidence must come from staging or production",
            )
        )
    for field, code in (
        ("simulation_evidence_uri", "trajectory_simulation_evidence_missing"),
        ("live_undo_backend_uri", "trajectory_live_undo_backend_missing"),
        ("adversarial_stress_packet_uri", "trajectory_stress_packet_missing"),
        ("incident_change_record_uri", "trajectory_change_record_missing"),
        ("operator_signoff_uri", "trajectory_operator_signoff_missing"),
    ):
        if not getattr(evidence, field).strip():
            blockers.append(_blocker(code, f"{field} is required"))
    if not evidence.rollback_hook_verified:
        blockers.append(
            _blocker(
                "trajectory_rollback_hook_unverified",
                "trajectory rollback hook is not verified against the live backend",
            )
        )
    if not evidence.idempotency_verified:
        blockers.append(
            _blocker(
                "trajectory_idempotency_unverified",
                "trajectory rollback idempotency is not verified",
            )
        )
    if not evidence.tenant_safe_audit_verified:
        blockers.append(
            _blocker(
                "trajectory_tenant_safe_audit_unverified",
                "trajectory rollback audit evidence is not tenant-safe",
            )
        )
    if not _is_sha256(evidence.evidence_hash):
        blockers.append(
            _blocker(
                "trajectory_evidence_hash_invalid",
                "trajectory evidence_hash must be a sha256 hex digest",
            )
        )


def _extend_multimodal_temporal_blockers(
    evidence: MultimodalTemporalEvidence,
    blockers: list[dict[str, str]],
) -> None:
    if not evidence.ready:
        blockers.append(
            _blocker(
                "multimodal_temporal_not_ready",
                "multimodal temporal evidence is not ready",
            )
        )
    if evidence.environment.strip().lower() not in {"staging", "production"}:
        blockers.append(
            _blocker(
                "multimodal_temporal_environment_invalid",
                "multimodal temporal evidence must come from staging or production",
            )
        )
    for field, code in (
        ("vision_nli_benchmark_uri", "multimodal_vision_nli_benchmark_missing"),
        ("video_frame_validation_uri", "multimodal_video_validation_missing"),
        ("modality_coverage_uri", "multimodal_modality_coverage_missing"),
        ("operator_signoff_uri", "multimodal_operator_signoff_missing"),
    ):
        if not getattr(evidence, field).strip():
            blockers.append(_blocker(code, f"{field} is required"))
    if not evidence.image_guard_verified:
        blockers.append(
            _blocker(
                "multimodal_image_guard_unverified",
                "image guard validation is not verified",
            )
        )
    if not evidence.audio_guard_verified:
        blockers.append(
            _blocker(
                "multimodal_audio_guard_unverified",
                "audio guard validation is not verified",
            )
        )
    if not evidence.video_temporal_verified:
        blockers.append(
            _blocker(
                "multimodal_video_temporal_unverified",
                "video temporal consistency validation is not verified",
            )
        )
    if not evidence.caption_grounding_verified:
        blockers.append(
            _blocker(
                "multimodal_caption_grounding_unverified",
                "caption grounding validation is not verified",
            )
        )
    if not evidence.deployment_modalities_covered:
        blockers.append(
            _blocker(
                "multimodal_deployment_coverage_missing",
                "deployment-specific modality coverage is missing",
            )
        )
    if not _is_sha256(evidence.evidence_hash):
        blockers.append(
            _blocker(
                "multimodal_evidence_hash_invalid",
                "multimodal evidence_hash must be a sha256 hex digest",
            )
        )


def _extend_federated_privacy_blockers(
    evidence: FederatedPrivacyEvidence,
    blockers: list[dict[str, str]],
) -> None:
    if not evidence.ready:
        blockers.append(
            _blocker(
                "federated_privacy_not_ready",
                "federated privacy evidence is not ready",
            )
        )
    if evidence.environment.strip().lower() not in {"staging", "production"}:
        blockers.append(
            _blocker(
                "federated_privacy_environment_invalid",
                "federated privacy evidence must come from staging or production",
            )
        )
    for field, code in (
        ("external_federation_run_uri", "federated_external_run_missing"),
        ("malicious_secure_review_uri", "federated_malicious_review_missing"),
        ("poisoning_resilience_packet_uri", "federated_poisoning_packet_missing"),
        ("privacy_budget_ledger_uri", "federated_privacy_ledger_missing"),
        ("operator_signoff_uri", "federated_operator_signoff_missing"),
    ):
        if not getattr(evidence, field).strip():
            blockers.append(_blocker(code, f"{field} is required"))
    if not evidence.dp_aggregation_verified:
        blockers.append(
            _blocker(
                "federated_dp_aggregation_unverified",
                "DP aggregation is not verified",
            )
        )
    if not evidence.cohort_gate_verified:
        blockers.append(
            _blocker(
                "federated_cohort_gate_unverified",
                "minimum cohort gate is not verified",
            )
        )
    if not evidence.secret_sharing_verified:
        blockers.append(
            _blocker(
                "federated_secret_sharing_unverified",
                "secret-sharing aggregation is not verified",
            )
        )
    if not evidence.contribution_caps_verified:
        blockers.append(
            _blocker(
                "federated_contribution_caps_unverified",
                "tenant/category contribution caps are not verified",
            )
        )
    if not _is_sha256(evidence.evidence_hash):
        blockers.append(
            _blocker(
                "federated_evidence_hash_invalid",
                "federated privacy evidence_hash must be a sha256 hex digest",
            )
        )


def _extend_edge_mobile_blockers(
    evidence: EdgeMobileEvidence,
    blockers: list[dict[str, str]],
) -> None:
    if not evidence.ready:
        blockers.append(
            _blocker(
                "edge_mobile_not_ready",
                "edge/mobile runtime evidence is not ready",
            )
        )
    if evidence.environment.strip().lower() not in {"staging", "production"}:
        blockers.append(
            _blocker(
                "edge_mobile_environment_invalid",
                "edge/mobile evidence must come from staging or production",
            )
        )
    for field, code in (
        ("edge_runtime_evidence_uri", "edge_runtime_evidence_missing"),
        ("quantised_model_artifact_uri", "edge_quantised_model_missing"),
        ("wasm_package_evidence_uri", "edge_wasm_package_missing"),
        ("browser_worker_smoke_uri", "edge_browser_worker_smoke_missing"),
        ("mobile_smoke_evidence_uri", "edge_mobile_smoke_missing"),
        ("package_publish_evidence_uri", "edge_package_publish_missing"),
        ("latency_profile_uri", "edge_latency_profile_missing"),
        ("operator_signoff_uri", "edge_operator_signoff_missing"),
    ):
        if not getattr(evidence, field).strip():
            blockers.append(_blocker(code, f"{field} is required"))
    if not evidence.quantised_model_verified:
        blockers.append(
            _blocker(
                "edge_quantised_model_unverified",
                "quantised model artefact is not verified",
            )
        )
    if not evidence.wasm_package_verified:
        blockers.append(
            _blocker(
                "edge_wasm_package_unverified",
                "WASM package evidence is not verified",
            )
        )
    if not evidence.browser_worker_smoke_passed:
        blockers.append(
            _blocker(
                "edge_browser_worker_smoke_failed",
                "browser Web Worker smoke did not pass",
            )
        )
    if not evidence.mobile_or_embedded_smoke_passed:
        blockers.append(
            _blocker(
                "edge_mobile_smoke_failed",
                "mobile or embedded-device smoke did not pass",
            )
        )
    if not evidence.latency_budget_met:
        blockers.append(
            _blocker(
                "edge_latency_budget_failed",
                "edge/mobile latency profile did not meet the release budget",
            )
        )
    if not evidence.package_publish_verified:
        blockers.append(
            _blocker(
                "edge_package_publish_unverified",
                "edge/mobile package publish evidence is not verified",
            )
        )
    if not _is_sha256(evidence.evidence_hash):
        blockers.append(
            _blocker(
                "edge_mobile_evidence_hash_invalid",
                "edge/mobile evidence_hash must be a sha256 hex digest",
            )
        )


def _extend_auto_redteam_defence_blockers(
    evidence: AutoRedteamDefenceEvidence,
    blockers: list[dict[str, str]],
) -> None:
    if not evidence.ready:
        blockers.append(
            _blocker(
                "auto_redteam_defence_not_ready",
                "auto-redteam defence evidence is not ready",
            )
        )
    if evidence.environment.strip().lower() not in {"staging", "production"}:
        blockers.append(
            _blocker(
                "auto_redteam_defence_environment_invalid",
                "auto-redteam defence evidence must come from staging or production",
            )
        )
    for field, code in (
        ("nightly_run_uri", "auto_redteam_nightly_run_missing"),
        ("defence_update_packet_uri", "auto_redteam_update_packet_missing"),
        ("registry_snapshot_uri", "auto_redteam_registry_snapshot_missing"),
        (
            "external_adversarial_corpus_uri",
            "auto_redteam_external_corpus_missing",
        ),
        (
            "patch_integration_signoff_uri",
            "auto_redteam_patch_signoff_missing",
        ),
        ("rollback_plan_uri", "auto_redteam_rollback_plan_missing"),
        ("operator_signoff_uri", "auto_redteam_operator_signoff_missing"),
    ):
        if not getattr(evidence, field).strip():
            blockers.append(_blocker(code, f"{field} is required"))
    if not evidence.repeated_cycles_verified:
        blockers.append(
            _blocker(
                "auto_redteam_repeated_cycles_unverified",
                "repeated auto-redteam cycles are not verified",
            )
        )
    if not evidence.detection_uplift_verified:
        blockers.append(
            _blocker(
                "auto_redteam_detection_uplift_unverified",
                "detection uplift is not verified",
            )
        )
    if not evidence.registry_promotions_verified:
        blockers.append(
            _blocker(
                "auto_redteam_registry_promotions_unverified",
                "defence registry promotions are not verified",
            )
        )
    if not evidence.tenant_safe_reports_verified:
        blockers.append(
            _blocker(
                "auto_redteam_tenant_safe_reports_unverified",
                "auto-redteam reports are not verified as tenant-safe",
            )
        )
    if not evidence.rollback_plan_verified:
        blockers.append(
            _blocker(
                "auto_redteam_rollback_plan_unverified",
                "defence rollback plan is not verified",
            )
        )
    if not _is_sha256(evidence.evidence_hash):
        blockers.append(
            _blocker(
                "auto_redteam_evidence_hash_invalid",
                "auto-redteam defence evidence_hash must be a sha256 hex digest",
            )
        )


def _extend_formal_symbolic_blockers(
    evidence: FormalSymbolicEvidence,
    blockers: list[dict[str, str]],
) -> None:
    if not evidence.ready:
        blockers.append(
            _blocker(
                "formal_symbolic_not_ready",
                "formal-symbolic evidence is not ready",
            )
        )
    if evidence.environment.strip().lower() not in {"staging", "production"}:
        blockers.append(
            _blocker(
                "formal_symbolic_environment_invalid",
                "formal-symbolic evidence must come from staging or production",
            )
        )
    for field, code in (
        ("formal_symbolic_packet_uri", "formal_symbolic_packet_missing"),
        ("external_lean_proof_uri", "formal_external_lean_proof_missing"),
        ("z3_release_packet_uri", "formal_z3_release_packet_missing"),
        ("domain_contracts_uri", "formal_domain_contracts_missing"),
        ("code_contract_packet_uri", "formal_code_contract_packet_missing"),
        ("operator_signoff_uri", "formal_operator_signoff_missing"),
    ):
        if not getattr(evidence, field).strip():
            blockers.append(_blocker(code, f"{field} is required"))
    if not evidence.dpll_formula_guard_verified:
        blockers.append(
            _blocker(
                "formal_dpll_guard_unverified",
                "DPLL formula guard is not verified",
            )
        )
    if not evidence.lean_external_run_verified:
        blockers.append(
            _blocker(
                "formal_lean_external_run_unverified",
                "external Lean proof run is not verified",
            )
        )
    if not evidence.z3_actual_run_verified:
        blockers.append(
            _blocker(
                "formal_z3_actual_run_unverified",
                "actual Z3 release run is not verified",
            )
        )
    if not evidence.code_contract_ordering_verified:
        blockers.append(
            _blocker(
                "formal_code_contract_ordering_unverified",
                "code-contract ordering is not verified",
            )
        )
    if not evidence.tenant_safe_serialisation_verified:
        blockers.append(
            _blocker(
                "formal_tenant_safe_serialisation_unverified",
                "formal-symbolic serialisation is not verified as tenant-safe",
            )
        )
    if not evidence.domain_contracts_verified:
        blockers.append(
            _blocker(
                "formal_domain_contracts_unverified",
                "operator-owned formal domain contracts are not verified",
            )
        )
    if not _is_sha256(evidence.evidence_hash):
        blockers.append(
            _blocker(
                "formal_symbolic_evidence_hash_invalid",
                "formal-symbolic evidence_hash must be a sha256 hex digest",
            )
        )


def _extend_deployment_hardening_blockers(
    evidence: DeploymentHardeningEvidence,
    blockers: list[dict[str, str]],
) -> None:
    if not evidence.ready:
        blockers.append(
            _blocker(
                "deployment_hardening_not_ready",
                "deployment hardening evidence is not ready",
            )
        )
    if evidence.environment.strip().lower() not in {"staging", "production"}:
        blockers.append(
            _blocker(
                "deployment_hardening_environment_invalid",
                "deployment hardening evidence must come from staging or production",
            )
        )
    for field, code in (
        ("observation_window", "deployment_observation_window_missing"),
        ("telemetry_uri", "deployment_telemetry_uri_missing"),
        ("sustained_load_packet_uri", "deployment_sustained_load_packet_missing"),
        ("operator_signoff_uri", "deployment_operator_signoff_missing"),
    ):
        if not getattr(evidence, field).strip():
            blockers.append(_blocker(code, f"{field} is required"))
    if not evidence.async_ordering_passed:
        blockers.append(
            _blocker(
                "deployment_async_ordering_failed",
                "async ordering probe did not pass in deployment hardening evidence",
            )
        )
    if not evidence.tenant_poisoning_passed:
        blockers.append(
            _blocker(
                "deployment_tenant_poisoning_failed",
                "tenant poisoning probe did not pass in deployment hardening evidence",
            )
        )
    if not _is_sha256(evidence.evidence_hash):
        blockers.append(
            _blocker(
                "deployment_hardening_hash_invalid",
                "deployment hardening evidence_hash must be a sha256 hex digest",
            )
        )


def _extend_boundary_blockers(
    runtime_package: CustomerRuntimePackage,
    evidence_pack: CustomerEvidencePackManifest,
    monitoring_manifest: CustomerMonitoringManifest,
    risk_register: CustomerRiskRegister,
    blockers: list[dict[str, str]],
) -> None:
    for field, code in (
        ("customer_id", "customer_boundary_mismatch"),
        ("workspace_id", "workspace_boundary_mismatch"),
        ("tenant_id", "tenant_boundary_mismatch"),
        ("deployment_id", "deployment_boundary_mismatch"),
    ):
        values = {
            getattr(runtime_package, field),
            getattr(evidence_pack, field),
            getattr(monitoring_manifest, field),
            getattr(risk_register, field),
        }
        if len(values) != 1:
            blockers.append(
                _blocker(code, f"{field} differs between release artefacts")
            )
    if runtime_package.evidence_hash != evidence_pack.evidence_hash:
        blockers.append(
            _blocker(
                "runtime_evidence_hash_mismatch",
                "runtime evidence hash differs from evidence pack",
            )
        )
    if evidence_pack.evidence_hash != monitoring_manifest.evidence_hash:
        blockers.append(
            _blocker(
                "monitoring_evidence_hash_mismatch",
                "monitoring evidence hash differs from evidence pack",
            )
        )
    if evidence_pack.evidence_hash != risk_register.evidence_hash:
        blockers.append(
            _blocker(
                "risk_evidence_hash_mismatch",
                "risk register evidence hash differs from evidence pack",
            )
        )
    if monitoring_manifest.monitoring_hash != risk_register.monitoring_hash:
        blockers.append(
            _blocker(
                "risk_monitoring_hash_mismatch",
                "risk register monitoring hash differs from monitoring manifest",
            )
        )


def _blocker(code: str, message: str, **extra: str) -> dict[str, str]:
    payload = {"code": code, "severity": "error", "message": message}
    payload.update({key: value for key, value in extra.items() if value})
    return payload


def _stable_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)
