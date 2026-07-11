# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory release-gate evidence records

"""Evidence dataclasses for the Customer Model Factory release gate.

Ten frozen evidence records — one per readiness domain (deployment
hardening, observability operations, KB provenance lineage, conformal
routing, trajectory rollback, multimodal temporal, federated privacy,
edge/mobile, auto-redteam defence, formal-symbolic) — each carrying its
signed artefact URIs, verification flags, and evidence hash, with stable
``to_dict``/``from_dict`` JSON contracts. The gate assembly and the
per-domain blocker policy live in :mod:`.release_gate` and
:mod:`._gate_blockers`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "AutoRedteamDefenceEvidence",
    "ConformalRoutingEvidence",
    "DeploymentHardeningEvidence",
    "EdgeMobileEvidence",
    "FederatedPrivacyEvidence",
    "FormalSymbolicEvidence",
    "MultimodalTemporalEvidence",
    "ObservabilityOperationsEvidence",
    "ProvenanceLineageEvidence",
    "TrajectoryRollbackEvidence",
]


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
