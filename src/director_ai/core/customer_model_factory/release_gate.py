# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory release gate

"""Final promotion gate for Customer Model Factory packages.

This module owns the gate assembly: the
:class:`CustomerReleaseGateManifest` aggregate, the
:func:`build_release_gate_manifest` builder, and the release hash. The
ten per-domain evidence dataclasses live in :mod:`._gate_evidence` and
are re-exported here unchanged; the per-domain blocker policy lives in
:mod:`._gate_blockers`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._gate_evidence import AutoRedteamDefenceEvidence as AutoRedteamDefenceEvidence
from ._gate_evidence import ConformalRoutingEvidence as ConformalRoutingEvidence
from ._gate_evidence import (
    DeploymentHardeningEvidence as DeploymentHardeningEvidence,
)
from ._gate_evidence import EdgeMobileEvidence as EdgeMobileEvidence
from ._gate_evidence import FederatedPrivacyEvidence as FederatedPrivacyEvidence
from ._gate_evidence import FormalSymbolicEvidence as FormalSymbolicEvidence
from ._gate_evidence import MultimodalTemporalEvidence as MultimodalTemporalEvidence
from ._gate_evidence import (
    ObservabilityOperationsEvidence as ObservabilityOperationsEvidence,
)
from ._gate_evidence import ProvenanceLineageEvidence as ProvenanceLineageEvidence
from ._gate_evidence import TrajectoryRollbackEvidence as TrajectoryRollbackEvidence
from .evidence_pack import CustomerEvidencePackManifest
from .monitoring_manifest import CustomerMonitoringManifest
from .risk_register import CustomerRiskRegister
from .runtime_package import CustomerRuntimePackage

SCHEMA_VERSION = "1.0.0"


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
