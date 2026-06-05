# SPDX-License-Identifier: AGPL-3.0-or-later
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
            compliance_exports_available=bool(
                payload["compliance_exports_available"]
            ),
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
            "provenance_lineage_evidence": (
                self.provenance_lineage_evidence.to_dict()
            ),
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
