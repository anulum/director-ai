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

from ._gate_blockers import (
    _blocker,
    _extend_auto_redteam_defence_blockers,
    _extend_boundary_blockers,
    _extend_conformal_routing_blockers,
    _extend_deployment_hardening_blockers,
    _extend_edge_mobile_blockers,
    _extend_federated_privacy_blockers,
    _extend_formal_symbolic_blockers,
    _extend_multimodal_temporal_blockers,
    _extend_observability_operations_blockers,
    _extend_provenance_lineage_blockers,
    _extend_readiness_blockers,
    _extend_trajectory_rollback_blockers,
)
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


def _stable_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()
