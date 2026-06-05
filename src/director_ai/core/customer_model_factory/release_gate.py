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
