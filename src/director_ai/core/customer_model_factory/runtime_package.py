# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory runtime package

"""Runtime package configuration for customer-owned model deployments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .deployment_manifest import CustomerDeploymentManifest
from .evidence_pack import CustomerEvidencePackManifest

SCHEMA_VERSION = "1.0.0"
RUNTIME_MODES = frozenset({"offline_private", "customer_cloud_private", "on_prem"})


@dataclass(frozen=True)
class CustomerRuntimePackage:
    """Customer-implementable runtime configuration package."""

    schema_version: str
    runtime_id: str
    ready: bool
    customer_id: str
    workspace_id: str
    tenant_id: str
    deployment_id: str
    evidence_hash: str
    runtime_mode: str
    runtime_config: dict[str, Any]
    findings: tuple[dict[str, str], ...]
    runtime_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise the runtime package to stable JSON-safe data."""

        return {
            "schema_version": self.schema_version,
            "runtime_id": self.runtime_id,
            "ready": self.ready,
            "customer_id": self.customer_id,
            "workspace_id": self.workspace_id,
            "tenant_id": self.tenant_id,
            "deployment_id": self.deployment_id,
            "evidence_hash": self.evidence_hash,
            "runtime_mode": self.runtime_mode,
            "runtime_config": self.runtime_config,
            "findings": [dict(finding) for finding in self.findings],
            "runtime_hash": self.runtime_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CustomerRuntimePackage:
        """Rebuild a runtime package from serialised data."""

        return cls(
            schema_version=payload["schema_version"],
            runtime_id=payload["runtime_id"],
            ready=bool(payload["ready"]),
            customer_id=payload["customer_id"],
            workspace_id=payload["workspace_id"],
            tenant_id=payload["tenant_id"],
            deployment_id=payload["deployment_id"],
            evidence_hash=payload["evidence_hash"],
            runtime_mode=payload["runtime_mode"],
            runtime_config=dict(payload["runtime_config"]),
            findings=tuple(dict(finding) for finding in payload["findings"]),
            runtime_hash=payload["runtime_hash"],
        )

    @staticmethod
    def evidence_pack_from_dict(
        payload: dict[str, Any],
    ) -> CustomerEvidencePackManifest:
        """Rebuild an evidence pack for boundary-mismatch tests and tooling."""

        return CustomerEvidencePackManifest.from_dict(payload)

    def write_json(self, path: Path) -> Path:
        """Write the runtime package as deterministic JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


def build_customer_runtime_package(
    *,
    runtime_id: str,
    deployment_manifest: CustomerDeploymentManifest,
    evidence_pack: CustomerEvidencePackManifest,
    runtime_mode: str,
) -> CustomerRuntimePackage:
    """Build a customer runtime configuration package."""

    runtime_config = _runtime_config(deployment_manifest, evidence_pack)
    findings = _collect_findings(
        runtime_id=runtime_id,
        deployment_manifest=deployment_manifest,
        evidence_pack=evidence_pack,
        runtime_mode=runtime_mode,
    )
    runtime_hash = _stable_hash(
        {
            "runtime_id": runtime_id,
            "customer_id": deployment_manifest.customer_id,
            "workspace_id": deployment_manifest.workspace_id,
            "tenant_id": deployment_manifest.tenant_id,
            "deployment_id": deployment_manifest.deployment_id,
            "evidence_hash": evidence_pack.evidence_hash,
            "runtime_mode": runtime_mode,
            "runtime_config": runtime_config,
            "findings": findings,
        }
    )
    return CustomerRuntimePackage(
        schema_version=SCHEMA_VERSION,
        runtime_id=runtime_id,
        ready=not findings,
        customer_id=deployment_manifest.customer_id,
        workspace_id=deployment_manifest.workspace_id,
        tenant_id=deployment_manifest.tenant_id,
        deployment_id=deployment_manifest.deployment_id,
        evidence_hash=evidence_pack.evidence_hash,
        runtime_mode=runtime_mode,
        runtime_config=runtime_config,
        findings=tuple(findings),
        runtime_hash=runtime_hash,
    )


def _runtime_config(
    deployment_manifest: CustomerDeploymentManifest,
    evidence_pack: CustomerEvidencePackManifest,
) -> dict[str, Any]:
    policy = deployment_manifest.policy
    return {
        "customer_id": deployment_manifest.customer_id,
        "workspace_id": deployment_manifest.workspace_id,
        "tenant_id": deployment_manifest.tenant_id,
        "deployment_id": deployment_manifest.deployment_id,
        "deployment_hash": deployment_manifest.deployment_hash,
        "evidence_hash": evidence_pack.evidence_hash,
        "selected_benchmark_id": deployment_manifest.selected_benchmark_id,
        "selected_model_artifact_uri": deployment_manifest.selected_model_artifact_uri,
        "threshold": policy.threshold,
        "abstention_threshold": policy.abstention_threshold,
        "escalation_threshold": policy.escalation_threshold,
        "require_citations": policy.require_citations,
        "audit_log_uri": policy.audit_log_uri,
        "evidence_pack_uri": policy.evidence_pack_uri,
        "rollback_package_uri": policy.rollback_package_uri,
        "retention_days": policy.retention_days,
        "telemetry_mode": policy.telemetry_mode,
        "external_callbacks_allowed": evidence_pack.external_callbacks_allowed,
        "callback_endpoints": list(evidence_pack.callback_endpoints),
    }


def _collect_findings(
    *,
    runtime_id: str,
    deployment_manifest: CustomerDeploymentManifest,
    evidence_pack: CustomerEvidencePackManifest,
    runtime_mode: str,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if not runtime_id.strip():
        findings.append(_finding("runtime_id_missing", "runtime_id is required"))
    if runtime_mode not in RUNTIME_MODES:
        findings.append(
            _finding(
                "runtime_mode_unknown", f"runtime_mode {runtime_mode!r} is unsupported"
            )
        )
    if not deployment_manifest.ready:
        findings.append(
            _finding("deployment_not_ready", "deployment manifest is not ready")
        )
    if not evidence_pack.ready:
        findings.append(
            _finding("evidence_pack_not_ready", "evidence pack is not ready")
        )
    if deployment_manifest.customer_id != evidence_pack.customer_id:
        findings.append(
            _finding(
                "customer_boundary_mismatch", "customer_id differs between artefacts"
            )
        )
    if deployment_manifest.workspace_id != evidence_pack.workspace_id:
        findings.append(
            _finding(
                "workspace_boundary_mismatch", "workspace_id differs between artefacts"
            )
        )
    if deployment_manifest.tenant_id != evidence_pack.tenant_id:
        findings.append(
            _finding("tenant_boundary_mismatch", "tenant_id differs between artefacts")
        )
    if deployment_manifest.deployment_id != evidence_pack.deployment_id:
        findings.append(
            _finding(
                "deployment_boundary_mismatch",
                "deployment_id differs between artefacts",
            )
        )
    return findings


def _finding(code: str, message: str) -> dict[str, str]:
    return {"code": code, "severity": "error", "message": message}


def _stable_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()
