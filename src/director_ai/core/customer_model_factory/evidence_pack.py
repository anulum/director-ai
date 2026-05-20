# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory evidence pack

"""Customer-reviewable evidence package manifest for model factory exports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .deployment_manifest import CustomerDeploymentManifest
from .sector_extension import SectorEvidenceMapping

SCHEMA_VERSION = "1.0.0"
ALLOWED_CLASSIFICATIONS = frozenset({"internal", "confidential", "restricted"})


@dataclass(frozen=True)
class CustomerEvidencePackManifest:
    """Deterministic manifest for customer evidence-room exports."""

    schema_version: str
    package_id: str
    ready: bool
    customer_id: str
    workspace_id: str
    tenant_id: str
    deployment_id: str
    environment: str
    classification: str
    export_uri: str
    external_callbacks_allowed: bool
    callback_endpoints: tuple[str, ...]
    artefacts: dict[str, str]
    control_evidence: dict[str, tuple[str, ...]]
    findings: tuple[dict[str, str], ...]
    evidence_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise the evidence pack manifest to stable JSON-safe data."""

        return {
            "schema_version": self.schema_version,
            "package_id": self.package_id,
            "ready": self.ready,
            "customer_id": self.customer_id,
            "workspace_id": self.workspace_id,
            "tenant_id": self.tenant_id,
            "deployment_id": self.deployment_id,
            "environment": self.environment,
            "classification": self.classification,
            "export_uri": self.export_uri,
            "external_callbacks_allowed": self.external_callbacks_allowed,
            "callback_endpoints": list(self.callback_endpoints),
            "artefacts": dict(sorted(self.artefacts.items())),
            "control_evidence": {
                control: list(fields)
                for control, fields in sorted(self.control_evidence.items())
            },
            "findings": [dict(finding) for finding in self.findings],
            "evidence_hash": self.evidence_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CustomerEvidencePackManifest:
        """Rebuild an evidence pack manifest from serialised data."""

        return cls(
            schema_version=payload["schema_version"],
            package_id=payload["package_id"],
            ready=bool(payload["ready"]),
            customer_id=payload["customer_id"],
            workspace_id=payload["workspace_id"],
            tenant_id=payload["tenant_id"],
            deployment_id=payload["deployment_id"],
            environment=payload["environment"],
            classification=payload["classification"],
            export_uri=payload["export_uri"],
            external_callbacks_allowed=bool(payload["external_callbacks_allowed"]),
            callback_endpoints=tuple(payload["callback_endpoints"]),
            artefacts=dict(payload["artefacts"]),
            control_evidence={
                control: tuple(fields)
                for control, fields in payload["control_evidence"].items()
            },
            findings=tuple(dict(finding) for finding in payload["findings"]),
            evidence_hash=payload["evidence_hash"],
        )

    def write_json(self, path: Path) -> Path:
        """Write the evidence pack manifest as deterministic JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


def build_customer_evidence_pack(
    *,
    package_id: str,
    deployment_manifest: CustomerDeploymentManifest,
    regulation_mapping: SectorEvidenceMapping,
    classification: str,
    export_uri: str,
    external_callbacks_allowed: bool = False,
    callback_endpoints: tuple[str, ...] = (),
) -> CustomerEvidencePackManifest:
    """Build and validate a customer evidence-pack manifest."""

    artefacts = _artefacts(deployment_manifest, regulation_mapping)
    findings = _collect_findings(
        package_id=package_id,
        deployment_manifest=deployment_manifest,
        regulation_mapping=regulation_mapping,
        classification=classification,
        export_uri=export_uri,
        external_callbacks_allowed=external_callbacks_allowed,
        callback_endpoints=callback_endpoints,
    )
    evidence_hash = _stable_hash(
        {
            "package_id": package_id,
            "customer_id": deployment_manifest.customer_id,
            "workspace_id": deployment_manifest.workspace_id,
            "tenant_id": deployment_manifest.tenant_id,
            "deployment_id": deployment_manifest.deployment_id,
            "environment": deployment_manifest.environment,
            "classification": classification,
            "export_uri": export_uri,
            "external_callbacks_allowed": external_callbacks_allowed,
            "callback_endpoints": callback_endpoints,
            "artefacts": artefacts,
            "control_evidence": regulation_mapping.control_evidence,
            "findings": findings,
        }
    )
    return CustomerEvidencePackManifest(
        schema_version=SCHEMA_VERSION,
        package_id=package_id,
        ready=not findings,
        customer_id=deployment_manifest.customer_id,
        workspace_id=deployment_manifest.workspace_id,
        tenant_id=deployment_manifest.tenant_id,
        deployment_id=deployment_manifest.deployment_id,
        environment=deployment_manifest.environment,
        classification=classification,
        export_uri=export_uri,
        external_callbacks_allowed=external_callbacks_allowed,
        callback_endpoints=callback_endpoints,
        artefacts=artefacts,
        control_evidence=regulation_mapping.control_evidence,
        findings=tuple(findings),
        evidence_hash=evidence_hash,
    )


def _artefacts(
    deployment_manifest: CustomerDeploymentManifest,
    regulation_mapping: SectorEvidenceMapping,
) -> dict[str, str]:
    return {
        "audit_log_uri": deployment_manifest.policy.audit_log_uri,
        "deployment_hash": deployment_manifest.deployment_hash,
        "deployment_package_uri": deployment_manifest.package_uri,
        "evidence_pack_uri": deployment_manifest.policy.evidence_pack_uri,
        "regulation_mapping_hash": regulation_mapping.mapping_hash,
        "rollback_package_uri": deployment_manifest.policy.rollback_package_uri,
        "selected_benchmark_id": deployment_manifest.selected_benchmark_id,
        "selected_model_artifact_uri": deployment_manifest.selected_model_artifact_uri,
        "selection_hash": deployment_manifest.selection_hash,
        "telemetry_mode": deployment_manifest.policy.telemetry_mode,
    }


def _collect_findings(
    *,
    package_id: str,
    deployment_manifest: CustomerDeploymentManifest,
    regulation_mapping: SectorEvidenceMapping,
    classification: str,
    export_uri: str,
    external_callbacks_allowed: bool,
    callback_endpoints: tuple[str, ...],
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if not package_id.strip():
        findings.append(_finding("package_id_missing", "package_id is required"))
    if not deployment_manifest.ready:
        findings.append(
            _finding("deployment_not_ready", "deployment manifest is not ready")
        )
    if classification not in ALLOWED_CLASSIFICATIONS:
        findings.append(
            _finding(
                "classification_unknown",
                f"classification {classification!r} is not supported",
            )
        )
    if not export_uri.strip():
        findings.append(_finding("export_uri_missing", "export_uri is required"))
    if export_uri and export_uri != deployment_manifest.policy.evidence_pack_uri:
        findings.append(
            _finding(
                "evidence_uri_mismatch",
                "export_uri must match deployment policy evidence_pack_uri",
            )
        )
    if export_uri and export_uri != regulation_mapping.evidence_pack_uri:
        findings.append(
            _finding(
                "regulation_mapping_uri_mismatch",
                "export_uri must match regulation mapping evidence_pack_uri",
            )
        )
    if callback_endpoints and not external_callbacks_allowed:
        findings.append(
            _finding(
                "external_callback_not_allowed",
                "external callbacks require explicit customer export approval",
            )
        )
    return findings


def _finding(code: str, message: str) -> dict[str, str]:
    return {"code": code, "severity": "error", "message": message}


def _stable_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()
