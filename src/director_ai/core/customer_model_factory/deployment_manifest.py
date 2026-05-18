# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory deployment manifest

"""Deployment package manifest for customer-selected guardrail models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .benchmark_selection import CustomerModelSelectionReport

SCHEMA_VERSION = "1.0.0"
TELEMETRY_MODES = frozenset({"disabled", "customer_controlled", "internal_only"})
ENVIRONMENTS = frozenset({"development", "staging", "production"})


@dataclass(frozen=True)
class DeploymentPolicy:
    """Runtime policy bound to a customer deployment package."""

    threshold: float
    abstention_threshold: float
    escalation_threshold: float
    require_citations: bool
    audit_log_uri: str
    evidence_pack_uri: str
    rollback_package_uri: str
    retention_days: int
    telemetry_mode: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise policy to JSON-safe data."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DeploymentPolicy:
        """Rebuild policy from JSON-safe data."""

        return cls(
            threshold=float(payload["threshold"]),
            abstention_threshold=float(payload["abstention_threshold"]),
            escalation_threshold=float(payload["escalation_threshold"]),
            require_citations=bool(payload["require_citations"]),
            audit_log_uri=payload["audit_log_uri"],
            evidence_pack_uri=payload["evidence_pack_uri"],
            rollback_package_uri=payload["rollback_package_uri"],
            retention_days=int(payload["retention_days"]),
            telemetry_mode=payload["telemetry_mode"],
        )


@dataclass(frozen=True)
class CustomerDeploymentManifest:
    """Export-ready deployment package manifest."""

    schema_version: str
    deployment_id: str
    ready: bool
    customer_id: str
    workspace_id: str
    tenant_id: str
    selection_hash: str
    selected_benchmark_id: str
    selected_model_artifact_uri: str
    policy: DeploymentPolicy
    environment: str
    package_uri: str
    findings: tuple[dict[str, str], ...]
    deployment_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise deployment manifest to stable JSON-safe data."""

        return {
            "schema_version": self.schema_version,
            "deployment_id": self.deployment_id,
            "ready": self.ready,
            "customer_id": self.customer_id,
            "workspace_id": self.workspace_id,
            "tenant_id": self.tenant_id,
            "selection_hash": self.selection_hash,
            "selected_benchmark_id": self.selected_benchmark_id,
            "selected_model_artifact_uri": self.selected_model_artifact_uri,
            "policy": self.policy.to_dict(),
            "environment": self.environment,
            "package_uri": self.package_uri,
            "findings": [dict(finding) for finding in self.findings],
            "deployment_hash": self.deployment_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CustomerDeploymentManifest:
        """Rebuild deployment manifest from serialised data."""

        return cls(
            schema_version=payload["schema_version"],
            deployment_id=payload["deployment_id"],
            ready=bool(payload["ready"]),
            customer_id=payload["customer_id"],
            workspace_id=payload["workspace_id"],
            tenant_id=payload["tenant_id"],
            selection_hash=payload["selection_hash"],
            selected_benchmark_id=payload["selected_benchmark_id"],
            selected_model_artifact_uri=payload["selected_model_artifact_uri"],
            policy=DeploymentPolicy.from_dict(payload["policy"]),
            environment=payload["environment"],
            package_uri=payload["package_uri"],
            findings=tuple(dict(finding) for finding in payload["findings"]),
            deployment_hash=payload["deployment_hash"],
        )

    def write_json(self, path: Path) -> Path:
        """Write the deployment manifest as deterministic JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


def build_deployment_manifest(
    *,
    deployment_id: str,
    selection_report: CustomerModelSelectionReport,
    policy: DeploymentPolicy,
    environment: str,
    package_uri: str,
) -> CustomerDeploymentManifest:
    """Build and validate a customer deployment package manifest."""

    findings = _collect_findings(
        deployment_id=deployment_id,
        selection_report=selection_report,
        policy=policy,
        environment=environment,
        package_uri=package_uri,
    )
    candidate = selection_report.candidates[0] if selection_report.candidates else None
    customer_id = candidate.customer_id if candidate else ""
    workspace_id = candidate.workspace_id if candidate else ""
    tenant_id = candidate.tenant_id if candidate else ""
    deployment_hash = _stable_hash(
        {
            "deployment_id": deployment_id,
            "selection_hash": selection_report.selection_hash,
            "selected_benchmark_id": selection_report.selected_benchmark_id,
            "selected_model_artifact_uri": selection_report.selected_model_artifact_uri,
            "policy": policy.to_dict(),
            "environment": environment,
            "package_uri": package_uri,
            "findings": findings,
        }
    )
    return CustomerDeploymentManifest(
        schema_version=SCHEMA_VERSION,
        deployment_id=deployment_id,
        ready=not findings,
        customer_id=customer_id,
        workspace_id=workspace_id,
        tenant_id=tenant_id,
        selection_hash=selection_report.selection_hash,
        selected_benchmark_id=selection_report.selected_benchmark_id,
        selected_model_artifact_uri=selection_report.selected_model_artifact_uri,
        policy=policy,
        environment=environment,
        package_uri=package_uri,
        findings=tuple(findings),
        deployment_hash=deployment_hash,
    )


def _collect_findings(
    *,
    deployment_id: str,
    selection_report: CustomerModelSelectionReport,
    policy: DeploymentPolicy,
    environment: str,
    package_uri: str,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if not deployment_id.strip():
        findings.append(_finding("deployment_id_missing", "deployment_id is required"))
    if not selection_report.ready:
        findings.append(
            _finding("selection_not_ready", "selection report is not ready")
        )
    if not selection_report.selected_model_artifact_uri:
        findings.append(
            _finding(
                "selected_model_missing", "selected model artefact URI is required"
            )
        )
    if environment not in ENVIRONMENTS:
        findings.append(
            _finding(
                "environment_unknown", f"environment {environment!r} is not supported"
            )
        )
    if not package_uri.strip():
        findings.append(_finding("package_uri_missing", "package_uri is required"))
    _extend_policy_findings(policy, environment, findings)
    return findings


def _extend_policy_findings(
    policy: DeploymentPolicy,
    environment: str,
    findings: list[dict[str, str]],
) -> None:
    for name, value in (
        ("threshold", policy.threshold),
        ("abstention_threshold", policy.abstention_threshold),
        ("escalation_threshold", policy.escalation_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            findings.append(
                _finding("threshold_out_of_range", f"{name} must be between 0 and 1")
            )
    if not policy.threshold > policy.abstention_threshold > policy.escalation_threshold:
        findings.append(
            _finding(
                "threshold_order_invalid",
                "threshold must be greater than abstention threshold, which must be greater than escalation threshold",
            )
        )
    if not policy.audit_log_uri.strip():
        findings.append(_finding("audit_log_missing", "audit_log_uri is required"))
    if not policy.evidence_pack_uri.strip():
        findings.append(
            _finding("evidence_pack_missing", "evidence_pack_uri is required")
        )
    if environment == "production" and not policy.rollback_package_uri.strip():
        findings.append(
            _finding(
                "rollback_missing",
                "production deployments require rollback_package_uri",
            )
        )
    if policy.retention_days < 1:
        findings.append(
            _finding("retention_invalid", "retention_days must be positive")
        )
    if policy.telemetry_mode not in TELEMETRY_MODES:
        findings.append(
            _finding(
                "telemetry_mode_unknown",
                f"telemetry_mode {policy.telemetry_mode!r} is not supported",
            )
        )


def _finding(code: str, message: str) -> dict[str, str]:
    return {"code": code, "severity": "error", "message": message}


def _stable_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()
