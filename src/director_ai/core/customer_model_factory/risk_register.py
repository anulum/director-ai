# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory risk register

"""Risk and exception register for customer model factory evidence gates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evidence_pack import CustomerEvidencePackManifest
from .monitoring_manifest import CustomerMonitoringManifest

SCHEMA_VERSION = "1.0.0"
ACCEPTED_STATUSES = frozenset({"accepted", "mitigated", "rejected"})


@dataclass(frozen=True)
class CustomerRiskException:
    """One customer factory risk or exception decision."""

    risk_id: str
    status: str
    severity: str
    owner: str
    accepted_at: str
    expires_at: str
    compensating_controls: tuple[str, ...]
    linked_artifact_hashes: dict[str, str]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise the risk exception to JSON-safe data."""

        return {
            "risk_id": self.risk_id,
            "status": self.status,
            "severity": self.severity,
            "owner": self.owner,
            "accepted_at": self.accepted_at,
            "expires_at": self.expires_at,
            "compensating_controls": list(self.compensating_controls),
            "linked_artifact_hashes": dict(sorted(self.linked_artifact_hashes.items())),
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CustomerRiskException:
        """Rebuild a risk exception from JSON-safe data."""

        return cls(
            risk_id=payload["risk_id"],
            status=payload["status"],
            severity=payload["severity"],
            owner=payload["owner"],
            accepted_at=payload["accepted_at"],
            expires_at=payload["expires_at"],
            compensating_controls=tuple(payload["compensating_controls"]),
            linked_artifact_hashes=dict(payload["linked_artifact_hashes"]),
            rationale=payload["rationale"],
        )


@dataclass(frozen=True)
class CustomerRiskRegister:
    """Risk register gate bound to evidence and monitoring artefacts."""

    schema_version: str
    register_id: str
    ready: bool
    customer_id: str
    workspace_id: str
    tenant_id: str
    deployment_id: str
    evidence_hash: str
    monitoring_hash: str
    generated_at: str
    risks: tuple[CustomerRiskException, ...]
    findings: tuple[dict[str, str], ...]
    register_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise the risk register to stable JSON-safe data."""

        return {
            "schema_version": self.schema_version,
            "register_id": self.register_id,
            "ready": self.ready,
            "customer_id": self.customer_id,
            "workspace_id": self.workspace_id,
            "tenant_id": self.tenant_id,
            "deployment_id": self.deployment_id,
            "evidence_hash": self.evidence_hash,
            "monitoring_hash": self.monitoring_hash,
            "generated_at": self.generated_at,
            "risks": [risk.to_dict() for risk in self.risks],
            "findings": [dict(finding) for finding in self.findings],
            "register_hash": self.register_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CustomerRiskRegister:
        """Rebuild a risk register from JSON-safe data."""

        return cls(
            schema_version=payload["schema_version"],
            register_id=payload["register_id"],
            ready=bool(payload["ready"]),
            customer_id=payload["customer_id"],
            workspace_id=payload["workspace_id"],
            tenant_id=payload["tenant_id"],
            deployment_id=payload["deployment_id"],
            evidence_hash=payload["evidence_hash"],
            monitoring_hash=payload["monitoring_hash"],
            generated_at=payload["generated_at"],
            risks=tuple(
                CustomerRiskException.from_dict(risk) for risk in payload["risks"]
            ),
            findings=tuple(dict(finding) for finding in payload["findings"]),
            register_hash=payload["register_hash"],
        )

    def write_json(self, path: Path) -> Path:
        """Write the risk register as deterministic JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


def build_risk_register(
    *,
    register_id: str,
    evidence_pack: CustomerEvidencePackManifest,
    monitoring_manifest: CustomerMonitoringManifest,
    risks: tuple[CustomerRiskException, ...],
    generated_at: str,
) -> CustomerRiskRegister:
    """Build a risk register and readiness gate for customer factory promotion."""

    findings = _collect_findings(
        register_id=register_id,
        evidence_pack=evidence_pack,
        monitoring_manifest=monitoring_manifest,
        risks=risks,
        generated_at=generated_at,
    )
    register_hash = _stable_hash(
        {
            "register_id": register_id,
            "customer_id": evidence_pack.customer_id,
            "workspace_id": evidence_pack.workspace_id,
            "tenant_id": evidence_pack.tenant_id,
            "deployment_id": evidence_pack.deployment_id,
            "evidence_hash": evidence_pack.evidence_hash,
            "monitoring_hash": monitoring_manifest.monitoring_hash,
            "generated_at": generated_at,
            "risks": [risk.to_dict() for risk in risks],
            "findings": findings,
        }
    )
    return CustomerRiskRegister(
        schema_version=SCHEMA_VERSION,
        register_id=register_id,
        ready=not findings,
        customer_id=evidence_pack.customer_id,
        workspace_id=evidence_pack.workspace_id,
        tenant_id=evidence_pack.tenant_id,
        deployment_id=evidence_pack.deployment_id,
        evidence_hash=evidence_pack.evidence_hash,
        monitoring_hash=monitoring_manifest.monitoring_hash,
        generated_at=generated_at,
        risks=risks,
        findings=tuple(findings),
        register_hash=register_hash,
    )


def _collect_findings(
    *,
    register_id: str,
    evidence_pack: CustomerEvidencePackManifest,
    monitoring_manifest: CustomerMonitoringManifest,
    risks: tuple[CustomerRiskException, ...],
    generated_at: str,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if not register_id.strip():
        findings.append(_finding("register_id_missing", "register_id is required"))
    if not generated_at.strip():
        findings.append(_finding("generated_at_missing", "generated_at is required"))
    if not evidence_pack.ready:
        findings.append(
            _finding("evidence_pack_not_ready", "evidence pack is not ready")
        )
    if not monitoring_manifest.ready:
        findings.append(
            _finding(
                "monitoring_manifest_not_ready", "monitoring manifest is not ready"
            )
        )
    _extend_boundary_findings(evidence_pack, monitoring_manifest, findings)
    for risk in risks:
        _extend_risk_findings(
            risk=risk,
            evidence_hash=evidence_pack.evidence_hash,
            monitoring_hash=monitoring_manifest.monitoring_hash,
            generated_at=generated_at,
            findings=findings,
        )
    return findings


def _extend_boundary_findings(
    evidence_pack: CustomerEvidencePackManifest,
    monitoring_manifest: CustomerMonitoringManifest,
    findings: list[dict[str, str]],
) -> None:
    if evidence_pack.customer_id != monitoring_manifest.customer_id:
        findings.append(
            _finding(
                "customer_boundary_mismatch", "customer_id differs between artefacts"
            )
        )
    if evidence_pack.workspace_id != monitoring_manifest.workspace_id:
        findings.append(
            _finding(
                "workspace_boundary_mismatch", "workspace_id differs between artefacts"
            )
        )
    if evidence_pack.tenant_id != monitoring_manifest.tenant_id:
        findings.append(
            _finding("tenant_boundary_mismatch", "tenant_id differs between artefacts")
        )
    if evidence_pack.deployment_id != monitoring_manifest.deployment_id:
        findings.append(
            _finding(
                "deployment_boundary_mismatch",
                "deployment_id differs between artefacts",
            )
        )
    if evidence_pack.evidence_hash != monitoring_manifest.evidence_hash:
        findings.append(
            _finding(
                "evidence_hash_mismatch",
                "monitoring evidence_hash differs from evidence pack",
            )
        )


def _extend_risk_findings(
    *,
    risk: CustomerRiskException,
    evidence_hash: str,
    monitoring_hash: str,
    generated_at: str,
    findings: list[dict[str, str]],
) -> None:
    if risk.status not in ACCEPTED_STATUSES:
        findings.append(
            _risk_finding(risk, "risk_status_unknown", "risk status is unsupported")
        )
    if risk.status != "accepted":
        return
    if not risk.owner.strip():
        findings.append(
            _risk_finding(
                risk, "accepted_risk_owner_missing", "accepted risk requires an owner"
            )
        )
    if not risk.expires_at.strip():
        findings.append(
            _risk_finding(
                risk, "accepted_risk_expiry_missing", "accepted risk requires expiry"
            )
        )
    elif risk.expires_at < generated_at:
        findings.append(
            _risk_finding(risk, "accepted_risk_expired", "accepted risk is expired")
        )
    if not risk.compensating_controls:
        findings.append(
            _risk_finding(
                risk,
                "accepted_risk_controls_missing",
                "accepted risk requires compensating controls",
            )
        )
    if risk.linked_artifact_hashes.get("evidence_hash") != evidence_hash:
        findings.append(
            _risk_finding(
                risk,
                "risk_evidence_hash_mismatch",
                "risk is not linked to evidence hash",
            )
        )
    if risk.linked_artifact_hashes.get("monitoring_hash") != monitoring_hash:
        findings.append(
            _risk_finding(
                risk,
                "risk_monitoring_hash_mismatch",
                "risk is not linked to monitoring hash",
            )
        )


def _finding(code: str, message: str) -> dict[str, str]:
    return {"code": code, "severity": "error", "message": message}


def _risk_finding(
    risk: CustomerRiskException, code: str, message: str
) -> dict[str, str]:
    return {
        "code": code,
        "severity": "error",
        "message": message,
        "risk_id": risk.risk_id,
    }


def _stable_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()
