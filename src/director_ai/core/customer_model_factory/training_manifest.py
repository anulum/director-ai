# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory training manifest

"""Training package manifest for customer-owned scorer models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .dataset_contract import CustomerDatasetValidationReport

SCHEMA_VERSION = "1.0.0"
OBJECTIVE_PROFILES = frozenset(
    {
        "conservative",
        "balanced",
        "low_latency",
        "high_recall",
        "zero_silent_unsafe_pass",
    }
)


class TrainingLane(StrEnum):
    """Supported execution lanes for customer training packages."""

    VERTEX = "vertex"
    CUSTOMER_CLOUD = "customer_cloud"
    ON_PREM = "on_prem"
    LOCAL_PILOT = "local_pilot"


@dataclass(frozen=True)
class CustomerTrainingManifest:
    """Immutable description of a customer model training package."""

    schema_version: str
    package_id: str
    ready: bool
    customer_id: str
    workspace_id: str
    tenant_id: str
    dataset_hash: str
    dataset_ready: bool
    lane: TrainingLane
    base_model_id: str
    base_model_revision: str
    base_model_artifact_uri: str
    output_uri: str
    hyperparameters: dict[str, Any]
    objective_profile: str
    requires_private_execution: bool
    findings: tuple[dict[str, str], ...]
    manifest_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise the manifest to a stable JSON-safe dictionary."""
        return {
            "schema_version": self.schema_version,
            "package_id": self.package_id,
            "ready": self.ready,
            "customer_id": self.customer_id,
            "workspace_id": self.workspace_id,
            "tenant_id": self.tenant_id,
            "dataset_hash": self.dataset_hash,
            "dataset_ready": self.dataset_ready,
            "lane": self.lane.value,
            "base_model_id": self.base_model_id,
            "base_model_revision": self.base_model_revision,
            "base_model_artifact_uri": self.base_model_artifact_uri,
            "output_uri": self.output_uri,
            "hyperparameters": dict(sorted(self.hyperparameters.items())),
            "objective_profile": self.objective_profile,
            "requires_private_execution": self.requires_private_execution,
            "findings": [dict(finding) for finding in self.findings],
            "manifest_hash": self.manifest_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CustomerTrainingManifest:
        """Rebuild a manifest from its serialised dictionary shape."""
        return cls(
            schema_version=payload["schema_version"],
            package_id=payload["package_id"],
            ready=bool(payload["ready"]),
            customer_id=payload["customer_id"],
            workspace_id=payload["workspace_id"],
            tenant_id=payload["tenant_id"],
            dataset_hash=payload["dataset_hash"],
            dataset_ready=bool(payload["dataset_ready"]),
            lane=TrainingLane(payload["lane"]),
            base_model_id=payload["base_model_id"],
            base_model_revision=payload["base_model_revision"],
            base_model_artifact_uri=payload["base_model_artifact_uri"],
            output_uri=payload["output_uri"],
            hyperparameters=dict(payload["hyperparameters"]),
            objective_profile=payload["objective_profile"],
            requires_private_execution=bool(payload["requires_private_execution"]),
            findings=tuple(dict(finding) for finding in payload["findings"]),
            manifest_hash=payload["manifest_hash"],
        )

    def write_json(self, path: Path) -> Path:
        """Write the manifest as deterministic JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


def build_training_manifest(
    *,
    package_id: str,
    dataset_report: CustomerDatasetValidationReport,
    lane: TrainingLane,
    base_model_id: str,
    base_model_revision: str,
    output_uri: str,
    hyperparameters: dict[str, Any],
    objective_profile: str,
) -> CustomerTrainingManifest:
    """Build and validate a customer training package manifest."""
    base_model_artifact_uri = base_model_id if base_model_id.startswith("gs://") else ""
    findings = _collect_findings(
        package_id=package_id,
        dataset_report=dataset_report,
        lane=lane,
        base_model_id=base_model_id,
        base_model_revision=base_model_revision,
        output_uri=output_uri,
        hyperparameters=hyperparameters,
        objective_profile=objective_profile,
    )
    ready = not findings
    manifest_hash = _manifest_hash(
        {
            "package_id": package_id,
            "customer_id": dataset_report.customer_id,
            "workspace_id": dataset_report.workspace_id,
            "tenant_id": dataset_report.tenant_id,
            "dataset_hash": dataset_report.dataset_hash,
            "lane": lane.value,
            "base_model_id": base_model_id,
            "base_model_revision": base_model_revision,
            "base_model_artifact_uri": base_model_artifact_uri,
            "output_uri": output_uri,
            "hyperparameters": hyperparameters,
            "objective_profile": objective_profile,
            "findings": findings,
        }
    )
    return CustomerTrainingManifest(
        schema_version=SCHEMA_VERSION,
        package_id=package_id,
        ready=ready,
        customer_id=dataset_report.customer_id,
        workspace_id=dataset_report.workspace_id,
        tenant_id=dataset_report.tenant_id,
        dataset_hash=dataset_report.dataset_hash,
        dataset_ready=dataset_report.ready,
        lane=lane,
        base_model_id=base_model_id,
        base_model_revision=base_model_revision,
        base_model_artifact_uri=base_model_artifact_uri,
        output_uri=output_uri,
        hyperparameters=dict(sorted(hyperparameters.items())),
        objective_profile=objective_profile,
        requires_private_execution=lane
        in {TrainingLane.VERTEX, TrainingLane.CUSTOMER_CLOUD, TrainingLane.ON_PREM},
        findings=tuple(findings),
        manifest_hash=manifest_hash,
    )


def _collect_findings(
    *,
    package_id: str,
    dataset_report: CustomerDatasetValidationReport,
    lane: TrainingLane,
    base_model_id: str,
    base_model_revision: str,
    output_uri: str,
    hyperparameters: dict[str, Any],
    objective_profile: str,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if not package_id.strip():
        findings.append(_finding("package_id_missing", "package_id is required"))
    if not dataset_report.ready:
        findings.append(
            _finding("dataset_not_ready", "dataset validation report is not ready")
        )
    if not base_model_id.strip():
        findings.append(_finding("base_model_missing", "base_model_id is required"))
    if not _base_model_is_immutable(base_model_id, base_model_revision):
        findings.append(
            _finding(
                "base_model_not_immutable",
                "base model must use an immutable revision or managed artefact URI",
            )
        )
    if not _output_uri_matches_lane(lane, output_uri):
        findings.append(
            _finding(
                "output_uri_incompatible",
                f"output_uri {output_uri!r} is incompatible with lane {lane.value!r}",
            )
        )
    if not hyperparameters:
        findings.append(
            _finding("hyperparameters_missing", "hyperparameters are required")
        )
    if objective_profile not in OBJECTIVE_PROFILES:
        findings.append(
            _finding(
                "objective_profile_unknown",
                f"objective_profile {objective_profile!r} is not supported",
            )
        )
    return findings


def _base_model_is_immutable(base_model_id: str, base_model_revision: str) -> bool:
    if base_model_id.startswith("gs://"):
        return True
    if base_model_revision.startswith("local-sha256:"):
        return True
    return len(base_model_revision) >= 12 and all(
        char in "0123456789abcdefABCDEF" for char in base_model_revision
    )


def _output_uri_matches_lane(lane: TrainingLane, output_uri: str) -> bool:
    if lane in {TrainingLane.VERTEX, TrainingLane.CUSTOMER_CLOUD}:
        return output_uri.startswith("gs://") or output_uri.startswith("s3://")
    if lane in {TrainingLane.ON_PREM, TrainingLane.LOCAL_PILOT}:
        return output_uri.startswith("file://") or output_uri.startswith("/")
    return False  # pragma: no cover — every TrainingLane is handled above


def _finding(code: str, message: str) -> dict[str, str]:
    return {"code": code, "severity": "error", "message": message}


def _manifest_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()
