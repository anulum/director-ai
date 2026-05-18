# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory monitoring manifest

"""Monitoring and retraining evidence for customer runtime packages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .runtime_package import CustomerRuntimePackage

SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class MonitoringMetrics:
    """Observed runtime metrics for one customer monitoring window."""

    total_decisions: int
    input_drift_score: float
    source_corpus_drift_score: float
    false_positive_review_count: int
    false_negative_incident_count: int
    abstention_rate: float
    escalation_rate: float
    latency_p95_ms: float
    cost_per_1k_decisions: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise metrics to JSON-safe data."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MonitoringMetrics:
        """Rebuild metrics from JSON-safe data."""

        return cls(
            total_decisions=int(payload["total_decisions"]),
            input_drift_score=float(payload["input_drift_score"]),
            source_corpus_drift_score=float(payload["source_corpus_drift_score"]),
            false_positive_review_count=int(payload["false_positive_review_count"]),
            false_negative_incident_count=int(payload["false_negative_incident_count"]),
            abstention_rate=float(payload["abstention_rate"]),
            escalation_rate=float(payload["escalation_rate"]),
            latency_p95_ms=float(payload["latency_p95_ms"]),
            cost_per_1k_decisions=float(payload["cost_per_1k_decisions"]),
        )


@dataclass(frozen=True)
class MonitoringThresholds:
    """Customer monitoring thresholds that trigger review or retraining."""

    max_input_drift_score: float
    max_source_corpus_drift_score: float
    max_false_negative_incidents: int
    max_abstention_rate: float
    max_escalation_rate: float
    max_latency_p95_ms: float
    max_cost_per_1k_decisions: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise thresholds to JSON-safe data."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MonitoringThresholds:
        """Rebuild thresholds from JSON-safe data."""

        return cls(
            max_input_drift_score=float(payload["max_input_drift_score"]),
            max_source_corpus_drift_score=float(
                payload["max_source_corpus_drift_score"]
            ),
            max_false_negative_incidents=int(payload["max_false_negative_incidents"]),
            max_abstention_rate=float(payload["max_abstention_rate"]),
            max_escalation_rate=float(payload["max_escalation_rate"]),
            max_latency_p95_ms=float(payload["max_latency_p95_ms"]),
            max_cost_per_1k_decisions=float(payload["max_cost_per_1k_decisions"]),
        )


@dataclass(frozen=True)
class CustomerMonitoringManifest:
    """Monitoring manifest for customer runtime evidence and retraining decisions."""

    schema_version: str
    monitoring_id: str
    ready: bool
    health_status: str
    customer_id: str
    workspace_id: str
    tenant_id: str
    runtime_id: str
    deployment_id: str
    evidence_hash: str
    decision_log_uri: str
    review_queue_uri: str
    incident_queue_uri: str
    observation_window: str
    monitored_at: str
    metrics: MonitoringMetrics
    thresholds: MonitoringThresholds
    package_version: dict[str, str]
    retraining_recommended: bool
    recommendations: tuple[dict[str, str], ...]
    findings: tuple[dict[str, str], ...]
    monitoring_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise the monitoring manifest to stable JSON-safe data."""

        return {
            "schema_version": self.schema_version,
            "monitoring_id": self.monitoring_id,
            "ready": self.ready,
            "health_status": self.health_status,
            "customer_id": self.customer_id,
            "workspace_id": self.workspace_id,
            "tenant_id": self.tenant_id,
            "runtime_id": self.runtime_id,
            "deployment_id": self.deployment_id,
            "evidence_hash": self.evidence_hash,
            "decision_log_uri": self.decision_log_uri,
            "review_queue_uri": self.review_queue_uri,
            "incident_queue_uri": self.incident_queue_uri,
            "observation_window": self.observation_window,
            "monitored_at": self.monitored_at,
            "metrics": self.metrics.to_dict(),
            "thresholds": self.thresholds.to_dict(),
            "package_version": dict(sorted(self.package_version.items())),
            "retraining_recommended": self.retraining_recommended,
            "recommendations": [dict(item) for item in self.recommendations],
            "findings": [dict(finding) for finding in self.findings],
            "monitoring_hash": self.monitoring_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CustomerMonitoringManifest:
        """Rebuild a monitoring manifest from JSON-safe data."""

        return cls(
            schema_version=payload["schema_version"],
            monitoring_id=payload["monitoring_id"],
            ready=bool(payload["ready"]),
            health_status=payload["health_status"],
            customer_id=payload["customer_id"],
            workspace_id=payload["workspace_id"],
            tenant_id=payload["tenant_id"],
            runtime_id=payload["runtime_id"],
            deployment_id=payload["deployment_id"],
            evidence_hash=payload["evidence_hash"],
            decision_log_uri=payload["decision_log_uri"],
            review_queue_uri=payload["review_queue_uri"],
            incident_queue_uri=payload["incident_queue_uri"],
            observation_window=payload["observation_window"],
            monitored_at=payload["monitored_at"],
            metrics=MonitoringMetrics.from_dict(payload["metrics"]),
            thresholds=MonitoringThresholds.from_dict(payload["thresholds"]),
            package_version=dict(payload["package_version"]),
            retraining_recommended=bool(payload["retraining_recommended"]),
            recommendations=tuple(dict(item) for item in payload["recommendations"]),
            findings=tuple(dict(finding) for finding in payload["findings"]),
            monitoring_hash=payload["monitoring_hash"],
        )

    def write_json(self, path: Path) -> Path:
        """Write the monitoring manifest as deterministic JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


def build_monitoring_manifest(
    *,
    monitoring_id: str,
    runtime_package: CustomerRuntimePackage,
    metrics: MonitoringMetrics,
    thresholds: MonitoringThresholds,
    observation_window: str,
    monitored_at: str,
    review_queue_uri: str,
    incident_queue_uri: str,
) -> CustomerMonitoringManifest:
    """Build a monitoring manifest and retraining recommendation set."""

    decision_log_uri = str(runtime_package.runtime_config.get("audit_log_uri", ""))
    findings = _collect_findings(
        monitoring_id=monitoring_id,
        runtime_package=runtime_package,
        decision_log_uri=decision_log_uri,
        observation_window=observation_window,
        monitored_at=monitored_at,
        review_queue_uri=review_queue_uri,
        incident_queue_uri=incident_queue_uri,
    )
    recommendations = _recommendations(metrics, thresholds)
    health_status = _health_status(findings, recommendations)
    package_version = _package_version(runtime_package)
    monitoring_hash = _stable_hash(
        {
            "monitoring_id": monitoring_id,
            "health_status": health_status,
            "customer_id": runtime_package.customer_id,
            "workspace_id": runtime_package.workspace_id,
            "tenant_id": runtime_package.tenant_id,
            "runtime_id": runtime_package.runtime_id,
            "deployment_id": runtime_package.deployment_id,
            "evidence_hash": runtime_package.evidence_hash,
            "decision_log_uri": decision_log_uri,
            "review_queue_uri": review_queue_uri,
            "incident_queue_uri": incident_queue_uri,
            "observation_window": observation_window,
            "monitored_at": monitored_at,
            "metrics": metrics.to_dict(),
            "thresholds": thresholds.to_dict(),
            "package_version": package_version,
            "recommendations": recommendations,
            "findings": findings,
        }
    )
    return CustomerMonitoringManifest(
        schema_version=SCHEMA_VERSION,
        monitoring_id=monitoring_id,
        ready=not findings,
        health_status=health_status,
        customer_id=runtime_package.customer_id,
        workspace_id=runtime_package.workspace_id,
        tenant_id=runtime_package.tenant_id,
        runtime_id=runtime_package.runtime_id,
        deployment_id=runtime_package.deployment_id,
        evidence_hash=runtime_package.evidence_hash,
        decision_log_uri=decision_log_uri,
        review_queue_uri=review_queue_uri,
        incident_queue_uri=incident_queue_uri,
        observation_window=observation_window,
        monitored_at=monitored_at,
        metrics=metrics,
        thresholds=thresholds,
        package_version=package_version,
        retraining_recommended=bool(recommendations),
        recommendations=tuple(recommendations),
        findings=tuple(findings),
        monitoring_hash=monitoring_hash,
    )


def _collect_findings(
    *,
    monitoring_id: str,
    runtime_package: CustomerRuntimePackage,
    decision_log_uri: str,
    observation_window: str,
    monitored_at: str,
    review_queue_uri: str,
    incident_queue_uri: str,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if not monitoring_id.strip():
        findings.append(_finding("monitoring_id_missing", "monitoring_id is required"))
    if not runtime_package.ready:
        findings.append(
            _finding("runtime_package_not_ready", "runtime package is not ready")
        )
    if runtime_package.runtime_config.get("customer_id") != runtime_package.customer_id:
        findings.append(
            _finding(
                "customer_boundary_mismatch",
                "customer_id differs inside runtime config",
            )
        )
    if (
        runtime_package.runtime_config.get("workspace_id")
        != runtime_package.workspace_id
    ):
        findings.append(
            _finding(
                "workspace_boundary_mismatch",
                "workspace_id differs inside runtime config",
            )
        )
    if runtime_package.runtime_config.get("tenant_id") != runtime_package.tenant_id:
        findings.append(
            _finding(
                "tenant_boundary_mismatch", "tenant_id differs inside runtime config"
            )
        )
    if (
        runtime_package.runtime_config.get("deployment_id")
        != runtime_package.deployment_id
    ):
        findings.append(
            _finding(
                "deployment_boundary_mismatch",
                "deployment_id differs inside runtime config",
            )
        )
    if not decision_log_uri.strip():
        findings.append(
            _finding(
                "decision_log_missing",
                "decision_log_uri is required before health status",
            )
        )
    if not observation_window.strip():
        findings.append(
            _finding("observation_window_missing", "observation_window is required")
        )
    if not monitored_at.strip():
        findings.append(_finding("monitored_at_missing", "monitored_at is required"))
    if not review_queue_uri.strip():
        findings.append(
            _finding("review_queue_missing", "review_queue_uri is required")
        )
    if not incident_queue_uri.strip():
        findings.append(
            _finding("incident_queue_missing", "incident_queue_uri is required")
        )
    return findings


def _recommendations(
    metrics: MonitoringMetrics,
    thresholds: MonitoringThresholds,
) -> list[dict[str, str]]:
    recommendations: list[dict[str, str]] = []
    if metrics.input_drift_score > thresholds.max_input_drift_score:
        recommendations.append(_recommendation("input_drift_threshold_breached"))
    if metrics.source_corpus_drift_score > thresholds.max_source_corpus_drift_score:
        recommendations.append(
            _recommendation("source_corpus_drift_threshold_breached")
        )
    if metrics.false_negative_incident_count > thresholds.max_false_negative_incidents:
        recommendations.append(
            _recommendation("false_negative_incident_threshold_breached")
        )
    if metrics.abstention_rate > thresholds.max_abstention_rate:
        recommendations.append(_recommendation("abstention_threshold_breached"))
    if metrics.escalation_rate > thresholds.max_escalation_rate:
        recommendations.append(_recommendation("escalation_threshold_breached"))
    if metrics.latency_p95_ms > thresholds.max_latency_p95_ms:
        recommendations.append(_recommendation("latency_threshold_breached"))
    if metrics.cost_per_1k_decisions > thresholds.max_cost_per_1k_decisions:
        recommendations.append(_recommendation("cost_threshold_breached"))
    return recommendations


def _package_version(runtime_package: CustomerRuntimePackage) -> dict[str, str]:
    return {
        "runtime_id": runtime_package.runtime_id,
        "runtime_hash": runtime_package.runtime_hash,
        "deployment_id": runtime_package.deployment_id,
        "deployment_hash": str(
            runtime_package.runtime_config.get("deployment_hash", "")
        ),
        "evidence_hash": runtime_package.evidence_hash,
        "selected_model_artifact_uri": str(
            runtime_package.runtime_config.get("selected_model_artifact_uri", "")
        ),
    }


def _health_status(
    findings: list[dict[str, str]],
    recommendations: list[dict[str, str]],
) -> str:
    if findings:
        return "evidence_blocked"
    if recommendations:
        return "review_required"
    return "within_control"


def _finding(code: str, message: str) -> dict[str, str]:
    return {"code": code, "severity": "error", "message": message}


def _recommendation(code: str) -> dict[str, str]:
    return {"code": code, "severity": "review", "message": code.replace("_", " ")}


def _stable_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()
