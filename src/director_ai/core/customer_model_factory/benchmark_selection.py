# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory benchmark selection

"""Benchmark result and model-selection primitives for customer packages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .training_manifest import OBJECTIVE_PROFILES, CustomerTrainingManifest

SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Severity-aware metrics required for customer model selection."""

    total_samples: int
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    high_risk_false_negative_rate: float
    abstention_rate: float
    escalation_rate: float
    latency_p95_ms: float
    severity_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Serialise metrics with stable key ordering for nested maps."""
        payload = asdict(self)
        payload["severity_counts"] = dict(sorted(self.severity_counts.items()))
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BenchmarkMetrics:
        """Rebuild metrics from a serialised dictionary."""
        return cls(
            total_samples=int(payload["total_samples"]),
            balanced_accuracy=float(payload["balanced_accuracy"]),
            precision=float(payload["precision"]),
            recall=float(payload["recall"]),
            f1=float(payload["f1"]),
            false_positive_rate=float(payload["false_positive_rate"]),
            false_negative_rate=float(payload["false_negative_rate"]),
            high_risk_false_negative_rate=float(
                payload["high_risk_false_negative_rate"]
            ),
            abstention_rate=float(payload["abstention_rate"]),
            escalation_rate=float(payload["escalation_rate"]),
            latency_p95_ms=float(payload["latency_p95_ms"]),
            severity_counts=dict(payload["severity_counts"]),
        )


@dataclass(frozen=True)
class CustomerBenchmarkResult:
    """One benchmark result bound to a customer training manifest."""

    schema_version: str
    benchmark_id: str
    ready: bool
    customer_id: str
    workspace_id: str
    tenant_id: str
    training_manifest_hash: str
    model_artifact_uri: str
    metrics: BenchmarkMetrics
    raw_result_uri: str
    claim_boundary: str
    findings: tuple[dict[str, str], ...]
    result_hash: str

    @classmethod
    def from_metrics(
        cls,
        *,
        benchmark_id: str,
        training_manifest: CustomerTrainingManifest,
        model_artifact_uri: str,
        metrics: BenchmarkMetrics,
        raw_result_uri: str,
        claim_boundary: str,
    ) -> CustomerBenchmarkResult:
        """Build and validate a benchmark result from measured metrics."""
        findings = _benchmark_findings(
            benchmark_id=benchmark_id,
            training_manifest=training_manifest,
            model_artifact_uri=model_artifact_uri,
            metrics=metrics,
            raw_result_uri=raw_result_uri,
            claim_boundary=claim_boundary,
        )
        result_hash = _stable_hash(
            {
                "benchmark_id": benchmark_id,
                "training_manifest_hash": training_manifest.manifest_hash,
                "model_artifact_uri": model_artifact_uri,
                "metrics": metrics.to_dict(),
                "raw_result_uri": raw_result_uri,
                "claim_boundary": claim_boundary,
                "findings": findings,
            }
        )
        return cls(
            schema_version=SCHEMA_VERSION,
            benchmark_id=benchmark_id,
            ready=not findings,
            customer_id=training_manifest.customer_id,
            workspace_id=training_manifest.workspace_id,
            tenant_id=training_manifest.tenant_id,
            training_manifest_hash=training_manifest.manifest_hash,
            model_artifact_uri=model_artifact_uri,
            metrics=metrics,
            raw_result_uri=raw_result_uri,
            claim_boundary=claim_boundary,
            findings=tuple(findings),
            result_hash=result_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the benchmark result to JSON-safe data."""
        return {
            "schema_version": self.schema_version,
            "benchmark_id": self.benchmark_id,
            "ready": self.ready,
            "customer_id": self.customer_id,
            "workspace_id": self.workspace_id,
            "tenant_id": self.tenant_id,
            "training_manifest_hash": self.training_manifest_hash,
            "model_artifact_uri": self.model_artifact_uri,
            "metrics": self.metrics.to_dict(),
            "raw_result_uri": self.raw_result_uri,
            "claim_boundary": self.claim_boundary,
            "findings": [dict(finding) for finding in self.findings],
            "result_hash": self.result_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CustomerBenchmarkResult:
        """Rebuild a benchmark result from a serialised dictionary."""
        return cls(
            schema_version=payload["schema_version"],
            benchmark_id=payload["benchmark_id"],
            ready=bool(payload["ready"]),
            customer_id=payload["customer_id"],
            workspace_id=payload["workspace_id"],
            tenant_id=payload["tenant_id"],
            training_manifest_hash=payload["training_manifest_hash"],
            model_artifact_uri=payload["model_artifact_uri"],
            metrics=BenchmarkMetrics.from_dict(payload["metrics"]),
            raw_result_uri=payload["raw_result_uri"],
            claim_boundary=payload["claim_boundary"],
            findings=tuple(dict(finding) for finding in payload["findings"]),
            result_hash=payload["result_hash"],
        )


@dataclass(frozen=True)
class CustomerModelSelectionReport:
    """Selection report for a customer deployable model candidate."""

    schema_version: str
    selection_id: str
    ready: bool
    objective_profile: str
    selected_benchmark_id: str
    selected_model_artifact_uri: str
    candidates: tuple[CustomerBenchmarkResult, ...]
    findings: tuple[dict[str, str], ...]
    selection_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise the selection report to stable JSON-safe data."""
        return {
            "schema_version": self.schema_version,
            "selection_id": self.selection_id,
            "ready": self.ready,
            "objective_profile": self.objective_profile,
            "selected_benchmark_id": self.selected_benchmark_id,
            "selected_model_artifact_uri": self.selected_model_artifact_uri,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "findings": [dict(finding) for finding in self.findings],
            "selection_hash": self.selection_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CustomerModelSelectionReport:
        """Rebuild a selection report from serialised data."""
        return cls(
            schema_version=payload["schema_version"],
            selection_id=payload["selection_id"],
            ready=bool(payload["ready"]),
            objective_profile=payload["objective_profile"],
            selected_benchmark_id=payload["selected_benchmark_id"],
            selected_model_artifact_uri=payload["selected_model_artifact_uri"],
            candidates=tuple(
                CustomerBenchmarkResult.from_dict(candidate)
                for candidate in payload["candidates"]
            ),
            findings=tuple(dict(finding) for finding in payload["findings"]),
            selection_hash=payload["selection_hash"],
        )

    def write_json(self, path: Path) -> Path:
        """Write the selection report as deterministic JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


def select_customer_model(
    *,
    selection_id: str,
    objective_profile: str,
    candidates: list[CustomerBenchmarkResult],
) -> CustomerModelSelectionReport:
    """Select a deployable customer model according to an objective profile."""
    findings = _selection_findings(selection_id, objective_profile, candidates)
    eligible = [
        candidate
        for candidate in candidates
        if candidate.ready
        and _candidate_satisfies_objective(candidate, objective_profile)
    ]
    selected = max(
        eligible,
        key=lambda candidate: _objective_score(candidate, objective_profile),
        default=None,
    )
    if selected is None and candidates and objective_profile in OBJECTIVE_PROFILES:
        findings.append(
            _finding(
                "no_candidate_satisfies_objective",
                "no ready benchmark candidate satisfies the requested objective",
            )
        )
    selected_benchmark_id = selected.benchmark_id if selected else ""
    selected_model_artifact_uri = selected.model_artifact_uri if selected else ""
    selection_hash = _stable_hash(
        {
            "selection_id": selection_id,
            "objective_profile": objective_profile,
            "selected_benchmark_id": selected_benchmark_id,
            "selected_model_artifact_uri": selected_model_artifact_uri,
            "candidates": [candidate.result_hash for candidate in candidates],
            "findings": findings,
        }
    )
    return CustomerModelSelectionReport(
        schema_version=SCHEMA_VERSION,
        selection_id=selection_id,
        ready=selected is not None and not findings,
        objective_profile=objective_profile,
        selected_benchmark_id=selected_benchmark_id,
        selected_model_artifact_uri=selected_model_artifact_uri,
        candidates=tuple(candidates),
        findings=tuple(findings),
        selection_hash=selection_hash,
    )


def _benchmark_findings(
    *,
    benchmark_id: str,
    training_manifest: CustomerTrainingManifest,
    model_artifact_uri: str,
    metrics: BenchmarkMetrics,
    raw_result_uri: str,
    claim_boundary: str,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if not benchmark_id.strip():
        findings.append(_finding("benchmark_id_missing", "benchmark_id is required"))
    if not training_manifest.ready:
        findings.append(
            _finding("training_manifest_not_ready", "training manifest is not ready")
        )
    if not model_artifact_uri.strip():
        findings.append(
            _finding("model_artifact_missing", "model_artifact_uri is required")
        )
    if not raw_result_uri.strip():
        findings.append(_finding("raw_result_missing", "raw_result_uri is required"))
    if not claim_boundary.strip():
        findings.append(
            _finding("claim_boundary_missing", "claim_boundary is required")
        )
    if metrics.total_samples <= 0:
        findings.append(_finding("empty_benchmark", "benchmark has no samples"))
    if not metrics.severity_counts:
        findings.append(
            _finding("severity_counts_missing", "severity_counts are required")
        )
    for name, value in metrics.to_dict().items():
        if name in {"total_samples", "latency_p95_ms", "severity_counts"}:
            continue
        if not 0.0 <= float(value) <= 1.0:
            findings.append(
                _finding("metric_out_of_range", f"{name} must be between 0 and 1")
            )
    return findings


def _selection_findings(
    selection_id: str,
    objective_profile: str,
    candidates: list[CustomerBenchmarkResult],
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if not selection_id.strip():
        findings.append(_finding("selection_id_missing", "selection_id is required"))
    if objective_profile not in OBJECTIVE_PROFILES:
        findings.append(
            _finding(
                "objective_profile_unknown",
                f"objective_profile {objective_profile!r} is not supported",
            )
        )
    if not candidates:
        findings.append(
            _finding("candidates_missing", "at least one candidate is required")
        )
    return findings


def _candidate_satisfies_objective(
    candidate: CustomerBenchmarkResult,
    objective_profile: str,
) -> bool:
    metrics = candidate.metrics
    if objective_profile == "zero_silent_unsafe_pass":
        return metrics.high_risk_false_negative_rate == 0.0
    if objective_profile == "high_recall":
        return metrics.recall >= 0.90
    if objective_profile == "low_latency":
        return metrics.latency_p95_ms <= 1000.0
    if objective_profile in {"balanced", "conservative"}:
        return metrics.total_samples > 0
    return False


def _objective_score(
    candidate: CustomerBenchmarkResult, objective_profile: str
) -> float:
    metrics = candidate.metrics
    if objective_profile == "low_latency":
        return (1.0 / max(metrics.latency_p95_ms, 1.0)) + (metrics.f1 / 100.0)
    if objective_profile == "high_recall":
        return metrics.recall - metrics.false_negative_rate
    if objective_profile == "zero_silent_unsafe_pass":
        return metrics.f1 - metrics.high_risk_false_negative_rate
    if objective_profile == "conservative":
        return metrics.recall - metrics.false_positive_rate
    return metrics.f1


def _finding(code: str, message: str) -> dict[str, str]:
    return {"code": code, "severity": "error", "message": message}


def _stable_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()
