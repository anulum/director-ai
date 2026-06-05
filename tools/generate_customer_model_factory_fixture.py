# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory end-to-end fixture generator

"""Generate a deterministic Customer Model Factory end-to-end fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from director_ai.core.customer_model_factory.benchmark_selection import (
    BenchmarkMetrics,
    CustomerBenchmarkResult,
    select_customer_model,
)
from director_ai.core.customer_model_factory.dataset_contract import (
    CustomerWorkspace,
    validate_customer_trace_dataset,
)
from director_ai.core.customer_model_factory.deployment_manifest import (
    DeploymentPolicy,
    build_deployment_manifest,
)
from director_ai.core.customer_model_factory.evidence_pack import (
    build_customer_evidence_pack,
)
from director_ai.core.customer_model_factory.monitoring_manifest import (
    MonitoringMetrics,
    MonitoringThresholds,
    build_monitoring_manifest,
)
from director_ai.core.customer_model_factory.release_gate import (
    DeploymentHardeningEvidence,
    build_release_gate_manifest,
)
from director_ai.core.customer_model_factory.risk_register import build_risk_register
from director_ai.core.customer_model_factory.runtime_package import (
    build_customer_runtime_package,
)
from director_ai.core.customer_model_factory.sector_extension import (
    build_sector_evidence_mapping,
)
from director_ai.core.customer_model_factory.training_manifest import (
    TrainingLane,
    build_training_manifest,
)

GENERATED_AT = "2026-05-18T18:45:00Z"


def main(argv: list[str] | None = None) -> int:
    """Run the deterministic fixture generator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = _build_manifests()
    for filename, payload in manifests.items():
        (args.output_dir / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


def _build_manifests() -> dict[str, dict]:
    workspace = CustomerWorkspace(
        customer_id="customer-alpha",
        workspace_id="customer-alpha-prod",
        tenant_id="customer-alpha-tenant",
        data_classification="restricted",
        allowed_splits=("train", "eval", "test"),
        regulation_mappings=("SOC2", "ISO27001", "ISO42001", "EU_AI_ACT", "FINMA"),
    )
    dataset_report = validate_customer_trace_dataset(
        [
            _row("trace-001", "train"),
            _row("trace-002", "eval"),
            _row("trace-003", "test"),
        ],
        workspace,
        vertical_profile="regulated-sector",
    )
    training_manifest = build_training_manifest(
        package_id="cmf-customer-alpha-20260518",
        dataset_report=dataset_report,
        lane=TrainingLane.VERTEX,
        base_model_id="microsoft/deberta-v3-small",
        base_model_revision="abcdef1234567890abcdef1234567890abcdef12",
        output_uri="gs://customer-artifacts/customer-alpha/models/cmf-customer-alpha-20260518",
        hyperparameters={"batch_size": 8, "epochs": 3, "learning_rate": 1e-5},
        objective_profile="zero_silent_unsafe_pass",
    )
    benchmark_result = CustomerBenchmarkResult.from_metrics(
        benchmark_id="customer-alpha-private-v1",
        training_manifest=training_manifest,
        model_artifact_uri="gs://customer-artifacts/customer-alpha/models/cmf-customer-alpha-20260518",
        metrics=BenchmarkMetrics(
            total_samples=240,
            balanced_accuracy=0.94,
            precision=0.91,
            recall=0.96,
            f1=0.92,
            false_positive_rate=0.03,
            false_negative_rate=0.0,
            high_risk_false_negative_rate=0.0,
            abstention_rate=0.08,
            escalation_rate=0.12,
            latency_p95_ms=42.0,
            severity_counts={"critical": 40, "high": 80, "low": 40, "medium": 80},
        ),
        raw_result_uri="gs://customer-artifacts/customer-alpha/benchmarks/private-v1.json",
        claim_boundary="Customer Alpha private guardrail validation only.",
    )
    selection_report = select_customer_model(
        selection_id="customer-alpha-selection-20260518",
        objective_profile="zero_silent_unsafe_pass",
        candidates=[benchmark_result],
    )
    deployment_manifest = build_deployment_manifest(
        deployment_id="customer-alpha-prod-20260518",
        selection_report=selection_report,
        policy=DeploymentPolicy(
            threshold=0.72,
            abstention_threshold=0.58,
            escalation_threshold=0.40,
            require_citations=True,
            audit_log_uri="gs://customer-artifacts/customer-alpha/audit/decision-log.jsonl",
            evidence_pack_uri="gs://customer-artifacts/customer-alpha/evidence/pack-20260518",
            rollback_package_uri="gs://customer-artifacts/customer-alpha/deployments/previous.json",
            retention_days=365,
            telemetry_mode="customer_controlled",
        ),
        environment="production",
        package_uri="gs://customer-artifacts/customer-alpha/deployments/customer-alpha-prod-20260518.json",
    )
    sector_mapping = build_sector_evidence_mapping(
        sector_id="regulated-sector",
        jurisdiction="CH",
        evidence_pack_uri="gs://customer-artifacts/customer-alpha/evidence/pack-20260518",
    )
    evidence_pack = build_customer_evidence_pack(
        package_id="evidence-customer-alpha-20260518",
        deployment_manifest=deployment_manifest,
        regulation_mapping=sector_mapping,
        classification="restricted",
        export_uri="gs://customer-artifacts/customer-alpha/evidence/pack-20260518",
    )
    runtime_package = build_customer_runtime_package(
        runtime_id="runtime-customer-alpha-20260518",
        deployment_manifest=deployment_manifest,
        evidence_pack=evidence_pack,
        runtime_mode="offline_private",
    )
    monitoring_manifest = build_monitoring_manifest(
        monitoring_id="monitor-customer-alpha-20260518",
        runtime_package=runtime_package,
        metrics=MonitoringMetrics(
            total_decisions=1200,
            input_drift_score=0.04,
            source_corpus_drift_score=0.03,
            false_positive_review_count=2,
            false_negative_incident_count=0,
            abstention_rate=0.05,
            escalation_rate=0.09,
            latency_p95_ms=48.0,
            cost_per_1k_decisions=0.12,
        ),
        thresholds=MonitoringThresholds(
            max_input_drift_score=0.20,
            max_source_corpus_drift_score=0.15,
            max_false_negative_incidents=0,
            max_abstention_rate=0.20,
            max_escalation_rate=0.30,
            max_latency_p95_ms=250.0,
            max_cost_per_1k_decisions=1.50,
        ),
        observation_window="2026-05-18T00:00:00Z/2026-05-18T12:00:00Z",
        monitored_at="2026-05-18T12:05:00Z",
        review_queue_uri="gs://customer-artifacts/customer-alpha/review/fp.jsonl",
        incident_queue_uri="gs://customer-artifacts/customer-alpha/incidents/fn.jsonl",
    )
    risk_register = build_risk_register(
        register_id="risk_register_customer_alpha-20260518",
        evidence_pack=evidence_pack,
        monitoring_manifest=monitoring_manifest,
        risks=(),
        generated_at="2026-05-18",
    )
    enterprise_readiness = {"ready": True, "blocking_debt_ids": []}
    deployment_hardening_evidence = DeploymentHardeningEvidence(
        ready=True,
        environment="staging",
        observation_window="2026-05-18T00:00:00Z/2026-05-18T12:00:00Z",
        telemetry_uri="gs://customer-artifacts/customer-alpha/telemetry/r17.jsonl",
        sustained_load_packet_uri="gs://customer-artifacts/customer-alpha/evidence/sustained-load.json",
        operator_signoff_uri="gs://customer-artifacts/customer-alpha/signoff/r17.json",
        async_ordering_passed=True,
        tenant_poisoning_passed=True,
        evidence_hash="f" * 64,
    )
    release_gate = build_release_gate_manifest(
        release_id="release-customer-alpha-20260518",
        enterprise_ready=True,
        enterprise_blocking_debt_ids=(),
        runtime_package=runtime_package,
        evidence_pack=evidence_pack,
        monitoring_manifest=monitoring_manifest,
        risk_register=risk_register,
        deployment_hardening_evidence=deployment_hardening_evidence,
        generated_at=GENERATED_AT,
    )
    return {
        "dataset_report.json": dataset_report.to_dict(),
        "training_manifest.json": training_manifest.to_dict(),
        "benchmark_result.json": benchmark_result.to_dict(),
        "selection_report.json": selection_report.to_dict(),
        "deployment_manifest.json": deployment_manifest.to_dict(),
        "sector_evidence_mapping.json": sector_mapping.to_dict(),
        "evidence_pack.json": evidence_pack.to_dict(),
        "runtime_package.json": runtime_package.to_dict(),
        "monitoring_manifest.json": monitoring_manifest.to_dict(),
        "risk_register.json": risk_register.to_dict(),
        "deployment_hardening_evidence.json": deployment_hardening_evidence.to_dict(),
        "enterprise_readiness.json": enterprise_readiness,
        "release_gate.json": release_gate.to_dict(),
    }


def _row(trace_id: str, split: str) -> dict[str, object]:
    return {
        "trace_id": trace_id,
        "customer_id": "customer-alpha",
        "tenant_id": "customer-alpha-tenant",
        "split": split,
        "prompt": f"Review customer communication {trace_id}",
        "response": f"Escalate {trace_id} to compliance.",
        "expected_decision": "escalate",
        "severity": "high",
        "label": "policy_violation",
        "source_refs": [f"policy://customer-alpha/{trace_id}"],
        "policy_refs": ["policy://customer-alpha/advice-boundary"],
        "reviewer_role": "compliance_reviewer",
        "observed_at": "2026-05-18T12:00:00Z",
        "contains_pii": False,
        "contains_secret": False,
        "redaction_status": "not_required",
        "metadata": {
            "sector_class": "customer_policy",
            "knowledge_class": "advice_boundary",
            "requires_citation": True,
            "jurisdiction": "CH",
            "evidence_refs": ["policy://customer-alpha/advice-boundary"],
            "numeric_evidence_refs": [],
            "requires_escalation": True,
            "customer_segment": "retail",
            "product_family": "mortgage",
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
