# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory benchmark selection tests

from __future__ import annotations

import json
from pathlib import Path

from director_ai.core.customer_model_factory.benchmark_selection import (
    BenchmarkMetrics,
    CustomerBenchmarkResult,
    CustomerModelSelectionReport,
    select_customer_model,
)
from director_ai.core.customer_model_factory.dataset_contract import (
    CustomerWorkspace,
    validate_customer_trace_dataset,
)
from director_ai.core.customer_model_factory.training_manifest import (
    TrainingLane,
    build_training_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def _workspace() -> CustomerWorkspace:
    return CustomerWorkspace(
        customer_id="customer-alpha",
        workspace_id="customer-alpha-prod",
        tenant_id="customer-alpha-tenant",
        data_classification="confidential",
        allowed_splits=("train", "eval", "test"),
        regulation_mappings=("SOC2", "ISO27001", "ISO42001", "EU_AI_ACT"),
    )


def _row(trace_id: str, split: str) -> dict:
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


def _training_manifest():
    report = validate_customer_trace_dataset(
        [
            _row("trace-001", "train"),
            _row("trace-002", "eval"),
            _row("trace-003", "test"),
        ],
        _workspace(),
        vertical_profile="regulated-sector",
    )
    return build_training_manifest(
        package_id="cmf-customer-alpha-20260518",
        dataset_report=report,
        lane=TrainingLane.VERTEX,
        base_model_id="microsoft/deberta-v3-small",
        base_model_revision="abcdef1234567890abcdef1234567890abcdef12",
        output_uri="gs://customer-artifacts/customer-alpha/models/cmf-customer-alpha-20260518",
        hyperparameters={"epochs": 3, "batch_size": 8, "learning_rate": 1e-5},
        objective_profile="zero_silent_unsafe_pass",
    )


def _metrics(
    *,
    recall: float = 0.96,
    false_positive_rate: float = 0.03,
    false_negative_rate: float = 0.0,
    high_risk_false_negative_rate: float = 0.0,
    abstention_rate: float = 0.08,
    escalation_rate: float = 0.12,
    f1: float = 0.92,
    latency_p95_ms: float = 42.0,
) -> BenchmarkMetrics:
    return BenchmarkMetrics(
        total_samples=240,
        balanced_accuracy=0.94,
        precision=0.91,
        recall=recall,
        f1=f1,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        high_risk_false_negative_rate=high_risk_false_negative_rate,
        abstention_rate=abstention_rate,
        escalation_rate=escalation_rate,
        latency_p95_ms=latency_p95_ms,
        severity_counts={"critical": 40, "high": 80, "medium": 80, "low": 40},
    )


def test_benchmark_result_is_ready_when_training_manifest_and_metrics_are_ready():
    result = CustomerBenchmarkResult.from_metrics(
        benchmark_id="customer-alpha-private-v1",
        training_manifest=_training_manifest(),
        model_artifact_uri="gs://customer-artifacts/customer-alpha/models/cmf-customer-alpha-20260518",
        metrics=_metrics(),
        raw_result_uri="gs://customer-artifacts/customer-alpha/benchmarks/private-v1.json",
        claim_boundary="Customer Alpha private guardrail validation only.",
    )

    assert result.ready is True
    assert result.findings == ()
    assert result.training_manifest_hash == _training_manifest().manifest_hash
    assert result.metrics.high_risk_false_negative_rate == 0.0
    assert len(result.result_hash) == 64


def test_benchmark_result_blocks_not_ready_training_manifest():
    invalid_training = build_training_manifest(
        package_id="invalid",
        dataset_report=validate_customer_trace_dataset(
            [_row("trace-001", "train")], _workspace()
        ),
        lane=TrainingLane.VERTEX,
        base_model_id="microsoft/deberta-v3-small",
        base_model_revision="",
        output_uri="file:///tmp/invalid",
        hyperparameters={},
        objective_profile="balanced",
    )

    result = CustomerBenchmarkResult.from_metrics(
        benchmark_id="customer-alpha-private-v1",
        training_manifest=invalid_training,
        model_artifact_uri="gs://customer-artifacts/customer-alpha/models/invalid",
        metrics=_metrics(),
        raw_result_uri="gs://customer-artifacts/customer-alpha/benchmarks/private-v1.json",
        claim_boundary="Customer Alpha private guardrail validation only.",
    )

    assert result.ready is False
    assert any(
        finding["code"] == "training_manifest_not_ready" for finding in result.findings
    )


def test_benchmark_result_blocks_incomplete_customer_evidence_claims():
    result = CustomerBenchmarkResult.from_metrics(
        benchmark_id=" ",
        training_manifest=_training_manifest(),
        model_artifact_uri=" ",
        metrics=BenchmarkMetrics(
            total_samples=0,
            balanced_accuracy=1.01,
            precision=0.91,
            recall=0.96,
            f1=0.92,
            false_positive_rate=0.03,
            false_negative_rate=0.0,
            high_risk_false_negative_rate=0.0,
            abstention_rate=0.08,
            escalation_rate=0.12,
            latency_p95_ms=42.0,
            severity_counts={},
        ),
        raw_result_uri=" ",
        claim_boundary=" ",
    )

    assert result.ready is False
    assert {finding["code"] for finding in result.findings} >= {
        "benchmark_id_missing",
        "model_artifact_missing",
        "raw_result_missing",
        "claim_boundary_missing",
        "empty_benchmark",
        "severity_counts_missing",
        "metric_out_of_range",
    }


def test_zero_silent_unsafe_pass_rejects_high_risk_false_negatives():
    bad_result = CustomerBenchmarkResult.from_metrics(
        benchmark_id="customer-alpha-private-v1",
        training_manifest=_training_manifest(),
        model_artifact_uri="gs://customer-artifacts/customer-alpha/models/cmf-customer-alpha-20260518",
        metrics=_metrics(high_risk_false_negative_rate=0.01),
        raw_result_uri="gs://customer-artifacts/customer-alpha/benchmarks/private-v1.json",
        claim_boundary="Customer Alpha private guardrail validation only.",
    )

    selection = select_customer_model(
        selection_id="customer-alpha-selection-20260518",
        objective_profile="zero_silent_unsafe_pass",
        candidates=[bad_result],
    )

    assert selection.ready is False
    assert selection.selected_benchmark_id == ""
    assert any(
        finding["code"] == "no_candidate_satisfies_objective"
        for finding in selection.findings
    )


def test_selection_blocks_unknown_objective_and_empty_candidate_set():
    selection = select_customer_model(
        selection_id=" ",
        objective_profile="unsupported-risk-objective",
        candidates=[],
    )

    assert selection.ready is False
    assert selection.selected_benchmark_id == ""
    assert {finding["code"] for finding in selection.findings} == {
        "selection_id_missing",
        "objective_profile_unknown",
        "candidates_missing",
    }


def test_selection_rejects_candidates_for_unknown_objective_profile():
    result = CustomerBenchmarkResult.from_metrics(
        benchmark_id="candidate-default",
        training_manifest=_training_manifest(),
        model_artifact_uri="gs://customer-artifacts/customer-alpha/models/default",
        metrics=_metrics(),
        raw_result_uri="gs://customer-artifacts/customer-alpha/benchmarks/default.json",
        claim_boundary="Customer Alpha private guardrail validation only.",
    )

    selection = select_customer_model(
        selection_id="customer-alpha-unsupported-selection",
        objective_profile="unsupported-risk-objective",
        candidates=[result],
    )

    assert selection.ready is False
    assert selection.selected_benchmark_id == ""
    assert any(
        finding["code"] == "objective_profile_unknown" for finding in selection.findings
    )


def test_selection_scores_high_recall_and_conservative_objectives():
    safer = CustomerBenchmarkResult.from_metrics(
        benchmark_id="candidate-safer",
        training_manifest=_training_manifest(),
        model_artifact_uri="gs://customer-artifacts/customer-alpha/models/safer",
        metrics=_metrics(
            f1=0.88,
            recall=0.95,
            false_positive_rate=0.01,
            false_negative_rate=0.01,
        ),
        raw_result_uri="gs://customer-artifacts/customer-alpha/benchmarks/safer.json",
        claim_boundary="Customer Alpha private guardrail validation only.",
    )
    noisier = CustomerBenchmarkResult.from_metrics(
        benchmark_id="candidate-noisier",
        training_manifest=_training_manifest(),
        model_artifact_uri="gs://customer-artifacts/customer-alpha/models/noisier",
        metrics=_metrics(
            f1=0.96,
            recall=0.97,
            false_positive_rate=0.08,
            false_negative_rate=0.08,
        ),
        raw_result_uri="gs://customer-artifacts/customer-alpha/benchmarks/noisier.json",
        claim_boundary="Customer Alpha private guardrail validation only.",
    )

    high_recall = select_customer_model(
        selection_id="customer-alpha-high-recall-selection",
        objective_profile="high_recall",
        candidates=[noisier, safer],
    )
    conservative = select_customer_model(
        selection_id="customer-alpha-conservative-selection",
        objective_profile="conservative",
        candidates=[noisier, safer],
    )

    assert high_recall.ready is True
    assert high_recall.selected_benchmark_id == "candidate-safer"
    assert conservative.ready is True
    assert conservative.selected_benchmark_id == "candidate-safer"


def test_selection_scores_balanced_objective_by_f1():
    lower_f1 = CustomerBenchmarkResult.from_metrics(
        benchmark_id="candidate-lower-f1",
        training_manifest=_training_manifest(),
        model_artifact_uri="gs://customer-artifacts/customer-alpha/models/lower-f1",
        metrics=_metrics(f1=0.81),
        raw_result_uri="gs://customer-artifacts/customer-alpha/benchmarks/lower-f1.json",
        claim_boundary="Customer Alpha private guardrail validation only.",
    )
    higher_f1 = CustomerBenchmarkResult.from_metrics(
        benchmark_id="candidate-higher-f1",
        training_manifest=_training_manifest(),
        model_artifact_uri="gs://customer-artifacts/customer-alpha/models/higher-f1",
        metrics=_metrics(f1=0.93),
        raw_result_uri="gs://customer-artifacts/customer-alpha/benchmarks/higher-f1.json",
        claim_boundary="Customer Alpha private guardrail validation only.",
    )

    selection = select_customer_model(
        selection_id="customer-alpha-balanced-selection",
        objective_profile="balanced",
        candidates=[lower_f1, higher_f1],
    )

    assert selection.ready is True
    assert selection.selected_benchmark_id == "candidate-higher-f1"


def test_model_selection_prefers_highest_objective_score_among_ready_candidates():
    slower = CustomerBenchmarkResult.from_metrics(
        benchmark_id="candidate-slower",
        training_manifest=_training_manifest(),
        model_artifact_uri="gs://customer-artifacts/customer-alpha/models/slower",
        metrics=_metrics(f1=0.94, latency_p95_ms=120.0),
        raw_result_uri="gs://customer-artifacts/customer-alpha/benchmarks/slower.json",
        claim_boundary="Customer Alpha private guardrail validation only.",
    )
    faster = CustomerBenchmarkResult.from_metrics(
        benchmark_id="candidate-faster",
        training_manifest=_training_manifest(),
        model_artifact_uri="gs://customer-artifacts/customer-alpha/models/faster",
        metrics=_metrics(f1=0.91, latency_p95_ms=25.0),
        raw_result_uri="gs://customer-artifacts/customer-alpha/benchmarks/faster.json",
        claim_boundary="Customer Alpha private guardrail validation only.",
    )

    selection = select_customer_model(
        selection_id="customer-alpha-selection-20260518",
        objective_profile="low_latency",
        candidates=[slower, faster],
    )

    assert selection.ready is True
    assert selection.selected_benchmark_id == "candidate-faster"
    assert (
        selection.selected_model_artifact_uri
        == "gs://customer-artifacts/customer-alpha/models/faster"
    )
    assert len(selection.selection_hash) == 64


def test_selection_report_serialises_and_writes_stable_json(tmp_path: Path):
    result = CustomerBenchmarkResult.from_metrics(
        benchmark_id="candidate-default",
        training_manifest=_training_manifest(),
        model_artifact_uri="gs://customer-artifacts/customer-alpha/models/default",
        metrics=_metrics(),
        raw_result_uri="gs://customer-artifacts/customer-alpha/benchmarks/default.json",
        claim_boundary="Customer Alpha private guardrail validation only.",
    )
    selection = select_customer_model(
        selection_id="customer-alpha-selection-20260518",
        objective_profile="high_recall",
        candidates=[result],
    )

    output = selection.write_json(tmp_path / "selection.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    restored = CustomerModelSelectionReport.from_dict(payload)

    assert payload == selection.to_dict()
    assert restored == selection
    assert output.name == "selection.json"


def test_benchmark_selection_schema_is_machine_readable():
    schema_path = ROOT / "schemas" / "customer-model-factory-selection.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == "DIRECTOR-AI Customer Model Factory Selection Report"
    assert set(schema["required"]) >= {
        "selection_id",
        "objective_profile",
        "selected_benchmark_id",
        "selected_model_artifact_uri",
        "selection_hash",
    }
