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
        customer_id="bank-alpha",
        workspace_id="bank-alpha-prod",
        tenant_id="bank-alpha-tenant",
        data_classification="confidential",
        allowed_splits=("train", "eval", "test"),
        regulation_mappings=("SOC2", "ISO27001", "ISO42001", "EU_AI_ACT"),
    )


def _row(trace_id: str, split: str) -> dict:
    return {
        "trace_id": trace_id,
        "customer_id": "bank-alpha",
        "tenant_id": "bank-alpha-tenant",
        "split": split,
        "prompt": f"Review bank communication {trace_id}",
        "response": f"Escalate {trace_id} to compliance.",
        "expected_decision": "escalate",
        "severity": "high",
        "label": "policy_violation",
        "source_refs": [f"policy://bank-alpha/{trace_id}"],
        "policy_refs": ["policy://bank-alpha/advice-boundary"],
        "reviewer_role": "compliance_reviewer",
        "observed_at": "2026-05-18T12:00:00Z",
        "contains_pii": False,
        "contains_secret": False,
        "redaction_status": "not_required",
        "metadata": {
            "business_line": "retail_banking",
            "regulated_category": "financial_advice_boundary",
            "requires_citation": True,
            "jurisdiction": "CH",
            "evidence_refs": ["policy://bank-alpha/advice-boundary"],
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
        vertical_profile="banking",
    )
    return build_training_manifest(
        package_id="cmf-bank-alpha-20260518",
        dataset_report=report,
        lane=TrainingLane.VERTEX,
        base_model_id="microsoft/deberta-v3-small",
        base_model_revision="abcdef1234567890abcdef1234567890abcdef12",
        output_uri="gs://customer-artifacts/bank-alpha/models/cmf-bank-alpha-20260518",
        hyperparameters={"epochs": 3, "batch_size": 8, "learning_rate": 1e-5},
        objective_profile="zero_silent_unsafe_pass",
    )


def _metrics(
    *,
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
        recall=0.96,
        f1=f1,
        false_positive_rate=0.03,
        false_negative_rate=false_negative_rate,
        high_risk_false_negative_rate=high_risk_false_negative_rate,
        abstention_rate=abstention_rate,
        escalation_rate=escalation_rate,
        latency_p95_ms=latency_p95_ms,
        severity_counts={"critical": 40, "high": 80, "medium": 80, "low": 40},
    )


def test_benchmark_result_is_ready_when_training_manifest_and_metrics_are_ready():
    result = CustomerBenchmarkResult.from_metrics(
        benchmark_id="bank-alpha-private-v1",
        training_manifest=_training_manifest(),
        model_artifact_uri="gs://customer-artifacts/bank-alpha/models/cmf-bank-alpha-20260518",
        metrics=_metrics(),
        raw_result_uri="gs://customer-artifacts/bank-alpha/benchmarks/private-v1.json",
        claim_boundary="Bank Alpha private guardrail validation only.",
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
        benchmark_id="bank-alpha-private-v1",
        training_manifest=invalid_training,
        model_artifact_uri="gs://customer-artifacts/bank-alpha/models/invalid",
        metrics=_metrics(),
        raw_result_uri="gs://customer-artifacts/bank-alpha/benchmarks/private-v1.json",
        claim_boundary="Bank Alpha private guardrail validation only.",
    )

    assert result.ready is False
    assert any(
        finding["code"] == "training_manifest_not_ready" for finding in result.findings
    )


def test_zero_silent_unsafe_pass_rejects_high_risk_false_negatives():
    bad_result = CustomerBenchmarkResult.from_metrics(
        benchmark_id="bank-alpha-private-v1",
        training_manifest=_training_manifest(),
        model_artifact_uri="gs://customer-artifacts/bank-alpha/models/cmf-bank-alpha-20260518",
        metrics=_metrics(high_risk_false_negative_rate=0.01),
        raw_result_uri="gs://customer-artifacts/bank-alpha/benchmarks/private-v1.json",
        claim_boundary="Bank Alpha private guardrail validation only.",
    )

    selection = select_customer_model(
        selection_id="bank-alpha-selection-20260518",
        objective_profile="zero_silent_unsafe_pass",
        candidates=[bad_result],
    )

    assert selection.ready is False
    assert selection.selected_benchmark_id == ""
    assert any(
        finding["code"] == "no_candidate_satisfies_objective"
        for finding in selection.findings
    )


def test_model_selection_prefers_highest_objective_score_among_ready_candidates():
    slower = CustomerBenchmarkResult.from_metrics(
        benchmark_id="candidate-slower",
        training_manifest=_training_manifest(),
        model_artifact_uri="gs://customer-artifacts/bank-alpha/models/slower",
        metrics=_metrics(f1=0.94, latency_p95_ms=120.0),
        raw_result_uri="gs://customer-artifacts/bank-alpha/benchmarks/slower.json",
        claim_boundary="Bank Alpha private guardrail validation only.",
    )
    faster = CustomerBenchmarkResult.from_metrics(
        benchmark_id="candidate-faster",
        training_manifest=_training_manifest(),
        model_artifact_uri="gs://customer-artifacts/bank-alpha/models/faster",
        metrics=_metrics(f1=0.91, latency_p95_ms=25.0),
        raw_result_uri="gs://customer-artifacts/bank-alpha/benchmarks/faster.json",
        claim_boundary="Bank Alpha private guardrail validation only.",
    )

    selection = select_customer_model(
        selection_id="bank-alpha-selection-20260518",
        objective_profile="low_latency",
        candidates=[slower, faster],
    )

    assert selection.ready is True
    assert selection.selected_benchmark_id == "candidate-faster"
    assert (
        selection.selected_model_artifact_uri
        == "gs://customer-artifacts/bank-alpha/models/faster"
    )
    assert len(selection.selection_hash) == 64


def test_selection_report_serialises_and_writes_stable_json(tmp_path: Path):
    result = CustomerBenchmarkResult.from_metrics(
        benchmark_id="candidate-default",
        training_manifest=_training_manifest(),
        model_artifact_uri="gs://customer-artifacts/bank-alpha/models/default",
        metrics=_metrics(),
        raw_result_uri="gs://customer-artifacts/bank-alpha/benchmarks/default.json",
        claim_boundary="Bank Alpha private guardrail validation only.",
    )
    selection = select_customer_model(
        selection_id="bank-alpha-selection-20260518",
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
