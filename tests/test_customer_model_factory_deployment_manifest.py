# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory deployment manifest tests

from __future__ import annotations

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


def _selection():
    dataset_report = validate_customer_trace_dataset(
        [
            _row("trace-001", "train"),
            _row("trace-002", "eval"),
            _row("trace-003", "test"),
        ],
        _workspace(),
        vertical_profile="regulated-sector",
    )
    training = build_training_manifest(
        package_id="cmf-customer-alpha-20260518",
        dataset_report=dataset_report,
        lane=TrainingLane.VERTEX,
        base_model_id="microsoft/deberta-v3-small",
        base_model_revision="abcdef1234567890abcdef1234567890abcdef12",
        output_uri="gs://customer-artifacts/customer-alpha/models/cmf-customer-alpha-20260518",
        hyperparameters={"epochs": 3, "batch_size": 8},
        objective_profile="zero_silent_unsafe_pass",
    )
    benchmark = CustomerBenchmarkResult.from_metrics(
        benchmark_id="customer-alpha-private-v1",
        training_manifest=training,
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
            severity_counts={"critical": 40, "high": 80, "medium": 80, "low": 40},
        ),
        raw_result_uri="gs://customer-artifacts/customer-alpha/benchmarks/private-v1.json",
        claim_boundary="Customer Alpha private guardrail validation only.",
    )
    return select_customer_model(
        selection_id="customer-alpha-selection-20260518",
        objective_profile="zero_silent_unsafe_pass",
        candidates=[benchmark],
    )


def _policy() -> DeploymentPolicy:
    return DeploymentPolicy(
        threshold=0.72,
        abstention_threshold=0.58,
        escalation_threshold=0.40,
        require_citations=True,
        audit_log_uri="gs://customer-artifacts/customer-alpha/audit/decision-log.jsonl",
        evidence_pack_uri="gs://customer-artifacts/customer-alpha/evidence/pack-20260518",
        rollback_package_uri="gs://customer-artifacts/customer-alpha/deployments/previous.json",
        retention_days=365,
        telemetry_mode="customer_controlled",
    )


def test_deployment_manifest_ready_for_ready_selection_and_complete_policy():
    manifest = build_deployment_manifest(
        deployment_id="customer-alpha-prod-20260518",
        selection_report=_selection(),
        policy=_policy(),
        environment="production",
        package_uri="gs://customer-artifacts/customer-alpha/deployments/customer-alpha-prod-20260518.json",
    )

    assert manifest.ready is True
    assert manifest.findings == ()
    assert manifest.customer_id == "customer-alpha"
    assert manifest.tenant_id == "customer-alpha-tenant"
    assert manifest.selected_model_artifact_uri.endswith("cmf-customer-alpha-20260518")
    assert manifest.policy.require_citations is True
    assert len(manifest.deployment_hash) == 64


def test_deployment_manifest_blocks_not_ready_selection():
    selection = select_customer_model(
        selection_id="empty-selection",
        objective_profile="balanced",
        candidates=[],
    )

    manifest = build_deployment_manifest(
        deployment_id="bad-deployment",
        selection_report=selection,
        policy=_policy(),
        environment="production",
        package_uri="gs://customer-artifacts/customer-alpha/deployments/bad.json",
    )

    assert manifest.ready is False
    assert any(
        finding["code"] == "selection_not_ready" for finding in manifest.findings
    )


def test_deployment_manifest_blocks_unsafe_threshold_ordering():
    bad_policy = DeploymentPolicy(
        threshold=0.50,
        abstention_threshold=0.60,
        escalation_threshold=0.70,
        require_citations=True,
        audit_log_uri="gs://customer-artifacts/customer-alpha/audit/decision-log.jsonl",
        evidence_pack_uri="gs://customer-artifacts/customer-alpha/evidence/pack-20260518",
        rollback_package_uri="gs://customer-artifacts/customer-alpha/deployments/previous.json",
        retention_days=365,
        telemetry_mode="customer_controlled",
    )

    manifest = build_deployment_manifest(
        deployment_id="bad-thresholds",
        selection_report=_selection(),
        policy=bad_policy,
        environment="production",
        package_uri="gs://customer-artifacts/customer-alpha/deployments/bad-thresholds.json",
    )

    assert manifest.ready is False
    assert any(
        finding["code"] == "threshold_order_invalid" for finding in manifest.findings
    )


def test_deployment_manifest_requires_audit_evidence_and_rollback_for_production():
    bad_policy = DeploymentPolicy(
        threshold=0.72,
        abstention_threshold=0.58,
        escalation_threshold=0.40,
        require_citations=True,
        audit_log_uri="",
        evidence_pack_uri="",
        rollback_package_uri="",
        retention_days=365,
        telemetry_mode="customer_controlled",
    )

    manifest = build_deployment_manifest(
        deployment_id="missing-evidence",
        selection_report=_selection(),
        policy=bad_policy,
        environment="production",
        package_uri="gs://customer-artifacts/customer-alpha/deployments/missing-evidence.json",
    )

    assert manifest.ready is False
    assert {finding["code"] for finding in manifest.findings} >= {
        "audit_log_missing",
        "evidence_pack_missing",
        "rollback_missing",
    }


def test_deployment_manifest_serialises_and_writes_stable_json(tmp_path: Path):
    manifest = build_deployment_manifest(
        deployment_id="customer-alpha-prod-20260518",
        selection_report=_selection(),
        policy=_policy(),
        environment="production",
        package_uri="gs://customer-artifacts/customer-alpha/deployments/customer-alpha-prod-20260518.json",
    )

    output = manifest.write_json(tmp_path / "deployment.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload == manifest.to_dict()
    assert payload["schema_version"] == "1.0.0"
    assert payload["policy"]["telemetry_mode"] == "customer_controlled"


def test_deployment_manifest_schema_is_machine_readable():
    schema_path = ROOT / "schemas" / "customer-model-factory-deployment.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == "DIRECTOR-AI Customer Model Factory Deployment Manifest"
    assert set(schema["required"]) >= {
        "deployment_id",
        "selection_hash",
        "selected_model_artifact_uri",
        "policy",
        "environment",
        "package_uri",
        "deployment_hash",
    }
