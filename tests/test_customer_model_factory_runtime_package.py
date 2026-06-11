# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory runtime package tests

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
from director_ai.core.customer_model_factory.evidence_pack import (
    build_customer_evidence_pack,
)
from director_ai.core.customer_model_factory.runtime_package import (
    CustomerRuntimePackage,
    build_customer_runtime_package,
)
from director_ai.core.customer_model_factory.sector_extension import (
    build_sector_evidence_mapping,
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
        regulation_mappings=("SOC2", "ISO27001", "ISO42001", "EU_AI_ACT", "FINMA"),
    )


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


def _deployment():
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
    selection = select_customer_model(
        selection_id="customer-alpha-selection-20260518",
        objective_profile="zero_silent_unsafe_pass",
        candidates=[benchmark],
    )
    return build_deployment_manifest(
        deployment_id="customer-alpha-prod-20260518",
        selection_report=selection,
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


def _evidence_pack():
    deployment = _deployment()
    mapping = build_sector_evidence_mapping(
        sector_id="regulated-sector",
        jurisdiction="CH",
        evidence_pack_uri="gs://customer-artifacts/customer-alpha/evidence/pack-20260518",
    )
    return build_customer_evidence_pack(
        package_id="evidence-customer-alpha-20260518",
        deployment_manifest=deployment,
        regulation_mapping=mapping,
        classification="restricted",
        export_uri="gs://customer-artifacts/customer-alpha/evidence/pack-20260518",
    )


def test_runtime_package_ready_for_private_offline_customer_mode():
    package = build_customer_runtime_package(
        runtime_id="runtime-customer-alpha-20260518",
        deployment_manifest=_deployment(),
        evidence_pack=_evidence_pack(),
        runtime_mode="offline_private",
    )

    assert package.ready is True
    assert package.findings == ()
    assert package.customer_id == "customer-alpha"
    assert package.runtime_config["tenant_id"] == "customer-alpha-tenant"
    assert package.runtime_config["selected_model_artifact_uri"].endswith(
        "cmf-customer-alpha-20260518"
    )
    assert package.runtime_config["telemetry_mode"] == "customer_controlled"
    assert package.runtime_config["external_callbacks_allowed"] is False
    assert len(package.runtime_hash) == 64


def test_runtime_package_blocks_not_ready_evidence_pack():
    bad_evidence = build_customer_evidence_pack(
        package_id="bad-evidence",
        deployment_manifest=_deployment(),
        regulation_mapping=build_sector_evidence_mapping(
            sector_id="regulated-sector",
            jurisdiction="CH",
            evidence_pack_uri="gs://customer-artifacts/customer-alpha/evidence/other",
        ),
        classification="restricted",
        export_uri="gs://customer-artifacts/customer-alpha/evidence/pack-20260518",
    )

    package = build_customer_runtime_package(
        runtime_id="runtime-customer-alpha-bad",
        deployment_manifest=_deployment(),
        evidence_pack=bad_evidence,
        runtime_mode="offline_private",
    )

    assert package.ready is False
    assert any(
        finding["code"] == "evidence_pack_not_ready" for finding in package.findings
    )


def test_runtime_package_blocks_mismatched_customer_boundaries():
    payload = _evidence_pack().to_dict()
    payload["tenant_id"] = "wrong-tenant"
    bad_evidence = CustomerRuntimePackage.evidence_pack_from_dict(payload)

    package = build_customer_runtime_package(
        runtime_id="runtime-customer-alpha-mismatch",
        deployment_manifest=_deployment(),
        evidence_pack=bad_evidence,
        runtime_mode="offline_private",
    )

    assert package.ready is False
    assert any(
        finding["code"] == "tenant_boundary_mismatch" for finding in package.findings
    )


def test_runtime_package_blocks_missing_identity_unknown_mode_and_all_boundaries():
    deployment = _deployment()
    evidence_payload = _evidence_pack().to_dict()
    evidence_payload.update(
        {
            "customer_id": "wrong-customer",
            "workspace_id": "wrong-workspace",
            "tenant_id": "wrong-tenant",
            "deployment_id": "wrong-deployment",
        }
    )

    package = build_customer_runtime_package(
        runtime_id=" ",
        deployment_manifest=deployment,
        evidence_pack=CustomerRuntimePackage.evidence_pack_from_dict(evidence_payload),
        runtime_mode="unsupported_mode",
    )

    assert package.ready is False
    assert {finding["code"] for finding in package.findings} >= {
        "runtime_id_missing",
        "runtime_mode_unknown",
        "customer_boundary_mismatch",
        "workspace_boundary_mismatch",
        "tenant_boundary_mismatch",
        "deployment_boundary_mismatch",
    }


def test_runtime_package_serialises_and_round_trips(tmp_path: Path):
    package = build_customer_runtime_package(
        runtime_id="runtime-customer-alpha-20260518",
        deployment_manifest=_deployment(),
        evidence_pack=_evidence_pack(),
        runtime_mode="offline_private",
    )

    output = package.write_json(tmp_path / "runtime_package.json")
    restored = CustomerRuntimePackage.from_dict(
        json.loads(output.read_text(encoding="utf-8"))
    )

    assert restored == package
    assert restored.to_dict() == package.to_dict()


def test_runtime_package_schema_is_machine_readable():
    schema_path = (
        ROOT / "schemas" / "customer-model-factory-runtime-package.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == "DIRECTOR-AI Customer Model Factory Runtime Package"
    assert set(schema["required"]) >= {
        "runtime_id",
        "deployment_id",
        "evidence_hash",
        "runtime_mode",
        "runtime_config",
        "runtime_hash",
    }
