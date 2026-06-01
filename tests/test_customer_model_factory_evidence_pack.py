# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory evidence pack tests

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
    CustomerEvidencePackManifest,
    build_customer_evidence_pack,
)
from director_ai.core.customer_model_factory.sector_extension import (
    build_sector_evidence_mapping,
)
from director_ai.core.customer_model_factory.training_manifest import (
    TrainingLane,
    build_training_manifest,
)
from tools.export_customer_model_factory_evidence_pack import main as export_main

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


def _mapping():
    return build_sector_evidence_mapping(
        sector_id="regulated-sector",
        jurisdiction="CH",
        evidence_pack_uri="gs://customer-artifacts/customer-alpha/evidence/pack-20260518",
    )


def test_evidence_pack_is_ready_for_ready_deployment_and_mapping():
    manifest = build_customer_evidence_pack(
        package_id="evidence-customer-alpha-20260518",
        deployment_manifest=_deployment(),
        regulation_mapping=_mapping(),
        classification="restricted",
        export_uri="gs://customer-artifacts/customer-alpha/evidence/pack-20260518",
    )

    assert manifest.ready is True
    assert manifest.findings == ()
    assert manifest.customer_id == "customer-alpha"
    assert manifest.tenant_id == "customer-alpha-tenant"
    assert manifest.external_callbacks_allowed is False
    assert manifest.artefacts["deployment_hash"] == _deployment().deployment_hash
    assert manifest.artefacts["regulation_mapping_hash"] == _mapping().mapping_hash
    assert len(manifest.evidence_hash) == 64


def test_evidence_pack_blocks_not_ready_deployment():
    bad_deployment = build_deployment_manifest(
        deployment_id="bad",
        selection_report=select_customer_model(
            selection_id="empty",
            objective_profile="balanced",
            candidates=[],
        ),
        policy=DeploymentPolicy(
            threshold=0.72,
            abstention_threshold=0.58,
            escalation_threshold=0.40,
            require_citations=True,
            audit_log_uri="",
            evidence_pack_uri="",
            rollback_package_uri="",
            retention_days=365,
            telemetry_mode="customer_controlled",
        ),
        environment="production",
        package_uri="",
    )

    manifest = build_customer_evidence_pack(
        package_id="evidence-customer-alpha-bad",
        deployment_manifest=bad_deployment,
        regulation_mapping=_mapping(),
        classification="restricted",
        export_uri="gs://customer-artifacts/customer-alpha/evidence/bad",
    )

    assert manifest.ready is False
    assert any(
        finding["code"] == "deployment_not_ready" for finding in manifest.findings
    )


def test_evidence_pack_blocks_unsanctioned_external_callbacks():
    manifest = build_customer_evidence_pack(
        package_id="evidence-customer-alpha-callback",
        deployment_manifest=_deployment(),
        regulation_mapping=_mapping(),
        classification="restricted",
        export_uri="gs://customer-artifacts/customer-alpha/evidence/pack-20260518",
        callback_endpoints=("https://vendor.example/callback",),
    )

    assert manifest.ready is False
    assert any(
        finding["code"] == "external_callback_not_allowed"
        for finding in manifest.findings
    )


def test_evidence_pack_reports_package_classification_and_uri_findings():
    manifest = build_customer_evidence_pack(
        package_id=" ",
        deployment_manifest=_deployment(),
        regulation_mapping=_mapping(),
        classification="public",
        export_uri=" ",
    )

    assert manifest.ready is False
    assert {finding["code"] for finding in manifest.findings} >= {
        "package_id_missing",
        "classification_unknown",
        "export_uri_missing",
        "evidence_uri_mismatch",
        "regulation_mapping_uri_mismatch",
    }


def test_evidence_pack_allows_customer_approved_callback_endpoints():
    manifest = build_customer_evidence_pack(
        package_id="evidence-customer-alpha-callback-approved",
        deployment_manifest=_deployment(),
        regulation_mapping=_mapping(),
        classification="confidential",
        export_uri="gs://customer-artifacts/customer-alpha/evidence/pack-20260518",
        external_callbacks_allowed=True,
        callback_endpoints=("https://customer.example/evidence",),
    )

    assert manifest.ready is True
    assert manifest.external_callbacks_allowed is True
    assert manifest.callback_endpoints == ("https://customer.example/evidence",)


def test_evidence_pack_serialises_and_round_trips(tmp_path: Path):
    manifest = build_customer_evidence_pack(
        package_id="evidence-customer-alpha-20260518",
        deployment_manifest=_deployment(),
        regulation_mapping=_mapping(),
        classification="restricted",
        export_uri="gs://customer-artifacts/customer-alpha/evidence/pack-20260518",
    )

    output = manifest.write_json(tmp_path / "evidence_pack.json")
    restored = CustomerEvidencePackManifest.from_dict(
        json.loads(output.read_text(encoding="utf-8"))
    )

    assert restored == manifest
    assert restored.to_dict() == manifest.to_dict()


def test_evidence_pack_cli_writes_manifest(tmp_path: Path):
    deployment_path = _deployment().write_json(tmp_path / "deployment.json")
    mapping_path = tmp_path / "sector_mapping.json"
    mapping_path.write_text(
        json.dumps(_mapping().to_dict(), sort_keys=True), encoding="utf-8"
    )
    output_path = tmp_path / "evidence_pack.json"

    exit_code = export_main(
        [
            "--deployment-manifest",
            str(deployment_path),
            "--regulation-mapping",
            str(mapping_path),
            "--package-id",
            "evidence-customer-alpha-20260518",
            "--classification",
            "restricted",
            "--export-uri",
            "gs://customer-artifacts/customer-alpha/evidence/pack-20260518",
            "--output",
            str(output_path),
        ]
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["ready"] is True
    assert payload["package_id"] == "evidence-customer-alpha-20260518"


def test_evidence_pack_schema_is_machine_readable():
    schema_path = ROOT / "schemas" / "customer-model-factory-evidence-pack.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == "DIRECTOR-AI Customer Model Factory Evidence Pack"
    assert set(schema["required"]) >= {
        "package_id",
        "deployment_id",
        "classification",
        "export_uri",
        "artefacts",
        "evidence_hash",
    }
