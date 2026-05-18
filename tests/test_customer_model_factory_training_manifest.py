# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory training manifest tests

from __future__ import annotations

import json
from pathlib import Path

from director_ai.core.customer_model_factory.dataset_contract import (
    CustomerWorkspace,
    validate_customer_trace_dataset,
)
from director_ai.core.customer_model_factory.training_manifest import (
    CustomerTrainingManifest,
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


def _ready_report():
    return validate_customer_trace_dataset(
        [
            _row("trace-001", "train"),
            _row("trace-002", "eval"),
            _row("trace-003", "test"),
        ],
        _workspace(),
        vertical_profile="banking",
    )


def test_training_manifest_is_ready_for_valid_dataset_and_immutable_base_model():
    report = _ready_report()

    manifest = build_training_manifest(
        package_id="cmf-bank-alpha-20260518",
        dataset_report=report,
        lane=TrainingLane.VERTEX,
        base_model_id="microsoft/deberta-v3-small",
        base_model_revision="abcdef1234567890abcdef1234567890abcdef12",
        output_uri="gs://customer-artifacts/bank-alpha/models/cmf-bank-alpha-20260518",
        hyperparameters={"epochs": 3, "batch_size": 8, "learning_rate": 1e-5},
        objective_profile="zero_silent_unsafe_pass",
    )

    assert manifest.ready is True
    assert manifest.findings == ()
    assert manifest.customer_id == "bank-alpha"
    assert manifest.dataset_hash == report.dataset_hash
    assert manifest.lane == TrainingLane.VERTEX
    assert manifest.requires_private_execution is True
    assert len(manifest.manifest_hash) == 64


def test_training_manifest_blocks_invalid_dataset_report():
    invalid_report = validate_customer_trace_dataset(
        [_row("trace-001", "train")],
        _workspace(),
        vertical_profile="banking",
    )

    manifest = build_training_manifest(
        package_id="cmf-bank-alpha-invalid",
        dataset_report=invalid_report,
        lane=TrainingLane.LOCAL_PILOT,
        base_model_id="microsoft/deberta-v3-small",
        base_model_revision="abcdef1234567890abcdef1234567890abcdef12",
        output_uri="file:///tmp/cmf-bank-alpha-invalid",
        hyperparameters={"epochs": 1},
        objective_profile="balanced",
    )

    assert manifest.ready is False
    assert any(finding["code"] == "dataset_not_ready" for finding in manifest.findings)


def test_training_manifest_requires_base_model_revision_or_managed_artifact():
    manifest = build_training_manifest(
        package_id="cmf-bank-alpha-unpinned",
        dataset_report=_ready_report(),
        lane=TrainingLane.VERTEX,
        base_model_id="microsoft/deberta-v3-small",
        base_model_revision="",
        output_uri="gs://customer-artifacts/bank-alpha/models/cmf-bank-alpha-unpinned",
        hyperparameters={"epochs": 3},
        objective_profile="balanced",
    )

    assert manifest.ready is False
    assert any(
        finding["code"] == "base_model_not_immutable" for finding in manifest.findings
    )


def test_training_manifest_requires_lane_compatible_output_uri():
    manifest = build_training_manifest(
        package_id="cmf-bank-alpha-bad-output",
        dataset_report=_ready_report(),
        lane=TrainingLane.VERTEX,
        base_model_id="microsoft/deberta-v3-small",
        base_model_revision="abcdef1234567890abcdef1234567890abcdef12",
        output_uri="file:///tmp/not-cloud",
        hyperparameters={"epochs": 3},
        objective_profile="balanced",
    )

    assert manifest.ready is False
    assert any(
        finding["code"] == "output_uri_incompatible" for finding in manifest.findings
    )


def test_training_manifest_serialises_and_writes_stable_json(tmp_path: Path):
    manifest = build_training_manifest(
        package_id="cmf-bank-alpha-20260518",
        dataset_report=_ready_report(),
        lane=TrainingLane.CUSTOMER_CLOUD,
        base_model_id="gs://gotm-director-ai-training/managed-training/base-model",
        base_model_revision="",
        output_uri="gs://customer-artifacts/bank-alpha/models/cmf-bank-alpha-20260518",
        hyperparameters={"epochs": 2, "batch_size": 4},
        objective_profile="high_recall",
    )

    output = manifest.write_json(tmp_path / "training_manifest.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload == manifest.to_dict()
    assert payload["schema_version"] == "1.0.0"
    assert payload["lane"] == "customer_cloud"
    assert (
        payload["base_model_artifact_uri"]
        == "gs://gotm-director-ai-training/managed-training/base-model"
    )


def test_training_manifest_dataclass_can_roundtrip_from_dict():
    manifest = build_training_manifest(
        package_id="cmf-bank-alpha-20260518",
        dataset_report=_ready_report(),
        lane=TrainingLane.ON_PREM,
        base_model_id="/models/deberta-v3-small",
        base_model_revision="local-sha256:abc123",
        output_uri="file:///customer/models/cmf-bank-alpha-20260518",
        hyperparameters={"epochs": 1},
        objective_profile="conservative",
    )

    restored = CustomerTrainingManifest.from_dict(manifest.to_dict())

    assert restored == manifest


def test_training_manifest_schema_is_machine_readable():
    schema_path = (
        ROOT / "schemas" / "customer-model-factory-training-manifest.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == "DIRECTOR-AI Customer Model Factory Training Manifest"
    assert set(schema["required"]) >= {
        "package_id",
        "customer_id",
        "workspace_id",
        "tenant_id",
        "dataset_hash",
        "lane",
        "base_model_id",
        "output_uri",
        "objective_profile",
        "manifest_hash",
    }
    assert schema["properties"]["lane"]["enum"] == [
        "vertex",
        "customer_cloud",
        "on_prem",
        "local_pilot",
    ]
