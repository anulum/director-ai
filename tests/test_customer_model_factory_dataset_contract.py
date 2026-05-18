# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory dataset contract tests

from __future__ import annotations

import json
from pathlib import Path

from director_ai.core.customer_model_factory.dataset_contract import (
    CustomerWorkspace,
    validate_customer_trace_dataset,
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


def _row(trace_id: str, split: str, *, severity: str = "medium") -> dict:
    return {
        "trace_id": trace_id,
        "customer_id": "bank-alpha",
        "tenant_id": "bank-alpha-tenant",
        "split": split,
        "prompt": f"What should the advisor say for case {trace_id}?",
        "response": f"Escalate regulated case {trace_id} to compliance.",
        "expected_decision": "escalate",
        "severity": severity,
        "label": "grounded",
        "source_refs": [f"policy://bank-alpha/{trace_id}"],
        "policy_refs": ["policy://bank-alpha/advice-boundary"],
        "reviewer_role": "compliance_reviewer",
        "observed_at": "2026-05-18T12:00:00Z",
        "contains_pii": False,
        "contains_secret": False,
        "redaction_status": "not_required",
        "vertical": "banking",
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


def test_valid_customer_dataset_produces_ready_report():
    rows = [
        _row("trace-001", "train"),
        _row("trace-002", "eval", severity="high"),
        _row("trace-003", "test", severity="critical"),
    ]

    report = validate_customer_trace_dataset(
        rows, _workspace(), vertical_profile="banking"
    )

    assert report.ready is True
    assert report.findings == ()
    assert report.row_count == 3
    assert report.split_counts == {"eval": 1, "test": 1, "train": 1}
    assert report.high_risk_count == 2
    assert len(report.dataset_hash) == 64
    assert report.customer_id == "bank-alpha"


def test_dataset_missing_required_field_blocks_readiness():
    row = _row("trace-001", "train")
    row.pop("source_refs")

    report = validate_customer_trace_dataset(
        [row], _workspace(), vertical_profile="banking"
    )

    assert report.ready is False
    assert any(
        finding.code == "missing_required_field"
        and finding.field == "source_refs"
        and finding.severity == "error"
        for finding in report.findings
    )


def test_dataset_rejects_mixed_customer_or_tenant_rows():
    bad_customer = _row("trace-001", "train")
    bad_customer["customer_id"] = "other-bank"
    bad_tenant = _row("trace-002", "eval")
    bad_tenant["tenant_id"] = "other-tenant"

    report = validate_customer_trace_dataset(
        [bad_customer, bad_tenant],
        _workspace(),
        vertical_profile="banking",
    )

    assert report.ready is False
    assert {finding.code for finding in report.findings} >= {
        "customer_id_mismatch",
        "tenant_id_mismatch",
    }


def test_dataset_blocks_exact_cross_split_leakage():
    train = _row("trace-001", "train")
    eval_row = _row("trace-002", "eval")
    eval_row["prompt"] = train["prompt"]
    eval_row["response"] = train["response"]

    report = validate_customer_trace_dataset(
        [train, eval_row],
        _workspace(),
        vertical_profile="banking",
    )

    assert report.ready is False
    assert any(finding.code == "cross_split_duplicate" for finding in report.findings)


def test_banking_profile_requires_banking_metadata_for_high_risk_rows():
    high_risk = _row("trace-001", "eval", severity="critical")
    high_risk["metadata"] = {"business_line": "retail_banking"}

    report = validate_customer_trace_dataset(
        [high_risk],
        _workspace(),
        vertical_profile="banking",
    )

    assert report.ready is False
    assert {finding.code for finding in report.findings} >= {
        "banking_metadata_missing",
        "split_missing",
    }


def test_unredacted_secret_blocks_readiness():
    row = _row("trace-001", "train")
    row["contains_secret"] = True
    row["redaction_status"] = "not_required"

    report = validate_customer_trace_dataset([row], _workspace())

    assert report.ready is False
    assert any(finding.code == "unredacted_secret" for finding in report.findings)


def test_report_serialises_to_stable_manifest_shape():
    report = validate_customer_trace_dataset(
        [_row("trace-001", "train"), _row("trace-002", "eval")],
        _workspace(),
    )

    payload = report.to_dict()

    assert payload["schema_version"] == "1.0.0"
    assert payload["workspace_id"] == "bank-alpha-prod"
    assert payload["split_counts"] == {"eval": 1, "train": 1}
    assert "findings" in payload
    assert "dataset_hash" in payload


def test_report_writer_persists_json_manifest(tmp_path: Path):
    report = validate_customer_trace_dataset(
        [_row("trace-001", "train"), _row("trace-002", "eval")],
        _workspace(),
    )

    output = report.write_json(tmp_path / "validation_report.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output.name == "validation_report.json"
    assert payload == report.to_dict()
    assert payload["dataset_hash"] == report.dataset_hash


def test_customer_trace_schema_is_machine_readable():
    schema_path = ROOT / "schemas" / "customer-model-factory-trace.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == "DIRECTOR-AI Customer Model Factory Trace"
    assert set(schema["required"]) >= {
        "trace_id",
        "customer_id",
        "tenant_id",
        "split",
        "prompt",
        "response",
        "expected_decision",
        "severity",
        "source_refs",
        "policy_refs",
    }
    assert schema["properties"]["expected_decision"]["enum"] == [
        "approve",
        "block",
        "abstain",
        "escalate",
    ]
