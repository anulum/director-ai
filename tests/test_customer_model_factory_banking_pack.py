# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory banking pack tests

from __future__ import annotations

import json
from pathlib import Path

from director_ai.core.customer_model_factory.banking_pack import (
    BANKING_BUSINESS_LINES,
    BANKING_REGULATED_CATEGORIES,
    build_banking_regulation_mapping,
    validate_banking_trace_metadata,
)
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
        regulation_mappings=("SOC2", "ISO27001", "ISO42001", "EU_AI_ACT", "FINMA"),
    )


def _row(trace_id: str, split: str, metadata: dict[str, object]) -> dict[str, object]:
    return {
        "trace_id": trace_id,
        "customer_id": "bank-alpha",
        "tenant_id": "bank-alpha-tenant",
        "split": split,
        "prompt": f"Assess customer communication case {trace_id}.",
        "response": f"Escalate regulated banking case {trace_id} to compliance.",
        "expected_decision": "escalate",
        "severity": "high",
        "label": "regulated_boundary",
        "source_refs": [f"policy://bank-alpha/{trace_id}"],
        "policy_refs": ["policy://bank-alpha/banking-advice-boundary"],
        "reviewer_role": "banking_compliance_reviewer",
        "observed_at": "2026-05-18T12:00:00Z",
        "contains_pii": False,
        "contains_secret": False,
        "redaction_status": "not_required",
        "vertical": "banking",
        "metadata": metadata,
    }


def _valid_metadata(**overrides: object) -> dict[str, object]:
    metadata: dict[str, object] = {
        "business_line": "retail_banking",
        "regulated_category": "financial_advice_boundary",
        "jurisdiction": "CH",
        "requires_citation": True,
        "evidence_refs": ["policy://bank-alpha/advice-boundary"],
        "numeric_evidence_refs": [],
        "requires_escalation": True,
        "customer_segment": "retail",
        "product_family": "mortgage",
    }
    metadata.update(overrides)
    return metadata


def test_banking_taxonomy_contains_first_vertical_controls():
    assert {
        "retail_banking",
        "private_banking",
        "corporate_banking",
        "wealth_management",
        "risk_compliance",
    } <= BANKING_BUSINESS_LINES
    assert {
        "financial_advice_boundary",
        "fees_rates_terms",
        "kyc_aml",
        "investment_suitability",
    } <= BANKING_REGULATED_CATEGORIES


def test_valid_banking_metadata_has_no_blocking_findings():
    findings = validate_banking_trace_metadata(
        _valid_metadata(),
        trace_id="trace-001",
        expected_decision="escalate",
    )

    assert findings == ()


def test_financial_advice_boundary_requires_citation_and_escalation():
    findings = validate_banking_trace_metadata(
        _valid_metadata(
            requires_citation=False, evidence_refs=[], requires_escalation=False
        ),
        trace_id="trace-001",
        expected_decision="approve",
    )

    assert {finding.code for finding in findings} >= {
        "banking_citation_required",
        "banking_escalation_required",
    }
    assert all(finding.trace_id == "trace-001" for finding in findings)


def test_fees_rates_terms_require_numeric_evidence_references():
    findings = validate_banking_trace_metadata(
        _valid_metadata(
            regulated_category="fees_rates_terms",
            numeric_evidence_refs=[],
            product_family="current_account",
        ),
        trace_id="trace-002",
        expected_decision="escalate",
    )

    assert any(
        finding.code == "banking_numeric_evidence_missing" for finding in findings
    )


def test_unknown_banking_taxonomy_values_block_readiness():
    findings = validate_banking_trace_metadata(
        _valid_metadata(
            business_line="unregistered_business_line",
            regulated_category="unregistered_category",
        ),
        trace_id="trace-003",
        expected_decision="escalate",
    )

    assert {finding.code for finding in findings} >= {
        "banking_business_line_unknown",
        "banking_category_unknown",
    }


def test_banking_pack_findings_are_wired_into_dataset_validation():
    rows = [
        _row("trace-001", "train", _valid_metadata()),
        _row(
            "trace-002",
            "eval",
            _valid_metadata(
                regulated_category="fees_rates_terms",
                numeric_evidence_refs=[],
                product_family="credit_card",
            ),
        ),
        _row("trace-003", "test", _valid_metadata(regulated_category="kyc_aml")),
    ]

    report = validate_customer_trace_dataset(
        rows, _workspace(), vertical_profile="banking"
    )

    assert report.ready is False
    assert any(
        finding.code == "banking_numeric_evidence_missing"
        and finding.trace_id == "trace-002"
        for finding in report.findings
    )


def test_banking_regulation_mapping_is_evidence_oriented():
    mapping = build_banking_regulation_mapping(
        jurisdiction="CH",
        evidence_pack_uri="gs://customer-evidence/bank-alpha/pack.json",
    )
    payload = mapping.to_dict()

    assert payload["jurisdiction"] == "CH"
    assert payload["evidence_pack_uri"] == "gs://customer-evidence/bank-alpha/pack.json"
    assert set(payload["frameworks"]) >= {
        "EU_AI_ACT",
        "FINMA",
        "ISO27001",
        "ISO42001",
        "SOC2",
    }
    assert "dataset_lineage" in payload["control_evidence"]
    assert "human_escalation" in payload["control_evidence"]
    assert len(payload["mapping_hash"]) == 64


def test_banking_metadata_schema_is_machine_readable():
    schema_path = (
        ROOT / "schemas" / "customer-model-factory-banking-metadata.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == "DIRECTOR-AI Customer Model Factory Banking Metadata"
    assert set(schema["required"]) >= {
        "business_line",
        "regulated_category",
        "jurisdiction",
        "requires_citation",
        "evidence_refs",
        "requires_escalation",
    }
