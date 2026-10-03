# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory sector extension tests

"""Exercise generic sector controls and their dataset-validation integration."""

from __future__ import annotations

import json
from pathlib import Path

import director_ai.core.customer_model_factory as cmf
from director_ai.core.customer_model_factory.dataset_contract import (
    CustomerWorkspace,
    validate_customer_trace_dataset,
)
from director_ai.core.customer_model_factory.sector_extension import (
    _string,
    build_sector_evidence_mapping,
    validate_sector_trace_metadata,
)

ROOT = Path(__file__).resolve().parents[1]


def _workspace() -> CustomerWorkspace:
    """Build the customer/tenant workspace shared by sector-validation cases.

    Returns
    -------
    CustomerWorkspace
        Customer/tenant workspace fixture.
    """
    return CustomerWorkspace(
        customer_id="customer-alpha",
        workspace_id="customer-alpha-prod",
        tenant_id="customer-alpha-tenant",
        data_classification="confidential",
        allowed_splits=("train", "eval", "test"),
        regulation_mappings=("SOC2", "ISO27001", "ISO42001"),
    )


def _metadata(**overrides: object) -> dict[str, object]:
    """Build sector metadata with caller-specified negative or positive controls.

    Parameters
    ----------
    overrides : object
        Explicit field values substituted into the synthetic fixture.

    Returns
    -------
    dict[str, object]
        Metadata dictionary with requested substitutions.
    """
    metadata: dict[str, object] = {
        "sector_class": "regulated_knowledge",
        "knowledge_class": "customer_policy",
        "jurisdiction": "customer-controlled",
        "requires_citation": True,
        "evidence_refs": ["policy://customer-alpha/advice-boundary"],
        "requires_numeric_evidence": False,
        "numeric_evidence_refs": [],
        "requires_escalation": True,
    }
    metadata.update(overrides)
    return metadata


def _row(trace_id: str, split: str, metadata: dict[str, object]) -> dict[str, object]:
    """Build a trace row carrying the given split and sector metadata.

    Parameters
    ----------
    trace_id : str
        Unique identifier for the synthetic trace.
    split : str
        Dataset split assigned to the synthetic trace.
    metadata : dict[str, object]
        Sector controls and evidence references attached to the trace.

    Returns
    -------
    dict[str, object]
        Synthetic trace data.
    """
    return {
        "trace_id": trace_id,
        "customer_id": "customer-alpha",
        "tenant_id": "customer-alpha-tenant",
        "split": split,
        "prompt": f"Assess customer-controlled communication case {trace_id}.",
        "response": f"Escalate regulated case {trace_id} to reviewer.",
        "expected_decision": "escalate",
        "severity": "high",
        "label": "regulated_boundary",
        "source_refs": [f"policy://customer-alpha/{trace_id}"],
        "policy_refs": ["policy://customer-alpha/advice-boundary"],
        "reviewer_role": "sector_reviewer",
        "observed_at": "2026-05-20T12:00:00Z",
        "contains_pii": False,
        "contains_secret": False,
        "redaction_status": "not_required",
        "vertical": "regulated-sector",
        "metadata": metadata,
    }


def test_public_customer_model_factory_exports_generic_sector_extension_only() -> None:
    """Expose generic sector contracts without public bank-specific symbols."""
    assert hasattr(cmf, "SectorEvidenceMapping")
    assert hasattr(cmf, "build_sector_evidence_mapping")
    assert hasattr(cmf, "validate_sector_trace_metadata")

    assert all("bank" not in symbol.casefold() for symbol in cmf.__all__)


def test_sector_metadata_validation_is_generic_and_evidence_oriented() -> None:
    """Require citation references and escalation for their enabled controls."""
    findings = validate_sector_trace_metadata(
        _metadata(evidence_refs=[]),
        trace_id="trace-001",
        expected_decision="approve",
    )

    assert {finding.code for finding in findings} >= {
        "sector_citation_required",
        "sector_escalation_required",
    }


def test_sector_metadata_accepts_complete_citation_numeric_and_escalation_controls() -> (
    None
):
    """Accept supplied citation and numeric references with an escalation label."""
    findings = validate_sector_trace_metadata(
        _metadata(
            requires_numeric_evidence=True,
            numeric_evidence_refs=["metric://customer-alpha/error-rate"],
        ),
        trace_id="trace-001",
        expected_decision="escalate",
    )

    assert findings == ()


def test_sector_metadata_allows_non_escalation_rows_without_escalation_control() -> (
    None
):
    """Accept approval metadata when the escalation control is disabled."""
    findings = validate_sector_trace_metadata(
        _metadata(requires_escalation=False),
        trace_id="trace-001",
        expected_decision="approve",
    )

    assert findings == ()


def test_sector_string_helper_normalises_non_string_values() -> None:
    """Strip strings while normalizing non-string metadata values to empty text."""
    assert _string("  regulated  ") == "regulated"
    assert _string(None) == ""
    assert _string(123) == ""


def test_sector_metadata_reports_each_missing_required_field() -> None:
    """Report every blank or absent required sector metadata field."""
    findings = validate_sector_trace_metadata(
        _metadata(
            sector_class=" ",
            knowledge_class=None,
            jurisdiction="",
            requires_citation=None,
        ),
        trace_id="trace-001",
        expected_decision="escalate",
    )

    assert {
        finding.field
        for finding in findings
        if finding.code == "sector_metadata_missing"
    } == {
        "metadata.jurisdiction",
        "metadata.knowledge_class",
        "metadata.requires_citation",
        "metadata.sector_class",
    }


def test_sector_extension_findings_are_wired_into_dataset_validation() -> None:
    """Propagate missing numeric evidence into public dataset-validation findings."""
    rows = [
        _row("trace-001", "train", _metadata()),
        _row(
            "trace-002",
            "eval",
            _metadata(requires_numeric_evidence=True, numeric_evidence_refs=[]),
        ),
        _row("trace-003", "test", _metadata(knowledge_class="incident_policy")),
    ]

    report = validate_customer_trace_dataset(
        rows,
        _workspace(),
        vertical_profile="regulated-sector",
    )

    assert report.ready is False
    assert any(
        finding.code == "sector_numeric_evidence_missing"
        and finding.trace_id == "trace-002"
        for finding in report.findings
    )


def test_sector_evidence_mapping_is_machine_readable() -> None:
    """Serialize framework and control mappings with a deterministic hash."""
    mapping = build_sector_evidence_mapping(
        sector_id="regulated-sector",
        jurisdiction="customer-controlled",
        evidence_pack_uri="gs://customer-artifacts/customer-alpha/evidence/pack.json",
        frameworks=("SOC2", "ISO27001"),
    )
    payload = mapping.to_dict()

    assert payload["sector_id"] == "regulated-sector"
    assert payload["jurisdiction"] == "customer-controlled"
    assert payload["evidence_pack_uri"].endswith("/pack.json")
    assert set(payload["frameworks"]) == {"ISO27001", "SOC2"}
    assert "dataset_lineage" in payload["control_evidence"]
    assert "human_escalation" in payload["control_evidence"]
    assert len(payload["mapping_hash"]) == 64


def test_sector_evidence_mapping_round_trips_json_safe_payload() -> None:
    """Sort/deduplicate frameworks and retain custom controls through a round-trip."""
    mapping = build_sector_evidence_mapping(
        sector_id="regulated-sector",
        jurisdiction="customer-controlled",
        evidence_pack_uri="gs://customer/evidence.json",
        frameworks=("SOC2", "ISO27001", "SOC2"),
        control_evidence={"z_control": ("field_b",), "a_control": ("field_a",)},
    )
    payload = mapping.to_dict()

    rebuilt = type(mapping).from_dict(payload)

    assert payload["frameworks"] == ["ISO27001", "SOC2"]
    assert list(payload["control_evidence"]) == ["a_control", "z_control"]
    assert rebuilt == mapping


def test_sector_metadata_schema_replaces_public_vertical_schema() -> None:
    """Pin the generic sector schema dialect, title and required fields."""
    schema_path = (
        ROOT / "schemas" / "customer-model-factory-sector-metadata.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == "DIRECTOR-AI Customer Model Factory Sector Metadata"
    assert set(schema["required"]) >= {
        "sector_class",
        "knowledge_class",
        "jurisdiction",
        "requires_citation",
        "evidence_refs",
        "requires_escalation",
    }
