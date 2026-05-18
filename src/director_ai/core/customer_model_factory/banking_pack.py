# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory banking pack

"""Banking vertical controls layered on the Customer Model Factory core."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from .dataset_contract import CustomerTraceFinding

BANKING_BUSINESS_LINES = frozenset(
    {
        "retail_banking",
        "private_banking",
        "corporate_banking",
        "wealth_management",
        "risk_compliance",
    }
)

BANKING_REGULATED_CATEGORIES = frozenset(
    {
        "financial_advice_boundary",
        "product_disclosure",
        "fees_rates_terms",
        "kyc_aml",
        "complaints_disputes",
        "credit_risk",
        "investment_suitability",
    }
)

CITATION_REQUIRED_CATEGORIES = frozenset(
    {
        "financial_advice_boundary",
        "product_disclosure",
        "fees_rates_terms",
        "credit_risk",
        "investment_suitability",
    }
)

NUMERIC_EVIDENCE_REQUIRED_CATEGORIES = frozenset({"fees_rates_terms", "credit_risk"})

ESCALATION_REQUIRED_CATEGORIES = frozenset(
    {
        "financial_advice_boundary",
        "kyc_aml",
        "investment_suitability",
        "complaints_disputes",
    }
)

DEFAULT_BANKING_FRAMEWORKS = (
    "EU_AI_ACT",
    "FINMA",
    "ISO27001",
    "ISO42001",
    "SOC2",
)

CONTROL_EVIDENCE = {
    "audit_logging": (
        "decision_log",
        "policy_refs",
        "reviewer_role",
        "observed_at",
    ),
    "customer_data_protection": (
        "tenant_id",
        "redaction_status",
        "contains_pii",
        "contains_secret",
    ),
    "dataset_lineage": (
        "dataset_hash",
        "source_refs",
        "policy_refs",
        "regulated_category",
    ),
    "document_grounding": (
        "requires_citation",
        "evidence_refs",
        "numeric_evidence_refs",
    ),
    "human_escalation": (
        "expected_decision",
        "requires_escalation",
        "reviewer_role",
    ),
    "model_risk": (
        "severity",
        "business_line",
        "jurisdiction",
        "product_family",
    ),
}


@dataclass(frozen=True)
class BankingRegulationMapping:
    """Evidence mapping for banking customer-model review packages."""

    jurisdiction: str
    evidence_pack_uri: str
    frameworks: tuple[str, ...]
    control_evidence: dict[str, tuple[str, ...]]
    mapping_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise the mapping to a deterministic JSON-safe shape."""

        payload = asdict(self)
        payload["frameworks"] = list(self.frameworks)
        payload["control_evidence"] = {
            control: list(fields)
            for control, fields in sorted(self.control_evidence.items())
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BankingRegulationMapping:
        """Rebuild a banking regulation mapping from JSON-safe data."""

        control_evidence = {
            control: tuple(fields)
            for control, fields in payload["control_evidence"].items()
        }
        return cls(
            jurisdiction=payload["jurisdiction"],
            evidence_pack_uri=payload["evidence_pack_uri"],
            frameworks=tuple(payload["frameworks"]),
            control_evidence=control_evidence,
            mapping_hash=payload["mapping_hash"],
        )


def validate_banking_trace_metadata(
    metadata: dict[str, Any],
    *,
    trace_id: str,
    expected_decision: str,
) -> tuple[CustomerTraceFinding, ...]:
    """Validate banking-specific trace metadata and policy evidence controls."""

    findings: list[CustomerTraceFinding] = []
    business_line = _string(metadata.get("business_line"))
    regulated_category = _string(metadata.get("regulated_category"))

    if business_line not in BANKING_BUSINESS_LINES:
        findings.append(
            CustomerTraceFinding(
                code="banking_business_line_unknown",
                severity="error",
                message=f"unknown banking business_line {business_line!r}",
                trace_id=trace_id,
                field="metadata.business_line",
            )
        )
    if regulated_category not in BANKING_REGULATED_CATEGORIES:
        findings.append(
            CustomerTraceFinding(
                code="banking_category_unknown",
                severity="error",
                message=f"unknown banking regulated_category {regulated_category!r}",
                trace_id=trace_id,
                field="metadata.regulated_category",
            )
        )
        return tuple(findings)

    _validate_citation_controls(metadata, regulated_category, trace_id, findings)
    _validate_numeric_controls(metadata, regulated_category, trace_id, findings)
    _validate_escalation_controls(
        metadata,
        regulated_category,
        trace_id,
        expected_decision,
        findings,
    )
    return tuple(findings)


def build_banking_regulation_mapping(
    *,
    jurisdiction: str,
    evidence_pack_uri: str,
    frameworks: tuple[str, ...] = DEFAULT_BANKING_FRAMEWORKS,
) -> BankingRegulationMapping:
    """Build a regulation-flexible banking evidence mapping."""

    sorted_frameworks = tuple(sorted(set(frameworks)))
    sorted_controls = {
        control: tuple(fields) for control, fields in sorted(CONTROL_EVIDENCE.items())
    }
    payload = {
        "jurisdiction": jurisdiction,
        "evidence_pack_uri": evidence_pack_uri,
        "frameworks": sorted_frameworks,
        "control_evidence": sorted_controls,
    }
    mapping_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return BankingRegulationMapping(
        jurisdiction=jurisdiction,
        evidence_pack_uri=evidence_pack_uri,
        frameworks=sorted_frameworks,
        control_evidence=sorted_controls,
        mapping_hash=mapping_hash,
    )


def _validate_citation_controls(
    metadata: dict[str, Any],
    regulated_category: str,
    trace_id: str,
    findings: list[CustomerTraceFinding],
) -> None:
    if regulated_category not in CITATION_REQUIRED_CATEGORIES:
        return
    if metadata.get("requires_citation") is not True or not _non_empty_string_list(
        metadata.get("evidence_refs")
    ):
        findings.append(
            CustomerTraceFinding(
                code="banking_citation_required",
                severity="error",
                message="banking regulated category requires citation evidence",
                trace_id=trace_id,
                field="metadata.evidence_refs",
            )
        )


def _validate_numeric_controls(
    metadata: dict[str, Any],
    regulated_category: str,
    trace_id: str,
    findings: list[CustomerTraceFinding],
) -> None:
    if regulated_category not in NUMERIC_EVIDENCE_REQUIRED_CATEGORIES:
        return
    if not _non_empty_string_list(metadata.get("numeric_evidence_refs")):
        findings.append(
            CustomerTraceFinding(
                code="banking_numeric_evidence_missing",
                severity="error",
                message="banking numeric category requires numeric evidence references",
                trace_id=trace_id,
                field="metadata.numeric_evidence_refs",
            )
        )


def _validate_escalation_controls(
    metadata: dict[str, Any],
    regulated_category: str,
    trace_id: str,
    expected_decision: str,
    findings: list[CustomerTraceFinding],
) -> None:
    if regulated_category not in ESCALATION_REQUIRED_CATEGORIES:
        return
    if (
        metadata.get("requires_escalation") is not True
        or expected_decision != "escalate"
    ):
        findings.append(
            CustomerTraceFinding(
                code="banking_escalation_required",
                severity="error",
                message="banking high-risk category requires human escalation",
                trace_id=trace_id,
                field="metadata.requires_escalation",
            )
        )


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _non_empty_string_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )
