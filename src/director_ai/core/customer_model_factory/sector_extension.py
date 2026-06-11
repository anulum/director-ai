# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory sector extension

"""Generic sector-extension controls for private customer packages.

The public core validates the extension contract without publishing
sector-specific taxonomies, database-class mappings, or tuning recipes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from .dataset_contract import CustomerTraceFinding

SECTOR_REQUIRED_METADATA = frozenset(
    {"sector_class", "knowledge_class", "requires_citation", "jurisdiction"}
)

DEFAULT_SECTOR_FRAMEWORKS = ("ISO27001", "ISO42001", "SOC2")

DEFAULT_CONTROL_EVIDENCE = {
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
        "knowledge_class",
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
        "sector_class",
        "jurisdiction",
        "knowledge_class",
    ),
}


@dataclass(frozen=True)
class SectorEvidenceMapping:
    """Evidence mapping for customer-specific sector extension packages."""

    sector_id: str
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
    def from_dict(cls, payload: dict[str, Any]) -> SectorEvidenceMapping:
        """Rebuild a sector evidence mapping from JSON-safe data."""

        control_evidence = {
            control: tuple(fields)
            for control, fields in payload["control_evidence"].items()
        }
        return cls(
            sector_id=payload["sector_id"],
            jurisdiction=payload["jurisdiction"],
            evidence_pack_uri=payload["evidence_pack_uri"],
            frameworks=tuple(payload["frameworks"]),
            control_evidence=control_evidence,
            mapping_hash=payload["mapping_hash"],
        )


def validate_sector_trace_metadata(
    metadata: dict[str, Any],
    *,
    trace_id: str,
    expected_decision: str,
) -> tuple[CustomerTraceFinding, ...]:
    """Validate generic sector-extension metadata and evidence controls."""

    findings: list[CustomerTraceFinding] = []
    for field in sorted(SECTOR_REQUIRED_METADATA):
        if not _metadata_present(metadata.get(field)):
            findings.append(
                CustomerTraceFinding(
                    code="sector_metadata_missing",
                    severity="error",
                    message=f"sector extension requires metadata.{field}",
                    trace_id=trace_id,
                    field=f"metadata.{field}",
                )
            )

    _validate_citation_controls(metadata, trace_id, findings)
    _validate_numeric_controls(metadata, trace_id, findings)
    _validate_escalation_controls(metadata, trace_id, expected_decision, findings)
    return tuple(findings)


def build_sector_evidence_mapping(
    *,
    sector_id: str,
    jurisdiction: str,
    evidence_pack_uri: str,
    frameworks: tuple[str, ...] = DEFAULT_SECTOR_FRAMEWORKS,
    control_evidence: dict[str, tuple[str, ...]] | None = None,
) -> SectorEvidenceMapping:
    """Build a deterministic evidence mapping for a private sector extension."""

    sorted_frameworks = tuple(sorted(set(frameworks)))
    controls = (
        control_evidence if control_evidence is not None else DEFAULT_CONTROL_EVIDENCE
    )
    sorted_controls = {
        control: tuple(fields) for control, fields in sorted(controls.items())
    }
    payload = {
        "sector_id": sector_id,
        "jurisdiction": jurisdiction,
        "evidence_pack_uri": evidence_pack_uri,
        "frameworks": sorted_frameworks,
        "control_evidence": sorted_controls,
    }
    mapping_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SectorEvidenceMapping(
        sector_id=sector_id,
        jurisdiction=jurisdiction,
        evidence_pack_uri=evidence_pack_uri,
        frameworks=sorted_frameworks,
        control_evidence=sorted_controls,
        mapping_hash=mapping_hash,
    )


def _validate_citation_controls(
    metadata: dict[str, Any],
    trace_id: str,
    findings: list[CustomerTraceFinding],
) -> None:
    if metadata.get("requires_citation") is not True:
        return
    if not _non_empty_string_list(metadata.get("evidence_refs")):
        findings.append(
            CustomerTraceFinding(
                code="sector_citation_required",
                severity="error",
                message="sector extension requires citation evidence",
                trace_id=trace_id,
                field="metadata.evidence_refs",
            )
        )


def _validate_numeric_controls(
    metadata: dict[str, Any],
    trace_id: str,
    findings: list[CustomerTraceFinding],
) -> None:
    if metadata.get("requires_numeric_evidence") is not True:
        return
    if not _non_empty_string_list(metadata.get("numeric_evidence_refs")):
        findings.append(
            CustomerTraceFinding(
                code="sector_numeric_evidence_missing",
                severity="error",
                message="sector extension requires numeric evidence references",
                trace_id=trace_id,
                field="metadata.numeric_evidence_refs",
            )
        )


def _validate_escalation_controls(
    metadata: dict[str, Any],
    trace_id: str,
    expected_decision: str,
    findings: list[CustomerTraceFinding],
) -> None:
    if metadata.get("requires_escalation") is not True:
        return
    if expected_decision != "escalate":
        findings.append(
            CustomerTraceFinding(
                code="sector_escalation_required",
                severity="error",
                message="sector extension requires human escalation",
                trace_id=trace_id,
                field="expected_decision",
            )
        )


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _metadata_present(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None


def _non_empty_string_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item.strip() for item in value)
    )
