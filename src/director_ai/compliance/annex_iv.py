# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Fail-closed EU AI Act Annex IV technical-documentation template."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = [
    "AnnexIVTechnicalDocumentationContext",
    "build_annex_iv_template",
    "render_annex_iv_markdown",
]


@dataclass(frozen=True)
class AnnexIVTechnicalDocumentationContext:
    """Operator-authored deployment facts required by the Annex IV template.

    Every field is deliberately required. Use an explicit ``"not applicable —
    <reason>"`` value when an item does not apply to the deployment; the
    reporter must not infer a provider's architecture, standards, lifecycle
    controls, or conformity declaration from runtime telemetry.
    """

    provider_name: str
    system_version: str
    previous_version_relationship: str
    external_dependencies: str
    software_firmware_requirements: str
    distribution_forms: str
    intended_hardware: str
    user_interface: str
    instructions_for_use: str
    development_methods: str
    design_specifications: str
    architecture_and_resources: str
    data_requirements: str
    predetermined_changes: str
    validation_and_testing: str
    monitoring_functioning_control: str
    performance_metric_rationale: str
    lifecycle_changes: str
    standards_and_specifications: str
    eu_declaration_of_conformity_ref: str

    def __post_init__(self) -> None:
        """Reject omissions instead of emitting an apparently complete file."""
        for field_name, value in self.__dict__.items():
            if not value.strip():
                raise ValueError(f"annex_iv.{field_name} is required")


class _ReportMeasurements(Protocol):
    """Article 15 measurements consumed by the Annex IV projection."""

    report_timestamp: float
    period_start: float
    period_end: float
    total_interactions: int
    overall_hallucination_rate: float
    overall_hallucination_rate_ci: float
    avg_score: float
    avg_verdict_confidence: float
    avg_latency_ms: float
    drift_detected: bool
    drift_severity: float
    incident_count: int
    human_override_count: int
    human_override_rate: float


class _ArticleContext(Protocol):
    """Operator narrative fields consumed by the Annex IV projection."""

    @property
    def system_name(self) -> str: ...

    @property
    def intended_purpose(self) -> str: ...

    @property
    def deployment_context(self) -> str: ...

    @property
    def risk_management_summary(self) -> str: ...

    @property
    def cybersecurity_summary(self) -> str: ...

    @property
    def human_oversight_summary(self) -> str: ...

    @property
    def post_market_monitoring_summary(self) -> str: ...

    @property
    def known_limitations(self) -> tuple[str, ...]: ...

    @property
    def residual_risks(self) -> tuple[str, ...]: ...

    @property
    def evidence_refs(self) -> tuple[str, ...]: ...

    @property
    def annex_iv(self) -> AnnexIVTechnicalDocumentationContext | None: ...


def _measured_performance(report: _ReportMeasurements) -> dict[str, object]:
    """Project only supported runtime measurements into the template."""
    return {
        "reporting_period": {
            "start": time.strftime("%Y-%m-%d", time.gmtime(report.period_start)),
            "end": time.strftime("%Y-%m-%d", time.gmtime(report.period_end)),
        },
        "total_interactions": report.total_interactions,
        "overall_hallucination_rate": report.overall_hallucination_rate,
        "overall_hallucination_rate_ci": report.overall_hallucination_rate_ci,
        "avg_score": report.avg_score,
        "avg_verdict_confidence": report.avg_verdict_confidence,
        "avg_latency_ms": report.avg_latency_ms,
        "drift_detected": report.drift_detected,
        "drift_severity": report.drift_severity,
        "incident_count": report.incident_count,
        "human_override_count": report.human_override_count,
        "human_override_rate": report.human_override_rate,
    }


def build_annex_iv_template(
    report: _ReportMeasurements,
    context: _ArticleContext,
) -> dict[str, object]:
    """Build the official nine-section Annex IV structure."""
    annex = context.annex_iv
    if annex is None:
        raise ValueError("annex_iv context is required")
    measured_performance = _measured_performance(report)
    return {
        "document": "EU AI Act Annex IV technical documentation",
        "legal_basis": {
            "regulation": "Regulation (EU) 2024/1689",
            "article": "Article 11(1)",
            "annex": "Annex IV",
            "official_source": "https://eur-lex.europa.eu/eli/reg/2024/1689/oj",
        },
        "generated_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(report.report_timestamp),
        ),
        "sections": {
            "1_general_description": {
                "system_name": context.system_name,
                "intended_purpose": context.intended_purpose,
                "provider_name": annex.provider_name,
                "system_version": annex.system_version,
                "previous_version_relationship": annex.previous_version_relationship,
                "external_dependencies": annex.external_dependencies,
                "software_firmware_requirements": (
                    annex.software_firmware_requirements
                ),
                "distribution_forms": annex.distribution_forms,
                "intended_hardware": annex.intended_hardware,
                "user_interface": annex.user_interface,
                "instructions_for_use": annex.instructions_for_use,
                "deployment_context": context.deployment_context,
            },
            "2_development_and_system_elements": {
                "development_methods": annex.development_methods,
                "design_specifications": annex.design_specifications,
                "architecture_and_resources": annex.architecture_and_resources,
                "data_requirements": annex.data_requirements,
                "human_oversight": context.human_oversight_summary,
                "predetermined_changes": annex.predetermined_changes,
                "validation_and_testing": annex.validation_and_testing,
                "cybersecurity_measures": context.cybersecurity_summary,
                "measured_performance": measured_performance,
            },
            "3_monitoring_functioning_and_control": {
                "summary": annex.monitoring_functioning_control,
                "known_limitations": list(context.known_limitations),
                "residual_risks": list(context.residual_risks),
                "human_oversight": context.human_oversight_summary,
                "measured_performance": measured_performance,
            },
            "4_performance_metrics": {
                "appropriateness_rationale": annex.performance_metric_rationale,
                "measured_performance": measured_performance,
            },
            "5_risk_management_system": {
                "summary": context.risk_management_summary,
            },
            "6_lifecycle_changes": {"summary": annex.lifecycle_changes},
            "7_standards_and_specifications": {
                "summary": annex.standards_and_specifications,
            },
            "8_eu_declaration_of_conformity": {
                "reference": annex.eu_declaration_of_conformity_ref,
            },
            "9_post_market_monitoring": {
                "summary": context.post_market_monitoring_summary,
                "measured_performance": measured_performance,
            },
        },
        "evidence_refs": list(context.evidence_refs),
        "privacy": {
            "payload_classification": "tenant_safe",
            "raw_interaction_text_included": False,
        },
        "claim_boundary": {
            "operator_authored_context_required": True,
            "conformity_assessment_claimed": False,
            "legal_advice": False,
        },
    }


def _require_dict(value: object, field_name: str) -> dict[str, Any]:
    """Return a mapping payload field or raise a stable runtime error."""
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dict")
    return value


def _require_list(value: object, field_name: str) -> list[Any]:
    """Return a list payload field or raise a stable runtime error."""
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return value


def render_annex_iv_markdown(
    report: _ReportMeasurements,
    context: _ArticleContext,
) -> str:
    """Render the Annex IV structure as reviewable Markdown."""
    payload = build_annex_iv_template(report, context)
    sections = _require_dict(payload["sections"], "sections")
    section_titles = (
        ("1_general_description", "1. General Description"),
        (
            "2_development_and_system_elements",
            "2. Development Process and System Elements",
        ),
        (
            "3_monitoring_functioning_and_control",
            "3. Monitoring, Functioning, and Control",
        ),
        ("4_performance_metrics", "4. Performance Metrics"),
        ("5_risk_management_system", "5. Risk Management System"),
        ("6_lifecycle_changes", "6. Lifecycle Changes"),
        (
            "7_standards_and_specifications",
            "7. Standards and Technical Specifications",
        ),
        ("8_eu_declaration_of_conformity", "8. EU Declaration of Conformity"),
        ("9_post_market_monitoring", "9. Post-Market Monitoring"),
    )
    lines = [
        "# EU AI Act Annex IV Technical Documentation",
        "",
        "Structured according to Annex IV referenced by Article 11(1) of ",
        "[Regulation (EU) 2024/1689](https://eur-lex.europa.eu/eli/reg/2024/1689/oj).",
        "",
        "> Operator-authored evidence template; not a conformity assessment or legal advice.",
    ]

    def append_fields(values: dict[str, Any], *, indent: str = "") -> None:
        for field_name, value in values.items():
            label = field_name.replace("_", " ").capitalize()
            if isinstance(value, dict):
                lines.append(f"{indent}- {label}:")
                append_fields(value, indent=f"{indent}  ")
            elif isinstance(value, list):
                lines.append(f"{indent}- {label}:")
                if value:
                    lines.extend(f"{indent}  - {item}" for item in value)
                else:
                    lines.append(f"{indent}  - None supplied.")
            elif isinstance(value, bool):
                lines.append(f"{indent}- {label}: {str(value).lower()}")
            else:
                lines.append(f"{indent}- {label}: {value}")

    for section_key, title in section_titles:
        lines.extend(["", f"## {title}", ""])
        append_fields(_require_dict(sections[section_key], section_key))

    evidence_refs = _require_list(payload["evidence_refs"], "evidence_refs")
    lines.extend(["", "## Evidence References", ""])
    if evidence_refs:
        lines.extend(f"- {ref}" for ref in evidence_refs)
    else:
        lines.append("- No evidence references supplied.")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- Conformity assessment claimed: false",
            "- Legal advice: false",
            "- Raw interaction text included: false",
        ]
    )
    return "\n".join(lines)
