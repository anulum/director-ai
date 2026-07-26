# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# compliance subpackage — EU AI Act Article 15 reporting and audit trail

"""Compliance: audit log, drift/cost/feedback analysis, and readiness reporting."""

from .annex_iv import AnnexIVTechnicalDocumentationContext
from .audit_log import AuditEntry, AuditLog
from .drift_detector import DriftDetector, DriftResult
from .governance_controls import (
    ControlSignal,
    GovernanceControl,
    GovernanceControlsReport,
    compute_governance_controls,
)
from .readiness import (
    HipaaDeploymentObligation,
    HipaaDocumentationPacket,
    ReadinessStatus,
    Soc2IsoControl,
    Soc2IsoReadinessReport,
    build_hipaa_documentation_packet,
    build_soc2_iso_readiness_report,
    default_hipaa_obligations,
    default_readiness_controls,
)
from .reporter import Article15Report, Article15TemplateContext, ComplianceReporter

__all__ = [
    "AnnexIVTechnicalDocumentationContext",
    "Article15Report",
    "Article15TemplateContext",
    "AuditEntry",
    "AuditLog",
    "ComplianceReporter",
    "ControlSignal",
    "GovernanceControl",
    "GovernanceControlsReport",
    "compute_governance_controls",
    "DriftDetector",
    "DriftResult",
    "HipaaDeploymentObligation",
    "HipaaDocumentationPacket",
    "ReadinessStatus",
    "Soc2IsoControl",
    "Soc2IsoReadinessReport",
    "build_hipaa_documentation_packet",
    "build_soc2_iso_readiness_report",
    "default_hipaa_obligations",
    "default_readiness_controls",
]
