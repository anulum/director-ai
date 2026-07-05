# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — UI subpackage (Gradio-based config wizard + dashboard)

"""Gradio-based configuration wizard and safety/observability dashboard."""

from .safety_dashboard import (
    ComplianceExportRef,
    ObservabilityOperationsReport,
    TrustConsoleReport,
    TrustControl,
    build_observability_operations_report,
    build_trust_console_report,
)
from .streaming_halt_live import (
    SCENARIOS,
    Frame,
    Scenario,
    render_banner,
    render_gauge,
    render_token_span,
    run_live_demo,
    wrap_token_strip,
)

__all__ = [
    "ComplianceExportRef",
    "Frame",
    "ObservabilityOperationsReport",
    "SCENARIOS",
    "Scenario",
    "TrustConsoleReport",
    "TrustControl",
    "build_observability_operations_report",
    "build_trust_console_report",
    "render_banner",
    "render_gauge",
    "render_token_span",
    "run_live_demo",
    "wrap_token_strip",
]
