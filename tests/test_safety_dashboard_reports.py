# SPDX-License-Identifier: BUSL-1.1
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Safety dashboard report-model contracts

"""Contract tests for the safety-dashboard report models module.

``director_ai.ui._dashboard_reports`` owns the column layouts, status
vocabularies, and frozen record/report dataclasses of the safety
operations dashboard; ``safety_dashboard`` re-exports them unchanged.
These tests pin where the models live, the re-export identity, and the
validation contracts; the report content matrix stays in
``tests/test_safety_dashboard.py``.
"""

from __future__ import annotations

import pytest

import director_ai.ui._dashboard_reports as dashboard_reports_module
from director_ai.ui import safety_dashboard

_MODEL_NAMES = (
    "HaltDashboardRecord",
    "TrustControl",
    "ComplianceExportRef",
    "TrustConsoleReport",
    "ObservabilityOperationsReport",
)
_COLUMN_NAMES = (
    "TENANT_COLUMNS",
    "SOURCE_COLUMNS",
    "EVIDENCE_COLUMNS",
    "DRIFT_ALERT_COLUMNS",
    "COMPLIANCE_EXPORT_COLUMNS",
)


class TestModulePlacement:
    def test_models_are_defined_in_the_reports_module(self):
        for name in _MODEL_NAMES:
            cls = getattr(dashboard_reports_module, name)
            assert cls.__module__ == dashboard_reports_module.__name__

    def test_facade_re_exports_models_and_columns_unchanged(self):
        for name in (*_MODEL_NAMES, *_COLUMN_NAMES):
            assert getattr(safety_dashboard, name) is getattr(
                dashboard_reports_module, name
            )

    def test_module_all_covers_models_columns_and_status_aliases(self):
        assert set(dashboard_reports_module.__all__) == {
            *_MODEL_NAMES,
            *_COLUMN_NAMES,
            "TrustControlStatus",
            "ComplianceExportStatus",
        }

    def test_ui_package_surface_is_unchanged(self):
        from director_ai import ui

        assert ui.TrustControl is dashboard_reports_module.TrustControl
        assert (
            ui.ObservabilityOperationsReport
            is dashboard_reports_module.ObservabilityOperationsReport
        )


class TestValidationContracts:
    def test_trust_control_normalises_and_validates_status(self):
        control = dashboard_reports_module.TrustControl(
            control=" halt-drill ",
            status=" PASSED ",
            evidence_ref=" runs/1 ",
        )
        assert (control.control, control.status) == ("halt-drill", "passed")
        with pytest.raises(ValueError, match="status must be one of"):
            dashboard_reports_module.TrustControl(
                control="x", status="green", evidence_ref="runs/1"
            )

    def test_compliance_export_requires_core_fields(self):
        with pytest.raises(ValueError, match="standard is required"):
            dashboard_reports_module.ComplianceExportRef(
                standard=" ", name="SOC2", status="available", evidence_ref="runs/2"
            )

    def test_reports_serialise_tenant_safe_payloads(self):
        report = dashboard_reports_module.TrustConsoleReport(
            title="t",
            generated_at="now",
            summary={"total_events": 0},
            tenants=[],
            recent_evidence=[],
            controls=(),
        )
        payload = report.to_dict()
        assert payload["privacy"]["payload_classification"] == "tenant_safe"
        assert payload["tenant_columns"] == list(
            dashboard_reports_module.TENANT_COLUMNS
        )
        assert "No tenant events supplied." in report.to_markdown()
