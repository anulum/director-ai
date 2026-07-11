# SPDX-License-Identifier: BUSL-1.1
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Safety dashboard analytics contracts

"""Contract tests for the safety-dashboard event analytics module.

``director_ai.ui._dashboard_analytics`` owns the JSONL parsing and the
row/drift/risk aggregations of the safety operations dashboard;
``safety_dashboard`` re-exports ``parse_dashboard_records`` and builds
its reports from the private helpers. These tests pin where the
analytics live, the facade wiring, and the parser/validator contracts;
the aggregation content matrix stays in ``tests/test_safety_dashboard.py``.
"""

from __future__ import annotations

import json

import pytest

import director_ai.ui._dashboard_analytics as dashboard_analytics_module
from director_ai.ui import safety_dashboard


class TestModulePlacement:
    def test_parser_is_defined_in_the_analytics_module(self):
        parser = dashboard_analytics_module.parse_dashboard_records
        assert parser.__module__ == dashboard_analytics_module.__name__

    def test_facade_re_exports_the_same_parser_object(self):
        assert (
            safety_dashboard.parse_dashboard_records
            is dashboard_analytics_module.parse_dashboard_records
        )

    def test_module_all_names_the_facade_contract(self):
        assert set(dashboard_analytics_module.__all__) == {
            "_drift_alert_rows",
            "_evidence_rows",
            "_feedback_tune_samples",
            "_operations_risk_level",
            "_source_rows",
            "_summary_markdown",
            "_tenant_rows",
            "_trust_risk_level",
            "_validated_min_window_events",
            "_validated_rate",
            "parse_dashboard_records",
        }
        for name in dashboard_analytics_module.__all__:
            func = getattr(dashboard_analytics_module, name)
            assert func.__module__ == dashboard_analytics_module.__name__


class TestParserContracts:
    def test_parser_collects_records_and_line_scoped_warnings(self):
        events = "\n".join(
            [
                json.dumps({"tenant_id": "acme", "policy_decision": "halt"}),
                "not-json",
                json.dumps(["not", "an", "object"]),
            ]
        )
        records, errors = dashboard_analytics_module.parse_dashboard_records(events)
        assert [record.tenant_id for record in records] == ["acme"]
        assert any(error.startswith("events:2:") for error in errors)
        assert any(error.startswith("events:3:") for error in errors)

    def test_rate_validator_rejects_out_of_range_thresholds(self):
        with pytest.raises(ValueError, match="halt_alert_threshold"):
            safety_dashboard.build_safety_dashboard("", halt_alert_threshold=1.5)

    def test_min_window_validator_rejects_non_positive_counts(self):
        with pytest.raises(ValueError, match="min_drift_window_events"):
            safety_dashboard.build_observability_operations_report(
                "", min_drift_window_events=0
            )
