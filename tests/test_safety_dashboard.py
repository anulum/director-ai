# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - safety dashboard tests

from __future__ import annotations

import json

from director_ai.ui.safety_dashboard import (
    EVIDENCE_COLUMNS,
    SOURCE_COLUMNS,
    TENANT_COLUMNS,
    build_safety_dashboard,
    parse_dashboard_records,
)


def _line(payload: dict) -> str:
    return json.dumps(payload) + "\n"


class TestSafetyDashboard:
    def test_tables_have_stable_columns(self):
        assert TENANT_COLUMNS == [
            "tenant_id",
            "events",
            "halts",
            "halt_rate",
            "false_positives",
            "false_positive_rate",
            "alert",
        ]
        assert "source" in SOURCE_COLUMNS
        assert "action" in EVIDENCE_COLUMNS

    def test_safety_event_builds_per_tenant_halt_rate(self):
        events = _line(
            {
                "event_id": "e1",
                "tenant_id": "tenant-a",
                "timestamp": "2026-04-30T00:00:00Z",
                "policy_decision": "halt",
                "halt_reason": "contradiction",
                "observed_score": 0.22,
                "trace_attribution": {"fact_source": "kb://physics"},
                "tenant_safe_explanation": "Refresh the cited fact.",
            },
        ) + _line(
            {
                "event_id": "e2",
                "tenant_id": "tenant-a",
                "policy_decision": "allow",
                "observed_score": 0.91,
            },
        )

        summary, tenants, sources, evidence, command = build_safety_dashboard(
            events,
            halt_alert_threshold=0.4,
        )

        assert "Events: 2" in summary
        assert tenants == [["tenant-a", 2, 1, 0.5, 0, 0.0, "halt-rate"]]
        assert sources == [["kb://physics", 1, 1, "contradiction"]]
        assert evidence[0][2] == "e1"
        assert evidence[0][5] == 0.22
        assert "director-ai tune" in command

    def test_feedback_marks_false_positive_rate(self):
        events = _line(
            {
                "event_id": "e1",
                "tenant_id": "tenant-a",
                "policy_decision": "halt",
                "halt_reason": "window_average",
                "evidence_refs": ["kb://stale"],
            },
        )
        feedback = _line(
            {
                "event_id": "e1",
                "tenant_id": "tenant-a",
                "guardrail_approved": False,
                "human_approved": True,
                "source": "kb://stale",
            },
        )

        _summary, tenants, _sources, evidence, _command = build_safety_dashboard(
            events,
            feedback,
            false_positive_alert_threshold=0.2,
        )

        assert tenants[0][4] == 1
        assert tenants[0][5] == 1.0
        assert "false-positive" in tenants[0][6]
        assert evidence[-1][3] == "feedback"
        assert evidence[-1][7] == "Retune from labelled feedback."

    def test_parse_errors_are_reported_without_dropping_valid_rows(self):
        events = "{broken\n" + _line(
            {
                "tenant_id": "tenant-b",
                "decision": "block",
                "reason": "ontology",
                "attributes": {"contradiction_source": "kb://ontology"},
            },
        )

        summary, tenants, sources, _evidence, _command = build_safety_dashboard(events)

        assert "Parse warnings" in summary
        assert tenants[0][0] == "tenant-b"
        assert sources[0][0] == "kb://ontology"

    def test_non_false_positive_feedback_is_ignored(self):
        records, errors = parse_dashboard_records(
            "",
            _line(
                {
                    "tenant_id": "tenant-a",
                    "guardrail_approved": True,
                    "human_approved": True,
                },
            ),
        )

        assert records == []
        assert errors == []

    def test_empty_inputs_return_operator_guidance(self):
        summary, tenants, sources, evidence, command = build_safety_dashboard("")

        assert "load SafetyEvent JSONL" in summary
        assert tenants == []
        assert sources == []
        assert evidence == []
        assert command.startswith("director-ai tune")
