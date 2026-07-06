# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - safety dashboard tests

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import ModuleType, TracebackType
from typing import Literal, TypedDict

import pytest

import director_ai.ui.safety_dashboard as dashboard_mod
from director_ai.ui.safety_dashboard import (
    COMPLIANCE_EXPORT_COLUMNS,
    DRIFT_ALERT_COLUMNS,
    EVIDENCE_COLUMNS,
    SOURCE_COLUMNS,
    TENANT_COLUMNS,
    ComplianceExportRef,
    TrustControl,
    build_observability_operations_markdown,
    build_observability_operations_report,
    build_retune_guidance,
    build_safety_dashboard,
    build_trust_console_report,
    launch_safety_dashboard,
    parse_dashboard_records,
)


def _line(payload: dict[str, object]) -> str:
    return json.dumps(payload) + "\n"


@dataclass(frozen=True)
class _FakeTuneResult:
    """Typed result returned by the tuner test double."""

    threshold: float
    w_logic: float
    w_fact: float
    balanced_accuracy: float
    confidence_level: str


class _FakeTunerModule(ModuleType):
    """Strict-typed replacement for the optional tuner module."""

    def tune(self, samples: list[dict[str, object]]) -> _FakeTuneResult:
        """Return deterministic threshold guidance for dashboard tests."""

        labels = [bool(sample["label"]) for sample in samples]
        return _FakeTuneResult(
            threshold=0.42,
            w_logic=0.5,
            w_fact=0.5,
            balanced_accuracy=1.0 if len(set(labels)) > 1 else 0.5,
            confidence_level="unit-test",
        )

    def format_profile_overlay(
        self,
        result: _FakeTuneResult,
        *,
        profile: str,
        base_profile: str,
    ) -> str:
        """Render the same profile overlay shape used by the real tuner."""

        lines = [
            f'profile: "{profile}"',
            f"coherence_threshold: {result.threshold:.4f}",
            f"w_logic: {result.w_logic:.4f}",
            f"w_fact: {result.w_fact:.4f}",
        ]
        if base_profile:
            lines.append(f'tuned_from_profile: "{base_profile}"')
        return "\n".join(lines)


class TestSafetyDashboard:
    @pytest.fixture(autouse=True)
    def _fake_tuner_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_tuner = _FakeTunerModule("director_ai.core.training.tuner")
        monkeypatch.setitem(sys.modules, "director_ai.core.training.tuner", fake_tuner)

    def test_tables_have_stable_columns(self) -> None:
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

    def test_safety_event_builds_per_tenant_halt_rate(self) -> None:
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

    def test_feedback_marks_false_positive_rate(self) -> None:
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

    def test_parse_errors_are_reported_without_dropping_valid_rows(self) -> None:
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

    def test_non_false_positive_feedback_is_ignored(self) -> None:
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

    def test_empty_inputs_return_operator_guidance(self) -> None:
        summary, tenants, sources, evidence, command = build_safety_dashboard("")

        assert "load SafetyEvent JSONL" in summary
        assert tenants == []
        assert sources == []
        assert evidence == []
        assert command.startswith("director-ai tune")

    def test_retune_guidance_builds_profile_overlay_from_feedback(self) -> None:
        feedback = "".join(
            _line(
                {
                    "prompt": f"approved {idx}",
                    "response": "The sky is blue.",
                    "human_approved": True,
                },
            )
            for idx in range(4)
        ) + "".join(
            _line(
                {
                    "prompt": f"rejected {idx}",
                    "response": "Dolphins can fly.",
                    "human_approved": False,
                },
            )
            for idx in range(4)
        )

        summary, overlay = build_retune_guidance(
            feedback,
            profile="support_tuned",
            base_profile="customer_support",
            min_samples=4,
        )

        assert "Labelled samples: 8" in summary
        assert "Selected threshold" in summary
        assert 'profile: "support_tuned"' in overlay
        assert 'tuned_from_profile: "customer_support"' in overlay
        assert "coherence_threshold" in overlay

    def test_retune_guidance_requires_labelled_prompt_response_rows(self) -> None:
        summary, overlay = build_retune_guidance(
            _line({"event_id": "missing-fields", "human_approved": True}),
            min_samples=2,
        )

        assert "Required samples: 2" in summary
        assert "Parse warnings" in summary
        assert overlay == ""

    def test_retune_guidance_defaults_profile_from_base_profile(self) -> None:
        feedback = "".join(
            _line(
                {
                    "input": f"accepted {idx}",
                    "output": "The account is active.",
                    "label": "accepted",
                },
            )
            for idx in range(2)
        ) + "".join(
            _line(
                {
                    "query": f"blocked {idx}",
                    "completion": "The account is closed.",
                    "label": "blocked",
                },
            )
            for idx in range(2)
        )

        summary, overlay = build_retune_guidance(
            feedback,
            profile="",
            base_profile="finance",
            min_samples=4,
        )

        assert "Approved labels: 2" in summary
        assert "Rejected labels: 2" in summary
        assert 'profile: "finance_tuned"' in overlay

    def test_retune_guidance_reports_provisional_single_class_feedback(self) -> None:
        feedback = "".join(
            _line(
                {
                    "prompt": f"approved {idx}",
                    "response": "The invoice is paid.",
                    "label": 1,
                },
            )
            for idx in range(4)
        )

        summary, overlay = build_retune_guidance(feedback, min_samples=4)

        assert "Warning: only one label class present" in summary
        assert 'profile: "tuned"' in overlay

    def test_retune_guidance_keeps_parse_warnings_with_enough_valid_rows(self) -> None:
        feedback = _line({"prompt": "missing response", "human_approved": True})
        feedback += "".join(
            _line(
                {
                    "prompt": f"ok {idx}",
                    "response": "The response is supported.",
                    "human_approved": idx % 2 == 0,
                },
            )
            for idx in range(4)
        )

        summary, overlay = build_retune_guidance(feedback, min_samples=4)

        assert "Parse warnings" in summary
        assert "coherence_threshold" in overlay

    def test_retune_guidance_accepts_boolean_and_numeric_labels(self) -> None:
        feedback = _line(
            {
                "prompt": "bool accepted",
                "response": "The document is current.",
                "label": True,
            },
        )
        feedback += _line(
            {
                "prompt": "numeric rejected",
                "response": "The document is obsolete.",
                "label": 0,
            },
        )

        summary, overlay = build_retune_guidance(feedback, min_samples=2)

        assert "Approved labels: 1" in summary
        assert "Rejected labels: 1" in summary
        assert "coherence_threshold" in overlay

    def test_retune_guidance_rejects_unknown_label_text(self) -> None:
        summary, overlay = build_retune_guidance(
            _line(
                {
                    "prompt": "unknown",
                    "response": "unknown",
                    "label": "maybe",
                },
            ),
            min_samples=1,
        )

        assert "Labelled samples: 0" in summary
        assert "Parse warnings" in summary
        assert overlay == ""

    def test_retune_guidance_reports_parse_warnings_before_enough_samples(self) -> None:
        summary, overlay = build_retune_guidance(
            "{broken\n"
            + _line(
                {
                    "prompt": "approved",
                    "response": "The response is supported.",
                    "label": "approved",
                },
            ),
            min_samples=2,
        )

        assert "Labelled samples: 1" in summary
        assert "Parse warnings: feedback:1" in summary
        assert overlay == ""

    def test_retune_guidance_reports_clean_insufficient_sample_guidance(self) -> None:
        summary, overlay = build_retune_guidance(
            _line(
                {
                    "prompt": "approved",
                    "response": "The response is supported.",
                    "label": "approved",
                },
            ),
            min_samples=2,
        )

        assert "Labelled samples: 1" in summary
        assert "Parse warnings" not in summary
        assert overlay == ""

    def test_retune_guidance_accepts_label_synonyms(self) -> None:
        accepted = ["approved", "approve", "correct", "true", "1"]
        rejected = ["rejected", "reject", "incorrect", "false", "0"]
        feedback = "".join(
            _line(
                {
                    "prompt": f"approved {label}",
                    "response": "Supported answer.",
                    "label": label,
                },
            )
            for label in accepted
        )
        feedback += "".join(
            _line(
                {
                    "prompt": f"rejected {label}",
                    "response": "Unsupported answer.",
                    "label": label,
                },
            )
            for label in rejected
        )

        summary, overlay = build_retune_guidance(feedback, min_samples=10)

        assert "Approved labels: 5" in summary
        assert "Rejected labels: 5" in summary
        assert "coherence_threshold" in overlay

    def test_dashboard_reports_non_object_feedback_and_blank_lines(self) -> None:
        summary, tenants, sources, evidence, _command = build_safety_dashboard(
            "\n" + _line({"tenant_id": "tenant-a", "decision": "allow"}),
            "[]\n",
        )

        assert "feedback:1: expected object" in summary
        assert tenants == [["tenant-a", 1, 0, 0.0, 0, 0.0, "ok"]]
        assert sources == []
        assert evidence == []

    def test_dashboard_extracts_nested_sources_and_resilient_scores(self) -> None:
        events = _line(
            {
                "event_id": "direct",
                "tenant_id": "",
                "decision": "halted",
                "contradiction_source": "kb://direct",
                "score": "not-a-number",
            },
        )
        events += _line(
            {
                "event_id": "nested",
                "decision": "block",
                "attributes": {
                    "tenant_id": "tenant-nested",
                    "fact_source": "kb://attributes",
                },
                "halt_evidence": {"source": "kb://evidence"},
            },
        )
        events += _line(
            {
                "event_id": "chunk",
                "decision": "halt",
                "trace_attribution": {"source": "kb://trace"},
                "evidence_chunks": [{"id": "chunk-1"}],
            },
        )

        _summary, tenants, sources, evidence, _command = build_safety_dashboard(events)

        assert tenants[0][0] == "default"
        assert ["kb://direct", 1, 1, ""] in sources
        assert ["kb://attributes", 1, 1, ""] in sources
        assert ["kb://trace", 1, 1, ""] in sources
        assert evidence[0][5] == ""

    def test_launch_safety_dashboard_reports_missing_dependency(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setitem(sys.modules, "gradio", None)

        try:
            launch_safety_dashboard()
        except ImportError as exc:
            assert "director-ai[ui]" in str(exc)
        else:
            raise AssertionError("launch_safety_dashboard should require Gradio")

    def test_launch_safety_dashboard_wires_retune_command(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = _FakeGradio()
        monkeypatch.setitem(sys.modules, "gradio", fake.module)

        launch_safety_dashboard(port=7871, share=True)

        labels = [component.label for component in fake.components]
        assert "SafetyEvent JSONL" in labels
        assert "Feedback JSONL" in labels
        assert "Retune command" in labels
        assert "Drift alert threshold" in labels
        assert "Observability operations report" in labels
        assert "Render Dashboard" in labels
        assert fake.launch_kwargs == {"server_port": 7871, "share": True}

        dashboard_clicks = [
            click
            for click in fake.clicks
            if getattr(click["fn"], "__name__", "") == "build_safety_dashboard"
        ]
        assert len(dashboard_clicks) == 1
        assert [component.label for component in dashboard_clicks[0]["inputs"]] == [
            "SafetyEvent JSONL",
            "Feedback JSONL",
            "Halt-rate alert threshold",
            "False-positive alert threshold",
        ]
        operations_clicks = [
            click
            for click in fake.clicks
            if getattr(click["fn"], "__name__", "")
            == "build_observability_operations_markdown"
        ]
        assert len(operations_clicks) == 1
        assert [component.label for component in operations_clicks[0]["inputs"]] == [
            "SafetyEvent JSONL",
            "Feedback JSONL",
            "Halt-rate alert threshold",
            "False-positive alert threshold",
            "Drift alert threshold",
        ]


class TestTrustConsole:
    def test_report_is_tenant_safe_and_excludes_raw_payload_fields(self) -> None:
        events = _line(
            {
                "event_id": "e1",
                "tenant_id": "tenant-a",
                "timestamp": "2026-05-17T12:00:00Z",
                "policy_decision": "halt",
                "halt_reason": "contradiction",
                "observed_score": 0.19,
                "trace_attribution": {"fact_source": "kb://policy-v4"},
                "tenant_safe_explanation": "Refresh the policy source.",
                "prompt": "raw user prompt must not leave the tenant",
                "response": "raw model answer must not leave the tenant",
                "customer_email": "jane@example.com",
            },
        )

        report = build_trust_console_report(
            events,
            controls=[
                TrustControl(
                    control="PII redaction",
                    status="passed",
                    evidence_ref="docs/BENCHMARKS.md#pii-redaction",
                    owner="security",
                    updated_at="2026-05-17",
                ),
            ],
            generated_at="2026-05-17T12:05:00Z",
        )
        payload = report.to_dict()
        serialized = json.dumps(payload, sort_keys=True)

        assert payload["title"] == "Director-AI Trust Console"
        assert payload["privacy"] == {
            "payload_classification": "tenant_safe",
            "raw_event_text_included": False,
            "raw_feedback_text_included": False,
        }
        assert payload["summary"]["total_events"] == 1
        assert payload["summary"]["halt_rate"] == 1.0
        assert payload["summary"]["risk_level"] == "attention_required"
        assert payload["controls"] == [
            {
                "control": "PII redaction",
                "status": "passed",
                "evidence_ref": "docs/BENCHMARKS.md#pii-redaction",
                "owner": "security",
                "updated_at": "2026-05-17",
            }
        ]
        assert "tenant-a" in serialized
        assert "raw user prompt" not in serialized
        assert "raw model answer" not in serialized
        assert "jane@example.com" not in serialized

    def test_report_marks_failing_control_as_critical(self) -> None:
        report = build_trust_console_report(
            "",
            controls=[
                TrustControl(
                    control="External security test",
                    status="failing",
                    evidence_ref="docs/internal/security.md",
                ),
            ],
        )

        assert report.to_dict()["summary"]["risk_level"] == "critical"
        assert "External security test" in report.to_markdown()
        assert "critical" in report.to_markdown()

    def test_control_status_is_validated(self) -> None:
        try:
            TrustControl(control="SOC 2", status="maybe", evidence_ref="soc2.md")
        except ValueError as exc:
            assert "status" in str(exc)
        else:
            raise AssertionError("TrustControl should reject unknown statuses")

    @pytest.mark.parametrize(
        ("control", "evidence_ref", "match"),
        [
            ("", "soc2.md", "control"),
            ("SOC 2", "", "evidence_ref"),
        ],
    )
    def test_control_required_fields_are_validated(
        self, control: str, evidence_ref: str, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            TrustControl(control=control, status="passed", evidence_ref=evidence_ref)

    def test_markdown_renders_empty_control_and_tenant_sections(self) -> None:
        report = build_trust_console_report("")

        markdown = report.to_markdown()

        assert "No readiness controls supplied." in markdown
        assert "No tenant events supplied." in markdown

    def test_markdown_renders_parse_warnings_and_tenant_rows(self) -> None:
        report = build_trust_console_report(
            "{broken\n"
            + _line(
                {
                    "tenant_id": "tenant-a",
                    "policy_decision": "halt",
                    "halt_reason": "contradiction",
                },
            ),
        )

        markdown = report.to_markdown()

        assert "Parse Warnings" in markdown
        assert "| tenant-a | 1 | 1 | 1.0 | 0 | 0.0 | halt-rate |" in markdown


class TestObservabilityOperationsReport:
    def test_report_contains_drift_forensics_and_excludes_raw_payloads(self) -> None:
        events = "".join(
            [
                _line(
                    {
                        "event_id": "baseline-1",
                        "tenant_id": "tenant-a",
                        "timestamp": "2026-06-01T00:00:00Z",
                        "policy_decision": "allow",
                        "observed_score": 0.92,
                        "prompt": "raw prompt must stay tenant-local",
                    },
                ),
                _line(
                    {
                        "event_id": "baseline-2",
                        "tenant_id": "tenant-a",
                        "timestamp": "2026-06-01T00:01:00Z",
                        "policy_decision": "allow",
                        "observed_score": 0.88,
                        "response": "raw response must stay tenant-local",
                    },
                ),
                _line(
                    {
                        "event_id": "current-1",
                        "tenant_id": "tenant-a",
                        "timestamp": "2026-06-01T00:02:00Z",
                        "policy_decision": "halt",
                        "halt_reason": "contradiction",
                        "observed_score": 0.24,
                        "trace_attribution": {"fact_source": "kb://policy-v5"},
                        "tenant_safe_explanation": "Review policy-v5 source.",
                    },
                ),
                _line(
                    {
                        "event_id": "current-2",
                        "tenant_id": "tenant-a",
                        "timestamp": "2026-06-01T00:03:00Z",
                        "policy_decision": "halt",
                        "halt_reason": "contradiction",
                        "observed_score": 0.21,
                        "trace_attribution": {"fact_source": "kb://policy-v5"},
                        "customer_email": "jane@example.com",
                    },
                ),
            ],
        )

        report = build_observability_operations_report(
            events,
            controls=[
                TrustControl(
                    control="Trace retention",
                    status="passed",
                    evidence_ref="runbooks/trace-retention.md",
                ),
            ],
            compliance_exports=[
                ComplianceExportRef(
                    standard="EU AI Act Article 15",
                    name="30-day operations report",
                    status="available",
                    evidence_ref="reports/article15-june.md",
                    updated_at="2026-06-01",
                ),
            ],
            generated_at="2026-06-01T00:05:00Z",
            drift_alert_threshold=0.25,
        )
        payload = report.to_dict()
        serialised = json.dumps(payload, sort_keys=True)

        assert payload["drift_alert_columns"] == DRIFT_ALERT_COLUMNS
        assert payload["compliance_export_columns"] == COMPLIANCE_EXPORT_COLUMNS
        assert payload["summary"]["risk_level"] == "critical"
        assert payload["summary"]["drift_alerts"] == 1
        assert payload["drift_alerts"] == [
            [
                "tenant-a",
                2,
                2,
                0.0,
                1.0,
                1.0,
                "severe",
                "Freeze rollout, review halt traces, and retune before expansion.",
            ]
        ]
        assert payload["privacy"] == {
            "payload_classification": "tenant_safe",
            "raw_event_text_included": False,
            "raw_feedback_text_included": False,
            "raw_compliance_evidence_included": False,
        }
        assert "raw prompt" not in serialised
        assert "raw response" not in serialised
        assert "jane@example.com" not in serialised
        assert "kb://policy-v5" in serialised

    def test_missing_compliance_export_marks_report_critical(self) -> None:
        report = build_observability_operations_report(
            "",
            compliance_exports=[
                ComplianceExportRef(
                    standard="SOC 2",
                    name="Security evidence packet",
                    status="missing",
                    evidence_ref="pending/security-evidence.md",
                ),
            ],
        )

        payload = report.to_dict()
        markdown = report.to_markdown()

        assert payload["summary"]["risk_level"] == "critical"
        assert payload["summary"]["compliance_export_gaps"] == 1
        assert "Security evidence packet" in markdown
        assert "missing" in markdown

    @pytest.mark.parametrize(
        ("standard", "name", "status", "evidence_ref", "match"),
        [
            ("SOC 2", "Security packet", "unknown", "soc2.md", "status"),
            ("", "Security packet", "available", "soc2.md", "standard"),
            ("SOC 2", "", "available", "soc2.md", "name"),
            ("SOC 2", "Security packet", "available", "", "evidence_ref"),
        ],
    )
    def test_compliance_export_required_fields_are_validated(
        self,
        standard: str,
        name: str,
        status: str,
        evidence_ref: str,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            ComplianceExportRef(
                standard=standard,
                name=name,
                status=status,
                evidence_ref=evidence_ref,
            )

    def test_operations_markdown_renders_empty_compliance_and_evidence_sections(
        self,
    ) -> None:
        report = build_observability_operations_report("")

        markdown = report.to_markdown()

        assert "No compliance export references supplied." in markdown
        assert "No halt evidence supplied." in markdown

    def test_operations_markdown_renders_parse_warnings(self) -> None:
        report = build_observability_operations_report(
            "{broken\n"
            + _line(
                {
                    "tenant_id": "tenant-a",
                    "policy_decision": "allow",
                },
            ),
        )

        markdown = report.to_markdown()

        assert "Parse Warnings: events:1" in markdown

    def test_operations_markdown_renders_positive_tables(self) -> None:
        events = "".join(
            [
                _line({"tenant_id": "tenant-a", "policy_decision": "allow"}),
                _line({"tenant_id": "tenant-a", "policy_decision": "allow"}),
                _line(
                    {
                        "tenant_id": "tenant-a",
                        "policy_decision": "halt",
                        "halt_reason": "contradiction",
                        "trace_attribution": {"fact_source": "kb://policy"},
                    },
                ),
                _line(
                    {
                        "tenant_id": "tenant-a",
                        "policy_decision": "halt",
                        "halt_reason": "contradiction",
                        "trace_attribution": {"fact_source": "kb://policy"},
                    },
                ),
            ],
        )
        report = build_observability_operations_report(
            events,
            compliance_exports=[
                ComplianceExportRef(
                    standard="SOC 2",
                    name="Security packet",
                    status="available",
                    evidence_ref="soc2.md",
                ),
            ],
            drift_alert_threshold=0.25,
        )

        markdown = report.to_markdown()

        assert "| tenant-a | 2 | 2 | 0.0 | 1.0 | 1.0 | severe |" in markdown
        assert "| SOC 2 | Security packet | available | soc2.md |  |" in markdown
        assert "kb://policy" in markdown

    def test_drift_alert_requires_enough_events_per_window(self) -> None:
        events = _line(
            {
                "event_id": "baseline",
                "tenant_id": "tenant-a",
                "policy_decision": "allow",
            },
        ) + _line(
            {
                "event_id": "current",
                "tenant_id": "tenant-a",
                "policy_decision": "halt",
            },
        )

        report = build_observability_operations_report(
            events,
            drift_alert_threshold=0.1,
            min_drift_window_events=2,
        )

        assert report.to_dict()["drift_alerts"] == []
        assert report.to_dict()["summary"]["risk_level"] == "attention_required"

    def test_drift_alert_window_ignores_feedback_records(self) -> None:
        events = "".join(
            [
                _line({"tenant_id": "tenant-a", "policy_decision": "allow"}),
                _line({"tenant_id": "tenant-a", "policy_decision": "allow"}),
                _line({"tenant_id": "tenant-a", "policy_decision": "halt"}),
                _line({"tenant_id": "tenant-a", "policy_decision": "halt"}),
            ],
        )
        feedback = "".join(
            _line(
                {
                    "tenant_id": "tenant-a",
                    "event_id": f"feedback-{idx}",
                    "guardrail_approved": False,
                    "human_approved": True,
                },
            )
            for idx in range(4)
        )

        report = build_observability_operations_report(
            events,
            feedback,
            drift_alert_threshold=0.25,
            min_drift_window_events=2,
        )

        assert report.to_dict()["drift_alerts"] == [
            [
                "tenant-a",
                2,
                2,
                0.0,
                1.0,
                1.0,
                "severe",
                "Freeze rollout, review halt traces, and retune before expansion.",
            ]
        ]

    def test_operations_markdown_is_dashboard_ready(self) -> None:
        markdown = build_observability_operations_markdown(
            _line({"tenant_id": "tenant-a", "policy_decision": "allow"}),
        )

        assert "# Director-AI Observability Operations" in markdown
        assert "## Drift Alerts" in markdown
        assert "No drift alerts" in markdown


class TestSafetyDashboardUtilityContracts:
    def test_drift_severity_and_recommendations_are_stable(self) -> None:
        assert dashboard_mod._drift_severity(0.31) == "severe"
        assert dashboard_mod._drift_severity(0.16) == "moderate"
        assert dashboard_mod._drift_severity(0.01) == "mild"
        assert "Freeze rollout" in dashboard_mod._drift_recommendation("severe")
        assert "labelled feedback" in dashboard_mod._drift_recommendation("moderate")
        assert "Monitor the next window" in dashboard_mod._drift_recommendation("mild")

    def test_drift_alert_rows_skip_small_and_stable_windows(self) -> None:
        small = [
            dashboard_mod.HaltDashboardRecord(
                tenant_id="tenant-a",
                event_id="a1",
                timestamp="",
                decision="allow",
                reason="",
                halted=False,
                false_positive=False,
                score=None,
                contradiction_source="unknown",
                action="",
            )
        ]
        assert (
            dashboard_mod._drift_alert_rows(
                small,
                drift_alert_threshold=0.1,
                min_window_events=1,
            )
            == []
        )
        stable = [
            dashboard_mod.HaltDashboardRecord(
                tenant_id="tenant-a",
                event_id=f"a{idx}",
                timestamp="",
                decision="allow",
                reason="",
                halted=False,
                false_positive=False,
                score=None,
                contradiction_source="unknown",
                action="",
            )
            for idx in range(4)
        ]
        assert (
            dashboard_mod._drift_alert_rows(
                stable,
                drift_alert_threshold=0.1,
                min_window_events=2,
            )
            == []
        )

    def test_operations_risk_level_precedence(self) -> None:
        assert (
            dashboard_mod._operations_risk_level(
                tenant_alerts=0,
                drift_alerts=[],
                controls=(
                    TrustControl(
                        control="Security",
                        status="failing",
                        evidence_ref="security.md",
                    ),
                ),
                compliance_exports=(),
                halts=0,
                false_positives=0,
            )
            == "critical"
        )
        assert (
            dashboard_mod._operations_risk_level(
                tenant_alerts=0,
                drift_alerts=[],
                controls=(),
                compliance_exports=(
                    ComplianceExportRef(
                        standard="SOC 2",
                        name="Security packet",
                        status="stale",
                        evidence_ref="soc2.md",
                    ),
                ),
                halts=0,
                false_positives=0,
            )
            == "attention_required"
        )
        assert (
            dashboard_mod._operations_risk_level(
                tenant_alerts=0,
                drift_alerts=[],
                controls=(),
                compliance_exports=(),
                halts=1,
                false_positives=0,
            )
            == "monitored"
        )
        assert (
            dashboard_mod._operations_risk_level(
                tenant_alerts=0,
                drift_alerts=[],
                controls=(
                    TrustControl(
                        control="Evidence review",
                        status="warning",
                        evidence_ref="evidence.md",
                    ),
                ),
                compliance_exports=(),
                halts=0,
                false_positives=0,
            )
            == "attention_required"
        )
        assert (
            dashboard_mod._operations_risk_level(
                tenant_alerts=0,
                drift_alerts=[],
                controls=(),
                compliance_exports=(),
                halts=0,
                false_positives=0,
            )
            == "healthy"
        )

    def test_trust_risk_level_precedence(self) -> None:
        assert (
            dashboard_mod._trust_risk_level(
                tenants=[],
                controls=(
                    TrustControl(
                        control="Security",
                        status="warning",
                        evidence_ref="security.md",
                    ),
                ),
                halt_rate=0.0,
                false_positive_rate=0.0,
            )
            == "attention_required"
        )
        assert (
            dashboard_mod._trust_risk_level(
                tenants=[],
                controls=(),
                halt_rate=0.1,
                false_positive_rate=0.0,
            )
            == "monitored"
        )
        assert (
            dashboard_mod._trust_risk_level(
                tenants=[],
                controls=(),
                halt_rate=0.0,
                false_positive_rate=0.0,
            )
            == "healthy"
        )

    def test_source_and_nested_value_fallbacks(self) -> None:
        assert (
            dashboard_mod._contradiction_source(
                {"attributes": {"source": "kb://attributes"}}
            )
            == "kb://attributes"
        )
        assert (
            dashboard_mod._contradiction_source(
                {"attributes": {}, "trace_attribution": {"source": "kb://trace"}}
            )
            == "kb://trace"
        )
        assert (
            dashboard_mod._contradiction_source({"evidence_refs": ["kb://top-level"]})
            == "kb://top-level"
        )
        assert (
            dashboard_mod._contradiction_source(
                {
                    "trace_attribution": {},
                    "evidence_refs": ["kb://trace-fallback"],
                }
            )
            == "kb://trace-fallback"
        )
        assert (
            dashboard_mod._contradiction_source(
                {"halt_evidence": {"evidence_refs": ["kb://nested"]}}
            )
            == "kb://nested"
        )
        assert (
            dashboard_mod._contradiction_source(
                {"evidence_chunks": [{"id": "chunk-a"}]}
            )
            == "chunk-a"
        )
        assert (
            dashboard_mod._first_nested(
                {"attributes": {"tenant_id": "tenant-a"}},
                "tenant_id",
                default="default",
            )
            == "tenant-a"
        )
        assert (
            dashboard_mod._first_nested({"attributes": {}}, "tenant_id", default="x")
            == "x"
        )

    def test_truthy_accepts_numeric_and_string_forms(self) -> None:
        assert dashboard_mod._truthy(1) is True
        assert dashboard_mod._truthy(0) is False
        assert dashboard_mod._truthy("yes") is True
        assert dashboard_mod._truthy("no") is False
        assert dashboard_mod._truthy(object()) is False

    def test_feedback_label_string_contracts(self) -> None:
        assert dashboard_mod._feedback_label({"label": "accepted"}) is True
        assert dashboard_mod._feedback_label({"label": " approve "}) is True
        assert dashboard_mod._feedback_label({"label": "blocked"}) is False
        assert dashboard_mod._feedback_label({"label": "0"}) is False
        assert dashboard_mod._feedback_label({"label": "maybe"}) is None
        assert dashboard_mod._feedback_label({"label": []}) is None


class _FakeClick(TypedDict):
    """Captured Gradio click binding for UI wiring assertions."""

    component: _FakeComponent
    fn: Callable[..., object]
    inputs: list[_FakeComponent]
    outputs: list[_FakeComponent]


class _FakeComponent:
    """Minimal Gradio component double that records labels and callbacks."""

    def __init__(
        self,
        owner: _FakeGradio,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.owner = owner
        self.args = args
        self.kwargs = kwargs
        self.label = str(kwargs.get("label") or (args[0] if args else ""))
        owner.components.append(self)

    def click(
        self,
        *,
        fn: Callable[..., object],
        inputs: Sequence[_FakeComponent] | None = None,
        outputs: Sequence[_FakeComponent] | None = None,
    ) -> None:
        """Record the callback binding registered by the dashboard."""

        self.owner.clicks.append(
            {
                "component": self,
                "fn": fn,
                "inputs": list(inputs or []),
                "outputs": list(outputs or []),
            },
        )


class _FakeContext:
    """Minimal context-manager double for Gradio containers."""

    def __init__(self, owner: _FakeGradio, **kwargs: object) -> None:
        self.owner = owner
        self.kwargs = kwargs

    def __enter__(self) -> _FakeContext:
        """Enter a fake Gradio container context."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        """Propagate exceptions raised inside the fake context."""

        return False

    def launch(self, **kwargs: object) -> None:
        """Capture launch keyword arguments for assertions."""

        self.owner.launch_kwargs = dict(kwargs)


class _FakeGradioModule(ModuleType):
    """Module-shaped Gradio double with the attributes used by the UI."""

    Blocks: Callable[..., _FakeContext]
    Button: Callable[..., _FakeComponent]
    Code: Callable[..., _FakeComponent]
    Dataframe: Callable[..., _FakeComponent]
    Markdown: Callable[..., _FakeComponent]
    Row: Callable[..., _FakeContext]
    Slider: Callable[..., _FakeComponent]
    Textbox: Callable[..., _FakeComponent]


class _FakeGradio:
    """Container for fake Gradio components and click bindings."""

    def __init__(self) -> None:
        self.components: list[_FakeComponent] = []
        self.clicks: list[_FakeClick] = []
        self.launch_kwargs: dict[str, object] = {}
        self.module = _FakeGradioModule("gradio")
        self.module.Blocks = self._context
        self.module.Button = self._component
        self.module.Code = self._component
        self.module.Dataframe = self._component
        self.module.Markdown = self._component
        self.module.Row = self._context
        self.module.Slider = self._component
        self.module.Textbox = self._component

    def _context(self, *_args: object, **kwargs: object) -> _FakeContext:
        """Build a fake Gradio context object."""

        return _FakeContext(self, **kwargs)

    def _component(self, *args: object, **kwargs: object) -> _FakeComponent:
        """Build a fake Gradio component object."""

        return _FakeComponent(self, *args, **kwargs)
