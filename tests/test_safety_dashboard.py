# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - safety dashboard tests

from __future__ import annotations

import json
import sys
import types

from director_ai.ui.safety_dashboard import (
    EVIDENCE_COLUMNS,
    SOURCE_COLUMNS,
    TENANT_COLUMNS,
    build_retune_guidance,
    build_safety_dashboard,
    launch_safety_dashboard,
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

    def test_retune_guidance_builds_profile_overlay_from_feedback(self):
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

    def test_retune_guidance_requires_labelled_prompt_response_rows(self):
        summary, overlay = build_retune_guidance(
            _line({"event_id": "missing-fields", "human_approved": True}),
            min_samples=2,
        )

        assert "Required samples: 2" in summary
        assert "Parse warnings" in summary
        assert overlay == ""

    def test_retune_guidance_defaults_profile_from_base_profile(self):
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

    def test_retune_guidance_reports_provisional_single_class_feedback(self):
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

    def test_retune_guidance_keeps_parse_warnings_with_enough_valid_rows(self):
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

    def test_retune_guidance_accepts_boolean_and_numeric_labels(self):
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

    def test_retune_guidance_rejects_unknown_label_text(self):
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

    def test_retune_guidance_accepts_label_synonyms(self):
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

    def test_dashboard_reports_non_object_feedback_and_blank_lines(self):
        summary, tenants, sources, evidence, _command = build_safety_dashboard(
            "\n" + _line({"tenant_id": "tenant-a", "decision": "allow"}),
            "[]\n",
        )

        assert "feedback:1: expected object" in summary
        assert tenants == [["tenant-a", 1, 0, 0.0, 0, 0.0, "ok"]]
        assert sources == []
        assert evidence == []

    def test_dashboard_extracts_nested_sources_and_resilient_scores(self):
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

    def test_launch_safety_dashboard_reports_missing_dependency(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "gradio", None)

        try:
            launch_safety_dashboard()
        except ImportError as exc:
            assert "director-ai[ui]" in str(exc)
        else:
            raise AssertionError("launch_safety_dashboard should require Gradio")

    def test_launch_safety_dashboard_wires_retune_command(self, monkeypatch):
        fake = _FakeGradio()
        monkeypatch.setitem(sys.modules, "gradio", fake.module)

        launch_safety_dashboard(port=7871, share=True)

        labels = [component.label for component in fake.components]
        assert "SafetyEvent JSONL" in labels
        assert "Feedback JSONL" in labels
        assert "Retune command" in labels
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


class _FakeComponent:
    def __init__(self, owner, *args, **kwargs):
        self.owner = owner
        self.args = args
        self.kwargs = kwargs
        self.label = str(kwargs.get("label") or (args[0] if args else ""))
        owner.components.append(self)

    def click(self, *, fn, inputs=None, outputs=None):
        self.owner.clicks.append(
            {
                "component": self,
                "fn": fn,
                "inputs": list(inputs or []),
                "outputs": list(outputs or []),
            },
        )


class _FakeContext:
    def __init__(self, owner, **kwargs):
        self.owner = owner
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def launch(self, **kwargs):
        self.owner.launch_kwargs = kwargs


class _FakeGradio:
    def __init__(self):
        self.components = []
        self.clicks = []
        self.launch_kwargs = {}
        self.module = types.SimpleNamespace(
            Blocks=lambda *args, **kwargs: _FakeContext(self, **kwargs),
            Button=lambda *args, **kwargs: _FakeComponent(self, *args, **kwargs),
            Code=lambda *args, **kwargs: _FakeComponent(self, *args, **kwargs),
            Dataframe=lambda *args, **kwargs: _FakeComponent(self, *args, **kwargs),
            Markdown=lambda *args, **kwargs: _FakeComponent(self, *args, **kwargs),
            Row=lambda *args, **kwargs: _FakeContext(self, **kwargs),
            Slider=lambda *args, **kwargs: _FakeComponent(self, *args, **kwargs),
            Textbox=lambda *args, **kwargs: _FakeComponent(self, *args, **kwargs),
        )
