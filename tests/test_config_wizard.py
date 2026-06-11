# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for ``director_ai.ui.config_wizard``.

Covers YAML generation, field formatting, overrides, completeness,
and edge cases. Gradio launch is not tested (requires UI dep).
"""

from __future__ import annotations

import json
import sys
import types

import yaml

from director_ai.ui.config_wizard import (
    _format_yaml_field,
    build_trace_explorer,
    calibration_feedback_jsonl,
    generate_profile_yaml,
    generate_yaml,
    launch_cli,
    launch_gradio,
    normalise_facts_text,
    profile_summary,
)

# ── YAML field formatting ──────────────────────────────────────────────


class TestFormatYamlField:
    def test_bool_true(self):
        assert _format_yaml_field("use_nli", True) == "use_nli: true"

    def test_bool_false(self):
        assert _format_yaml_field("use_nli", False) == "use_nli: false"

    def test_float(self):
        assert _format_yaml_field("threshold", 0.5) == "threshold: 0.5"

    def test_int(self):
        assert _format_yaml_field("max_steps", 50) == "max_steps: 50"

    def test_string(self):
        assert _format_yaml_field("backend", "deberta") == "backend: deberta"

    def test_empty_string(self):
        assert _format_yaml_field("model", "") == 'model: ""'

    def test_string_with_colon(self):
        result = _format_yaml_field("url", "http://localhost:8080")
        assert '"http://localhost:8080"' in result

    def test_none(self):
        result = _format_yaml_field("optional", None)
        assert result.startswith("# optional:")


# ── generate_yaml ──────────────────────────────────────────────────────


class TestGenerateYaml:
    def test_returns_string(self):
        result = generate_yaml()
        assert isinstance(result, str)
        assert len(result) > 100

    def test_has_header_comment(self):
        result = generate_yaml()
        assert "# Director-AI Configuration" in result

    def test_has_group_comments(self):
        result = generate_yaml()
        assert "# --- Scoring ---" in result

    def test_override_applied(self):
        result = generate_yaml({"coherence_threshold": 0.9})
        assert "coherence_threshold: 0.9" in result

    def test_override_bool(self):
        result = generate_yaml({"use_nli": False})
        assert "use_nli: false" in result

    def test_override_string(self):
        result = generate_yaml({"scorer_backend": "rules"})
        assert "scorer_backend: rules" in result

    def test_new_rag_fields_present(self):
        result = generate_yaml()
        assert "parent_child_enabled" in result
        assert "hyde_enabled" in result
        assert "multi_vector_enabled" in result

    def test_parseable_yaml(self):
        result = generate_yaml({"coherence_threshold": 0.7, "use_nli": True})
        # Remove comment lines and parse
        lines = [
            ln for ln in result.split("\n") if ln.strip() and not ln.startswith("#")
        ]
        parsed = yaml.safe_load("\n".join(lines))
        assert isinstance(parsed, dict)
        assert parsed["coherence_threshold"] == 0.7
        assert parsed["use_nli"] is True

    def test_empty_overrides(self):
        result = generate_yaml({})
        assert len(result) > 100

    def test_no_duplicate_fields(self):
        result = generate_yaml()
        lines = [
            ln.split(":")[0].strip()
            for ln in result.split("\n")
            if ":" in ln and not ln.startswith("#")
        ]
        assert len(lines) == len(set(lines)), (
            f"Duplicates: {[x for x in lines if lines.count(x) > 1]}"
        )


class TestProfileWizard:
    def test_profile_summary_includes_metadata(self):
        summary = profile_summary("medical")
        assert "**medical**" in summary
        assert "Required extras" in summary
        assert "nli" in summary

    def test_generate_profile_yaml_applies_profile_defaults(self):
        result = generate_profile_yaml(
            "customer_support",
            {"coherence_threshold": 0.52},
        )
        parsed = yaml.safe_load(
            "\n".join(
                ln for ln in result.split("\n") if ln.strip() and not ln.startswith("#")
            ),
        )
        assert parsed["profile"] == "customer_support"
        assert parsed["coherence_threshold"] == 0.52
        assert parsed["hard_limit"] == 0.4
        assert parsed["w_logic"] == 0.5

    def test_launch_gradio_wires_retune_action(self, monkeypatch):
        fake = _FakeGradio()
        monkeypatch.setitem(sys.modules, "gradio", fake.module)

        launch_gradio(port=7870, share=False)

        labels = [component.label for component in fake.components]
        assert "Tuned profile name" in labels
        assert "Base profile" in labels
        assert "Tuned profile overlay" in labels
        assert "Retune from Feedback" in labels
        assert fake.launch_kwargs == {"server_port": 7870, "share": False}

        retune_clicks = [
            click
            for click in fake.clicks
            if getattr(click["fn"], "__name__", "") == "build_retune_guidance"
        ]
        assert len(retune_clicks) == 1
        click = retune_clicks[0]
        assert [component.label for component in click["inputs"]] == [
            "Feedback JSONL",
            "Tuned profile name",
            "Base profile",
        ]
        assert [component.label for component in click["outputs"]] == [
            "",
            "Tuned profile overlay",
        ]

        config_clicks = [
            click
            for click in fake.clicks
            if click["component"].label == "Generate Config"
        ]
        assert len(config_clicks) == 1
        generated = config_clicks[0]["fn"](
            coherence_threshold=0.72,
            use_nli=None,
            scorer_backend="rules",
        )
        assert "coherence_threshold: 0.72" in generated
        assert "scorer_backend: rules" in generated

    def test_launch_gradio_reports_missing_dependency(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "gradio", None)

        try:
            launch_gradio()
        except ImportError as exc:
            assert "director-ai[ui]" in str(exc)
        else:
            raise AssertionError("launch_gradio should require Gradio")

    def test_normalise_facts_text(self):
        content, count = normalise_facts_text(" one fact \n\n second fact\n")
        assert content == "one fact\nsecond fact\n"
        assert count == 2

    def test_calibration_feedback_jsonl(self):
        row = calibration_feedback_jsonl(
            "prompt",
            "response",
            guardrail_approved=False,
            human_approved=True,
            domain="support",
        )
        parsed = yaml.safe_load(row)
        assert parsed["prompt"] == "prompt"
        assert parsed["guardrail_approved"] is False
        assert parsed["human_approved"] is True
        assert parsed["domain"] == "support"

    def test_launch_cli_accepts_defaults(self, monkeypatch, capsys):
        responses = iter([""] * 10)
        monkeypatch.setattr("builtins.input", lambda _prompt: next(responses))

        result = launch_cli()

        output = capsys.readouterr().out
        assert "Director-AI Configuration Wizard" in output
        assert "Generated configuration" in output
        assert "# Director-AI Configuration" in result
        assert "coherence_threshold: 0.6" in result
        assert "use_nli: false" in result

    def test_launch_cli_casts_entered_values(self, monkeypatch):
        responses = iter(
            [
                "0.73",
                "no",
                "rules",
                "false",
                "yes",
                "1",
                "true",
                "false",
                "y",
                "0",
            ],
        )
        monkeypatch.setattr("builtins.input", lambda _prompt: next(responses))

        result = launch_cli()

        parsed = yaml.safe_load(
            "\n".join(
                ln for ln in result.split("\n") if ln.strip() and not ln.startswith("#")
            ),
        )
        assert parsed["coherence_threshold"] == 0.73
        assert parsed["use_nli"] is False
        assert parsed["scorer_backend"] == "rules"
        assert parsed["hybrid_retrieval"] is False
        assert parsed["reranker_enabled"] is True
        assert parsed["parent_child_enabled"] is True
        assert parsed["adaptive_retrieval_enabled"] is True
        assert parsed["hyde_enabled"] is False
        assert parsed["injection_detection_enabled"] is True
        assert parsed["multi_vector_enabled"] is False


class TestTraceExplorer:
    def test_empty_trace_reports_operator_prompt(self):
        summary, rows, detail = build_trace_explorer("   ")

        assert "Paste a streaming" in summary
        assert rows == []
        assert detail == {"error": "empty input"}

    def test_scalar_and_list_traces_are_normalised(self):
        scalar_summary, scalar_rows, scalar_detail = build_trace_explorer('"token"')
        list_summary, list_rows, _list_detail = build_trace_explorer(
            json.dumps(["raw event", {"scope": "manual", "warning": True}]),
        )

        assert "Events: 1" in scalar_summary
        assert scalar_rows[0][1] == "trace"
        assert scalar_rows[0][7] == "token"
        assert scalar_detail["halted"] is False
        assert "Events: 2" in list_summary
        assert list_rows[0][2] == "value"
        assert list_rows[1][1] == "manual"
        assert list_rows[1][3] == "warning"

    def test_trace_root_event_and_nested_reason(self):
        payload = {
            "halted": False,
            "halt_reason": "",
            "halt_evidence": {"suggested_action": "review source"},
        }

        summary, rows, detail = build_trace_explorer(json.dumps(payload))

        assert "Events: 1" in summary
        assert rows[0][1] == "streaming"
        assert rows[0][6] == "review source"
        assert detail["halted"] is False

    def test_trace_root_attribution_and_counterfactual_defaults(self):
        payload = {
            "trace_attribution": {"token_offset": 3},
            "counterfactual_diagnostic": {"required_score_delta": 0.2},
            "events": [{"event_type": "audit"}],
        }

        summary, rows, detail = build_trace_explorer(json.dumps(payload))

        assert "unknown scorer at token 3" in summary
        assert "unknown fact needs delta 0.2" in summary
        assert rows[0][1] == "streaming"
        assert detail["trace_attribution"] == {"token_offset": 3}
        assert detail["counterfactual"] == {"required_score_delta": 0.2}

    def test_streaming_trace_halt_summary(self):
        payload = {
            "halted": True,
            "halt_reason": "hard_limit",
            "events": [
                {"index": 0, "token": "A", "coherence": 0.91},
                {
                    "index": 1,
                    "token": " claim",
                    "coherence": 0.31,
                    "halted": True,
                    "halt_reason": "hard_limit",
                    "halt_evidence": {
                        "trace_attribution": {
                            "token_offset": 1,
                            "retrieval_path": "vector",
                            "scorer_path": "factcg",
                        },
                    },
                },
            ],
        }

        summary, rows, detail = build_trace_explorer(json.dumps(payload))

        assert "Events: 2" in summary
        assert "Halted: yes" in summary
        assert "hard_limit" in summary
        assert rows[1][1] == "streaming"
        assert rows[1][3] == "halted"
        assert rows[1][4] == "0.310"
        assert "scorer=factcg" in rows[1][7]
        assert detail["halted"] is True

    def test_agent_and_swarm_trace_rows(self):
        payload = {
            "agent_events": [
                {
                    "event_type": "agent_policy",
                    "agent_id": "planner",
                    "decision": "passed",
                    "score": 0.82,
                },
            ],
            "swarm_events": [
                {
                    "event_type": "swarm_equilibrium",
                    "policy_decision": "halted",
                    "hook_id": "swarm_guard",
                    "reason": "unstable quorum",
                },
            ],
        }

        _summary, rows, detail = build_trace_explorer(json.dumps(payload))

        assert rows[0][1] == "agent"
        assert rows[1][1] == "swarm"
        assert rows[1][3] == "halted"
        assert rows[1][5] == "swarm_guard"
        assert detail["scopes"] == ["agent", "swarm"]

    def test_counterfactual_detail(self):
        payload = {
            "halt_evidence_structured": {
                "reason": "window_average",
                "last_score": 0.44,
                "trace_attribution": {
                    "fact_source": "kb://physics",
                    "retrieval_path": "hybrid",
                    "scorer_path": "factcg",
                    "token_offset": 12,
                },
                "counterfactual_diagnostic": {
                    "best_change": {
                        "fact_source": "kb://physics",
                        "required_score_delta": 0.08,
                    },
                },
            },
        }

        summary, rows, detail = build_trace_explorer(json.dumps(payload))

        assert "Counterfactual" in summary
        assert "kb://physics" in summary
        assert rows[0][1] == "streaming"
        assert "delta=0.08" in rows[0][7]
        assert detail["counterfactual"]["required_score_delta"] == 0.08

    def test_invalid_json_reports_position(self):
        summary, rows, detail = build_trace_explorer("{")

        assert "Invalid JSON" in summary
        assert rows == []
        assert detail["line"] == 1


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
                "inputs": _component_list(inputs),
                "outputs": _component_list(outputs),
            },
        )

    def change(self, *, fn, inputs=None, outputs=None):
        self.owner.changes.append(
            {
                "component": self,
                "fn": fn,
                "inputs": _component_list(inputs),
                "outputs": _component_list(outputs),
            },
        )


def _component_list(value):
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


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
        self.changes = []
        self.launch_kwargs = {}
        self.module = types.SimpleNamespace(
            Accordion=lambda *args, **kwargs: _FakeContext(self, **kwargs),
            Blocks=lambda *args, **kwargs: _FakeContext(self, **kwargs),
            Button=lambda *args, **kwargs: _FakeComponent(self, *args, **kwargs),
            Checkbox=lambda *args, **kwargs: _FakeComponent(self, *args, **kwargs),
            Code=lambda *args, **kwargs: _FakeComponent(self, *args, **kwargs),
            Dataframe=lambda *args, **kwargs: _FakeComponent(self, *args, **kwargs),
            Dropdown=lambda *args, **kwargs: _FakeComponent(self, *args, **kwargs),
            JSON=lambda *args, **kwargs: _FakeComponent(self, *args, **kwargs),
            Markdown=lambda *args, **kwargs: _FakeComponent(self, *args, **kwargs),
            Number=lambda *args, **kwargs: _FakeComponent(self, *args, **kwargs),
            Row=lambda *args, **kwargs: _FakeContext(self, **kwargs),
            Slider=lambda *args, **kwargs: _FakeComponent(self, *args, **kwargs),
            Tab=lambda *args, **kwargs: _FakeContext(self, **kwargs),
            Textbox=lambda *args, **kwargs: _FakeComponent(self, *args, **kwargs),
        )


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_unknown_override_ignored(self):
        # Unknown fields should not crash
        result = generate_yaml({"nonexistent_field_xyz": 42})
        assert isinstance(result, str)

    def test_all_fields_covered(self):
        from dataclasses import fields as dc_fields

        from director_ai.core.config import DirectorConfig

        result = generate_yaml()
        config_fields = {f.name for f in dc_fields(DirectorConfig)}
        yaml_fields = set()
        for ln in result.split("\n"):
            if ":" in ln and not ln.startswith("#"):
                yaml_fields.add(ln.split(":")[0].strip())
        # Most fields should appear (some may be commented out)
        covered = yaml_fields & config_fields
        assert len(covered) > 30, f"Only {len(covered)} fields covered"
