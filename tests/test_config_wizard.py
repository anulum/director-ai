# SPDX-License-Identifier: AGPL-3.0-or-later
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

import yaml

from director_ai.ui.config_wizard import (
    _format_yaml_field,
    calibration_feedback_jsonl,
    generate_profile_yaml,
    generate_yaml,
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
