# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Rails-as-Config Loader Tests

"""Multi-angle tests for the NeMo Guardrails rails-as-config loader.

Covers the honest-subset contract: topical refusal rails become
enforced forbidden phrases, recognised self-check rails enable the
moderation detectors, and every unmapped construct is reported in
``unsupported`` — never silently dropped.
"""

import pytest

from director_ai.integrations.rails_config import (
    RailsLoadResult,
    load_rails_config,
)

TOPICAL_COLANG = """
define user ask_about_politics
  "who should I vote for"
  "what do you think about the election"

define user ask_weather
  "what is the weather like"

define flow politics
  user ask_about_politics
  bot refuse to respond

define flow weather
  user ask_weather
  bot respond with forecast
"""

RICH_COLANG = """
define bot refuse to respond
  "I cannot help with that."

define flow gadgets
  user ask_gadgets
  execute lookup_gadgets
  if $found
    bot present gadgets

define user ask_gadgets
  "recommend a gadget"
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestColangTopicalRails:
    def test_refusal_flow_phrases_become_forbidden(self, tmp_path):
        result = load_rails_config(_write(tmp_path, "rails.co", TOPICAL_COLANG))
        assert isinstance(result, RailsLoadResult)
        assert result.source_format == "colang"
        assert result.forbidden_from_intents == {
            "ask_about_politics": (
                "who should I vote for",
                "what do you think about the election",
            ),
        }
        violations = result.policy.check("Honestly, who should I vote for?")
        assert any(v.rule == "forbidden" for v in violations)
        assert result.policy.check("Tell me about photosynthesis.") == []

    def test_non_refusal_flow_is_reported_not_mapped(self, tmp_path):
        result = load_rails_config(_write(tmp_path, "rails.co", TOPICAL_COLANG))
        assert "ask_weather" not in result.forbidden_from_intents
        assert any(
            "flow weather" in item and "ask_weather" in item
            for item in result.unsupported
        )

    def test_unmapped_constructs_are_itemised(self, tmp_path):
        result = load_rails_config(_write(tmp_path, "rails.co", RICH_COLANG))
        assert any(item.startswith("define bot") for item in result.unsupported)
        assert any("execute lookup_gadgets" in item for item in result.unsupported)
        assert any("if $found" in item for item in result.unsupported)
        # The gadgets flow has no refusal, so its intent is not enforced.
        assert result.forbidden_from_intents == {}
        assert result.policy.forbidden == []

    def test_refusal_intent_without_examples_is_reported(self, tmp_path):
        source = "define flow blocked\n  user ask_mystery\n  bot refuse to respond\n"
        result = load_rails_config(_write(tmp_path, "rails.co", source))
        assert result.forbidden_from_intents == {}
        assert any("refusal intent ask_mystery" in item for item in result.unsupported)

    def test_comments_blanks_and_bad_toplevel_lines(self, tmp_path):
        source = (
            "# a comment\n"
            "\n"
            "stray line\n"
            "define user ask_x\n"
            '  "x question"\n'
            "  not a quoted phrase\n"
        )
        result = load_rails_config(_write(tmp_path, "rails.co", source))
        assert "stray line" in result.unsupported
        assert any("not a quoted phrase" in item for item in result.unsupported)

    def test_indented_line_before_any_define_is_reported(self, tmp_path):
        source = '  "floating utterance"\ndefine user ask_y\n  "y question"\n'
        result = load_rails_config(_write(tmp_path, "rails.co", source))
        assert '"floating utterance"' in result.unsupported

    def test_flow_with_no_intents_and_no_refusal(self, tmp_path):
        source = "define flow maintenance\n  execute rotate_logs\n"
        result = load_rails_config(_write(tmp_path, "rails.co", source))
        assert result.forbidden_from_intents == {}
        assert any("execute rotate_logs" in item for item in result.unsupported)
        assert not any("not mapped" in item for item in result.unsupported)


class TestNemoConfig:
    def test_self_check_rails_enable_moderation(self, tmp_path):
        config = (
            "rails:\n"
            "  input:\n"
            "    flows:\n"
            "      - self check input\n"
            "  output:\n"
            "    flows:\n"
            "      - self check output\n"
        )
        result = load_rails_config(_write(tmp_path, "config.yml", config))
        assert result.source_format == "nemo-config"
        assert result.moderation_enabled is True
        assert result.notes and "moderation detectors" in result.notes[0]
        violations = result.policy.check("Contact me at jane@example.com")
        assert any(v.rule.startswith("moderation:") for v in violations)

    def test_unrecognised_flows_and_keys_are_reported(self, tmp_path):
        config = (
            "models:\n"
            "  - type: main\n"
            "rails:\n"
            "  dialog:\n"
            "    flows: []\n"
            "  input:\n"
            "    flows:\n"
            "      - custom jailbreak check\n"
        )
        result = load_rails_config(_write(tmp_path, "config.yml", config))
        assert result.moderation_enabled is False
        assert "config key: models" in result.unsupported
        assert "rails.dialog" in result.unsupported
        assert any("custom jailbreak check" in item for item in result.unsupported)

    def test_config_without_rails_block(self, tmp_path):
        result = load_rails_config(_write(tmp_path, "config.yml", "models: []\n"))
        assert result.moderation_enabled is False
        assert result.policy.forbidden == []
        assert "config key: models" in result.unsupported

    def test_malformed_flow_and_direction_shapes(self, tmp_path):
        config = "rails:\n  input:\n    flows: not-a-list\n  output: just-a-string\n"
        result = load_rails_config(_write(tmp_path, "config.yml", config))
        assert "rails.input.flows: not a list" in result.unsupported
        assert result.moderation_enabled is False

    def test_non_mapping_config_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="mapping"):
            load_rails_config(_write(tmp_path, "config.yml", "- a\n- b\n"))


class TestConfigDirectory:
    def test_directory_merges_config_and_colang(self, tmp_path):
        _write(
            tmp_path,
            "config.yml",
            "rails:\n  input:\n    flows:\n      - self check input\n",
        )
        _write(tmp_path, "topics.co", TOPICAL_COLANG)
        result = load_rails_config(tmp_path)
        assert result.source_format == "nemo-config+colang"
        assert result.moderation_enabled is True
        assert "ask_about_politics" in result.forbidden_from_intents
        summary = result.to_dict()
        assert summary["source_format"] == "nemo-config+colang"
        assert summary["forbidden_phrases"] == 2
        assert summary["moderation_enabled"] is True

    def test_directory_with_only_colang(self, tmp_path):
        _write(tmp_path, "topics.co", TOPICAL_COLANG)
        result = load_rails_config(tmp_path)
        assert result.source_format == "colang"

    def test_config_yaml_extension_variant(self, tmp_path):
        _write(tmp_path, "config.yaml", "rails: {}\n")
        result = load_rails_config(tmp_path)
        assert result.source_format == "nemo-config"

    def test_empty_directory_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="no config.yml"):
            load_rails_config(tmp_path)


class TestDispatchAndErrors:
    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_rails_config(tmp_path / "absent.co")

    @pytest.mark.parametrize("name", ["spec.rail", "spec.xml"])
    def test_rail_xml_points_to_native_integration(self, tmp_path, name):
        with pytest.raises(ValueError, match="guardrails_ai"):
            load_rails_config(_write(tmp_path, name, "<rail/>"))

    def test_unknown_extension_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unsupported rails configuration"):
            load_rails_config(_write(tmp_path, "rails.txt", "x"))

    def test_duplicate_unsupported_entries_are_deduplicated(self, tmp_path):
        source = "stray line\nstray line\n"
        result = load_rails_config(_write(tmp_path, "rails.co", source))
        assert result.unsupported.count("stray line") == 1
