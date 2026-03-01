from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from director_ai.core.policy import Policy, Violation


class TestViolationDataclass:
    def test_fields(self):
        v = Violation(rule="forbidden", detail="bad phrase")
        assert v.rule == "forbidden"
        assert v.detail == "bad phrase"


class TestPolicyDefaults:
    def test_empty_policy_passes_everything(self):
        policy = Policy()
        assert policy.check("anything at all") == []

    def test_no_forbidden_no_block(self):
        policy = Policy(forbidden=[])
        assert policy.check("hello world") == []


class TestForbiddenPhrases:
    def test_exact_match(self):
        policy = Policy(forbidden=["as an AI language model"])
        violations = policy.check("As an AI language model, I cannot...")
        assert len(violations) == 1
        assert violations[0].rule == "forbidden"

    def test_case_insensitive(self):
        policy = Policy(forbidden=["forbidden phrase"])
        violations = policy.check("FORBIDDEN PHRASE detected")
        assert len(violations) == 1

    def test_multiple_forbidden(self):
        policy = Policy(forbidden=["bad", "worse"])
        violations = policy.check("This is bad and worse.")
        assert len(violations) == 2


class TestMaxLength:
    def test_under_limit(self):
        policy = Policy(max_length=100)
        assert policy.check("short") == []

    def test_over_limit(self):
        policy = Policy(max_length=10)
        violations = policy.check("a" * 11)
        assert len(violations) == 1
        assert violations[0].rule == "max_length"

    def test_zero_means_unlimited(self):
        policy = Policy(max_length=0)
        assert policy.check("a" * 100_000) == []


class TestRequiredCitations:
    def test_missing_citations(self):
        policy = Policy(
            required_citations_pattern=r"\[\d+\]",
            required_citations_min=1,
        )
        violations = policy.check("No citations here.")
        assert any(v.rule == "required_citations" for v in violations)

    def test_sufficient_citations(self):
        policy = Policy(
            required_citations_pattern=r"\[\d+\]",
            required_citations_min=2,
        )
        violations = policy.check("Source [1] and source [2] agree.")
        assert not any(v.rule == "required_citations" for v in violations)

    def test_disabled_when_min_zero(self):
        policy = Policy(
            required_citations_pattern=r"\[\d+\]",
            required_citations_min=0,
        )
        assert policy.check("No refs.") == []


class TestRegexValidation:
    def test_invalid_regex_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid regex.*broken"):
            Policy(
                patterns=[{"name": "broken", "regex": "[invalid(", "action": "block"}]
            )

    def test_valid_regex_compiles(self):
        policy = Policy(
            patterns=[{"name": "good", "regex": r"\btest\b", "action": "block"}]
        )
        assert len(policy._compiled_patterns) == 1


class TestRegexPatterns:
    def test_pattern_block(self):
        policy = Policy(
            patterns=[{"name": "profanity", "regex": r"\bdamn\b", "action": "block"}]
        )
        violations = policy.check("Well damn it")
        assert len(violations) == 1
        assert violations[0].rule == "pattern:profanity"

    def test_pattern_no_match(self):
        policy = Policy(
            patterns=[{"name": "profanity", "regex": r"\bdamn\b", "action": "block"}]
        )
        assert policy.check("Hello world") == []

    def test_empty_regex_ignored(self):
        policy = Policy(patterns=[{"name": "empty", "regex": "", "action": "block"}])
        assert policy.check("anything") == []


class TestFromDict:
    def test_round_trip(self):
        data = {
            "forbidden": ["bad phrase"],
            "patterns": [{"name": "test", "regex": r"\btest\b", "action": "warn"}],
            "style": {"max_length": 500},
            "required_citations": {"min_count": 1, "pattern": r"\[\d+\]"},
        }
        policy = Policy.from_dict(data)
        assert policy.forbidden == ["bad phrase"]
        assert policy.max_length == 500
        assert policy.required_citations_min == 1

    def test_empty_dict(self):
        policy = Policy.from_dict({})
        assert policy.forbidden == []
        assert policy.max_length == 0


class TestFromYaml:
    def test_json_fallback(self):
        data = {
            "forbidden": ["ignore previous instructions"],
            "style": {"max_length": 1000},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            f.flush()
            policy = Policy.from_yaml(f.name)
        assert policy.forbidden == ["ignore previous instructions"]
        assert policy.max_length == 1000
        Path(f.name).unlink()

    def test_invalid_content_returns_default(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write('"just a string"')
            f.flush()
            policy = Policy.from_yaml(f.name)
        assert policy.forbidden == []
        Path(f.name).unlink()

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            Policy.from_yaml("/tmp/nonexistent_policy_42.yaml")

    def test_malformed_content_raises(self):
        try:
            import yaml

            exc_type = (json.JSONDecodeError, yaml.YAMLError)
        except ImportError:
            exc_type = json.JSONDecodeError
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write("{not valid json")
            f.flush()
            path = f.name
        with pytest.raises(exc_type):
            Policy.from_yaml(path)
        Path(path).unlink()

    def test_list_content_returns_default(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(["a", "b"], f)
            f.flush()
            policy = Policy.from_yaml(f.name)
        assert policy.forbidden == []
        Path(f.name).unlink()


class TestPolicyCombined:
    def test_multiple_rules_all_fire(self):
        policy = Policy(
            forbidden=["bad"],
            max_length=5,
            patterns=[{"name": "digits", "regex": r"\d+", "action": "block"}],
        )
        violations = policy.check("bad 123 is very long text")
        rules = {v.rule for v in violations}
        assert "forbidden" in rules
        assert "max_length" in rules
        assert "pattern:digits" in rules
