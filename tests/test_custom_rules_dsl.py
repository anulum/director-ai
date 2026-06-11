# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — custom rules DSL tests

"""Strict JSON/YAML custom rule loading for enterprise policy operators."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from director_ai.enterprise import CustomRuleset, RulesDslError

pytestmark = pytest.mark.enterprise


def _ruleset_dict() -> dict:
    return {
        "version": "director.rules.v1",
        "name": "tenant-alpha.safety",
        "rules": [
            {
                "id": "no-passwords",
                "kind": "forbidden",
                "value": "share passwords",
                "name": "No password disclosure",
                "action": "block",
                "source": "security-baseline.md#L10",
            },
            {
                "id": "ticket-redaction",
                "kind": "pattern",
                "value": r"TICKET-\d{6}",
                "name": "Ticket reference",
                "action": "redact",
            },
            {
                "id": "length-cap",
                "kind": "max_length",
                "value": 80,
                "name": "Reply length cap",
            },
            {
                "id": "citations",
                "kind": "required_citations",
                "value": 2,
                "name": "Minimum citations",
            },
        ],
    }


def test_dict_ruleset_compiles_to_runtime_policy() -> None:
    ruleset = CustomRuleset.from_dict(_ruleset_dict())

    bundle = ruleset.to_policy_bundle(version=7)
    policy = ruleset.to_policy()

    assert bundle.version == 7
    assert [rule.id for rule in bundle.rules] == [
        "no-passwords",
        "ticket-redaction",
        "length-cap",
        "citations",
    ]
    assert policy.max_length == 80
    assert policy.required_citations_min == 2

    violations = policy.check("share passwords in TICKET-123456")

    assert {violation.rule for violation in violations} == {
        "forbidden",
        "pattern:Ticket reference",
        "required_citations",
    }
    assert next(
        v for v in violations if v.rule == "pattern:Ticket reference"
    ).detail == ("redact")


def test_json_round_trip_preserves_deterministic_schema() -> None:
    ruleset = CustomRuleset.from_json(json.dumps(_ruleset_dict()))

    assert ruleset.to_dict() == _ruleset_dict()
    assert ruleset.to_json() == json.dumps(_ruleset_dict(), sort_keys=True)


def test_yaml_file_loads_same_schema(tmp_path: Path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text(
        """
version: director.rules.v1
name: tenant-alpha.safety
rules:
  - id: no-passwords
    kind: forbidden
    value: share passwords
    name: No password disclosure
    action: block
""".strip(),
        encoding="utf-8",
    )

    ruleset = CustomRuleset.from_file(path)

    assert ruleset.name == "tenant-alpha.safety"
    assert ruleset.rules[0].id == "no-passwords"
    assert ruleset.to_policy().check("share passwords")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.update({"unexpected": True}), "unexpected"),
        (lambda data: data["rules"][0].update({"extra": "field"}), "extra"),
    ],
)
def test_unknown_fields_are_rejected(mutation, message: str) -> None:
    data = _ruleset_dict()
    mutation(data)

    with pytest.raises(RulesDslError, match=message):
        CustomRuleset.from_dict(data)


def test_duplicate_rule_ids_are_rejected() -> None:
    data = _ruleset_dict()
    data["rules"].append(dict(data["rules"][0]))

    with pytest.raises(RulesDslError, match="duplicate rule id"):
        CustomRuleset.from_dict(data)


def test_invalid_regex_is_rejected_before_runtime_policy() -> None:
    data = _ruleset_dict()
    data["rules"][1]["value"] = "[broken("

    with pytest.raises(RulesDslError, match="invalid regex"):
        CustomRuleset.from_dict(data)


@pytest.mark.parametrize("kind", ["max_length", "required_citations"])
def test_numeric_rule_values_must_be_positive_integers(kind: str) -> None:
    data = _ruleset_dict()
    numeric_rule = next(rule for rule in data["rules"] if rule["kind"] == kind)
    numeric_rule["value"] = 0

    with pytest.raises(RulesDslError, match="positive integer"):
        CustomRuleset.from_dict(data)


def test_threshold_must_be_probability() -> None:
    data = _ruleset_dict()
    data["rules"][0]["threshold"] = 1.5

    with pytest.raises(RulesDslError, match="less than or equal to 1"):
        CustomRuleset.from_dict(data)
