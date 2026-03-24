# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for JSON output verification."""

from __future__ import annotations

from director_ai.core.verification.json_verifier import verify_json


class TestParseValidation:
    def test_valid_json(self):
        r = verify_json('{"key": "value"}')
        assert r.valid_json is True
        assert r.error_count == 0

    def test_invalid_json(self):
        r = verify_json('{"key": value}')
        assert r.valid_json is False
        assert r.parse_error != ""

    def test_empty_string(self):
        r = verify_json("")
        assert r.valid_json is False

    def test_json_array(self):
        r = verify_json("[1, 2, 3]")
        assert r.valid_json is True

    def test_nested_json(self):
        r = verify_json('{"a": {"b": {"c": 1}}}')
        assert r.valid_json is True


class TestSchemaValidation:
    def test_required_field_present(self):
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }
        r = verify_json('{"name": "Alice"}', schema=schema)
        assert r.schema_valid is True

    def test_required_field_missing(self):
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }
        r = verify_json('{"age": 30}', schema=schema)
        assert r.schema_valid is False
        assert any(v.verdict == "missing" for v in r.field_verdicts)

    def test_wrong_type(self):
        schema = {"type": "object", "properties": {"age": {"type": "integer"}}}
        r = verify_json('{"age": "thirty"}', schema=schema)
        assert any(v.verdict == "invalid_type" for v in r.field_verdicts)

    def test_extra_field_not_allowed(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }
        r = verify_json('{"name": "Alice", "extra": true}', schema=schema)
        assert any(v.verdict == "extra" for v in r.field_verdicts)

    def test_no_schema(self):
        r = verify_json('{"a": 1}')
        assert r.schema_valid is None

    def test_nested_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                }
            },
        }
        r = verify_json('{"address": {"city": "Prague"}}', schema=schema)
        assert r.schema_valid is True

    def test_boolean_not_integer(self):
        schema = {"type": "object", "properties": {"count": {"type": "integer"}}}
        r = verify_json('{"count": true}', schema=schema)
        assert any(v.verdict == "invalid_type" for v in r.field_verdicts)


class TestValueGrounding:
    def test_grounded_value(self):
        r = verify_json(
            '{"status": "shipped"}',
            score_fn=lambda claim: 0.1,
        )
        assert r.error_count == 0

    def test_ungrounded_value(self):
        r = verify_json(
            '{"status": "shipped"}',
            score_fn=lambda claim: 0.8,
        )
        assert r.error_count == 1
        assert any(v.verdict == "invalid_value" for v in r.field_verdicts)

    def test_numeric_values_skipped(self):
        r = verify_json(
            '{"count": "42"}',
            score_fn=lambda claim: 0.9,
        )
        # "42" matches numeric regex, should be skipped
        assert r.error_count == 0

    def test_empty_string_skipped(self):
        r = verify_json(
            '{"note": ""}',
            score_fn=lambda claim: 0.9,
        )
        assert r.error_count == 0


class TestCombined:
    def test_schema_plus_grounding(self):
        schema = {
            "type": "object",
            "required": ["status"],
            "properties": {"status": {"type": "string"}},
        }
        r = verify_json(
            '{"status": "cancelled"}',
            schema=schema,
            score_fn=lambda claim: 0.7,
        )
        assert r.valid_json is True
        # Schema valid but grounding fails
        status_verdicts = [v for v in r.field_verdicts if v.path == "status"]
        assert len(status_verdicts) == 1
        assert status_verdicts[0].verdict == "invalid_value"
