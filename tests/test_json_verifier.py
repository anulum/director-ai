# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for JSON output verification pipeline.

Covers: schema validation, required fields, type checking, nested objects,
array validation, null handling, pipeline integration with CoherenceScorer
tool call verification, and performance documentation.
"""

from __future__ import annotations

import pytest

from director_ai.core.verification import json_verifier
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

    def test_boolean_not_number(self):
        schema = {"type": "object", "properties": {"score": {"type": "number"}}}
        r = verify_json('{"score": true}', schema=schema)
        assert any(v.verdict == "invalid_type" for v in r.field_verdicts)

    def test_array_root_with_schema(self):
        schema = {"type": "array"}
        r = verify_json("[1, 2, 3]", schema=schema)
        assert r.valid_json is True
        assert r.schema_valid is True

    def test_array_root_wrong_type(self):
        schema = {"type": "object"}
        r = verify_json("[1, 2, 3]", schema=schema)
        assert r.schema_valid is False
        assert any(v.verdict == "invalid_type" for v in r.field_verdicts)

    def test_string_root_with_schema(self):
        schema = {"type": "string"}
        r = verify_json('"hello world"', schema=schema)
        assert r.valid_json is True
        assert r.schema_valid is True

    def test_number_root_with_schema(self):
        schema = {"type": "number"}
        r = verify_json("42.5", schema=schema)
        assert r.valid_json is True
        assert r.schema_valid is True

    def test_permissive_empty_schema(self):
        r = verify_json('{"a": 1}', schema={})
        assert r.schema_valid is True

    def test_properties_only_schema(self):
        schema = {"properties": {"a": {"type": "number"}}}
        r = verify_json('{"a": 1}', schema=schema)
        assert r.schema_valid is True

    def test_enum_schema_valid(self):
        schema = {"enum": [42, "hello"]}
        r = verify_json("42", schema=schema)
        assert r.schema_valid is True

    def test_enum_schema_invalid(self):
        schema = {"enum": [42]}
        r = verify_json("43", schema=schema)
        assert r.schema_valid is False

    def test_const_schema_valid(self):
        schema = {"const": "ok"}
        r = verify_json('"ok"', schema=schema)
        assert r.schema_valid is True

    def test_const_schema_invalid(self):
        schema = {"const": "ok"}
        r = verify_json('"nope"', schema=schema)
        assert r.schema_valid is False

    def test_unknown_schema_type_is_treated_as_permissive_extension(self):
        schema = {"type": "vendor-extension"}

        r = verify_json('"opaque"', schema=schema)

        assert r.valid_json is True
        assert r.schema_valid is True

    def test_missing_schema_type_is_permissive(self):
        schema = {"enum": ["ok", "pending"]}

        r = verify_json('"ok"', schema=schema)

        assert r.schema_valid is True

    def test_explicit_null_schema_type_is_permissive(self):
        schema = {"type": None}

        r = verify_json('{"status": "approved"}', schema=schema)

        assert r.schema_valid is True
        assert json_verifier._check_schema_type({"status": "approved"}, None) is True

    def test_union_schema_type_accepts_any_matching_type(self):
        schema = {"type": ["string", "number"]}

        valid = verify_json('"ready"', schema=schema)
        invalid = verify_json("false", schema=schema)

        assert valid.schema_valid is True
        assert invalid.schema_valid is False

    def test_array_items_are_validated_recursively(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["sku", "quantity"],
                        "properties": {
                            "sku": {"type": "string"},
                            "quantity": {"type": "integer"},
                        },
                        "additionalProperties": False,
                    },
                }
            },
        }

        r = verify_json(
            '{"items": [{"sku": "A-1", "quantity": 2}, {"sku": "B-2", "quantity": "two", "extra": true}]}',
            schema=schema,
        )

        assert r.schema_valid is False
        assert any(v.path == "items[1].quantity" for v in r.field_verdicts)
        assert any(v.path == "items[1].extra" for v in r.field_verdicts)

    def test_nested_enum_and_const_are_validated(self):
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["approved", "rejected"]},
                "schema_version": {"const": "v1"},
            },
        }

        r = verify_json(
            '{"status": "pending", "schema_version": "v2"}',
            schema=schema,
        )

        assert r.schema_valid is False
        invalid = {v.path: v.verdict for v in r.field_verdicts}
        assert invalid["status"] == "invalid_value"
        assert invalid["schema_version"] == "invalid_value"

    def test_pydantic_model_validation_reports_field_path(self):
        pydantic = pytest.importorskip("pydantic")

        class Payload(pydantic.BaseModel):
            status: str
            quantity: int

        r = verify_json(
            '{"status": "approved", "quantity": "two"}',
            pydantic_model=Payload,
        )

        assert r.schema_valid is False
        assert any(
            v.path == "quantity" and v.verdict == "invalid_type"
            for v in r.field_verdicts
        )

    def test_pydantic_v1_parse_obj_success_is_accepted(self):
        class LegacyModel:
            @classmethod
            def parse_obj(cls, data):
                assert data == {"status": "approved"}

        r = verify_json('{"status": "approved"}', pydantic_model=LegacyModel)

        assert r.schema_valid is True
        assert r.error_count == 0

    def test_invalid_pydantic_adapter_reports_root_error(self):
        class InvalidAdapter:
            pass

        r = verify_json('{"status": "approved"}', pydantic_model=InvalidAdapter)

        assert r.schema_valid is False
        assert any(
            v.path == "$" and v.verdict == "invalid_value" for v in r.field_verdicts
        )


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

    def test_nested_object_and_array_values_are_grounded_by_path(self):
        claims = []

        def score_fn(claim):
            claims.append(claim)
            return 0.8 if "items[2]" in claim else 0.1

        r = verify_json(
            '{"order": {"status": "shipped"}, "items": ["book", ["nested item"], "unknown item"]}',
            score_fn=score_fn,
        )

        assert claims == [
            "order.status is shipped",
            "items[0] is book",
            "items[1][0] is nested item",
            "items[2] is unknown item",
        ]
        assert any(
            v.path == "items[2]" and v.verdict == "invalid_value"
            for v in r.field_verdicts
        )

    def test_empty_root_array_has_no_fields_to_ground(self):
        r = verify_json("[]", score_fn=lambda claim: 0.9)

        assert r.valid_json is True
        assert r.error_count == 0

    def test_scalar_root_has_no_nested_fields_to_ground(self):
        r = verify_json('"plain scalar"', score_fn=lambda claim: 0.9)

        assert r.valid_json is True
        assert r.error_count == 0

    def test_score_function_failure_leaves_field_without_grounding_error(self):
        def score_fn(claim):
            raise RuntimeError("grounding service unavailable")

        r = verify_json('{"status": "pending"}', score_fn=score_fn)

        assert r.valid_json is True
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

    def test_schema_field_remains_valid_when_grounding_accepts_existing_verdict(self):
        schema = {
            "type": "object",
            "properties": {"status": {"type": "string"}},
        }

        r = verify_json(
            '{"status": "shipped"}',
            schema=schema,
            score_fn=lambda claim: 0.1,
        )

        assert r.schema_valid is True
        assert r.error_count == 0
        assert r.field_verdicts[0].verdict == "valid"


class TestAcceleratedCounting:
    def test_sum_uses_python_fallback_when_rust_path_disabled(self, monkeypatch):
        monkeypatch.setattr(json_verifier, "_RUST_JSON_VERIFY", False)

        assert json_verifier._sum_int([1, 0, 1]) == 2


def test_verify_json_returns_the_structured_verification_contract():
    from director_ai.core.verification.types import (
        FieldVerdict,
        StructuredVerificationResult,
    )

    result = verify_json(
        '{"status": "shipped"}',
        schema={"type": "object", "properties": {"status": {"type": "string"}}},
    )

    assert isinstance(result, StructuredVerificationResult)
    assert result.valid_json is True
    assert result.schema_valid is True
    assert result.error_count == 0
    assert result.parse_error == ""
    assert all(isinstance(v, FieldVerdict) for v in result.field_verdicts)

    malformed = verify_json('{"status": shipped}')
    assert isinstance(malformed, StructuredVerificationResult)
    assert malformed.valid_json is False
    assert malformed.parse_error != ""
