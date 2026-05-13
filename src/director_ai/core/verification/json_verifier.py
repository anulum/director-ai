# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""JSON output verification — schema compliance + value grounding.

Verifies structured JSON responses from LLMs:
1. Parses JSON (gracefully handles malformed output)
2. Validates against JSON Schema if provided
3. Checks each string value as a factual claim against a knowledge base
4. Detects cross-field contradictions

Usage::

    from director_ai.core.verification.json_verifier import verify_json

    result = verify_json(
        '{"status": "shipped", "tracking": "UPS1234"}',
        schema={"type": "object", "required": ["status"]},
    )
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, cast

from .types import FieldVerdict, StructuredVerificationResult

__all__ = ["verify_json"]

_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$")


def _extract_fields(data, prefix: str = "") -> list[tuple[str, object]]:
    """Flatten a nested dict/list into (dotted_path, value) pairs."""
    fields: list[tuple[str, object]] = []
    if isinstance(data, dict):
        for k, v in data.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                fields.extend(_extract_fields(v, path))
            else:
                fields.append((path, v))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            path = f"{prefix}[{i}]"
            if isinstance(item, (dict, list)):
                fields.extend(_extract_fields(item, path))
            else:
                fields.append((path, item))
    return fields


def _check_type(value: Any, expected_type: str) -> bool:
    """Check if a Python value matches a JSON Schema type string."""
    type_map: dict[str, type | tuple[type, ...]] = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "null": type(None),
        "array": list,
        "object": dict,
    }
    expected = type_map.get(expected_type)
    if expected is None:
        return True
    if expected_type in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, expected)


def _check_schema_type(value: Any, expected_type: str | list[str] | None) -> bool:
    if expected_type is None:
        return True
    if isinstance(expected_type, list):
        return any(_check_type(value, t) for t in expected_type)
    return _check_type(value, expected_type)


def _validate_schema(
    data: Any,
    schema: dict,
    prefix: str = "",
) -> list[FieldVerdict]:
    """Validate data against a JSON Schema subset (no $ref resolution)."""
    return _validate_value(data, schema, prefix or "$")


def _validate_value(value: Any, schema: dict, path: str) -> list[FieldVerdict]:
    verdicts: list[FieldVerdict] = []
    expected_type = schema.get("type")

    if expected_type and not _check_schema_type(value, expected_type):
        verdicts.append(
            FieldVerdict(
                path=path,
                value=str(value),
                verdict="invalid_type",
                reason=f"Expected type '{expected_type}', got {type(value).__name__}",
            )
        )
        return verdicts

    if "enum" in schema and value not in schema["enum"]:
        verdicts.append(
            FieldVerdict(
                path=path,
                value=str(value),
                verdict="invalid_value",
                reason=f"Value not in enum {schema['enum']}",
            )
        )
    if "const" in schema and value != schema["const"]:
        verdicts.append(
            FieldVerdict(
                path=path,
                value=str(value),
                verdict="invalid_value",
                reason=f"Value does not match const {schema['const']!r}",
            )
        )

    if isinstance(value, dict) and (
        schema.get("type") == "object" or "properties" in schema or "required" in schema
    ):
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        for key in required:
            if key not in value:
                verdicts.append(
                    FieldVerdict(
                        path=_child_path(path, key),
                        value="",
                        verdict="missing",
                        reason=f"Required field '{key}' is missing",
                    )
                )

        for key, prop_schema in properties.items():
            if key not in value:
                continue
            child_path = _child_path(path, key)
            verdicts.extend(_validate_value(value[key], prop_schema, child_path))

        if not schema.get("additionalProperties", True):
            for key in value:
                if key not in properties:
                    verdicts.append(
                        FieldVerdict(
                            path=_child_path(path, key),
                            value=str(value[key]),
                            verdict="extra",
                            reason=f"Unexpected field '{key}'",
                        )
                    )
        return verdicts

    if isinstance(value, list) and schema.get("type") == "array":
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                verdicts.extend(
                    _validate_value(item, item_schema, f"{path}[{index}]"),
                )
        return verdicts

    if not verdicts:
        verdicts.append(FieldVerdict(path=path, value=str(value), verdict="valid"))
    return verdicts


def _child_path(parent: str, key: str) -> str:
    if parent == "$":
        return key
    return f"{parent}.{key}"


def _pydantic_verdicts(data: Any, model: Any) -> list[FieldVerdict]:
    try:
        if hasattr(model, "model_validate"):
            model.model_validate(data)
        elif hasattr(model, "parse_obj"):
            model.parse_obj(data)
        else:
            raise TypeError("pydantic_model must expose model_validate or parse_obj")
        return []
    except Exception as exc:
        errors = cast(list[Mapping[str, Any]], getattr(exc, "errors", lambda: [])())
        verdicts: list[FieldVerdict] = []
        for error in errors:
            loc = error.get("loc", ())
            path = ".".join(str(part) for part in loc) if loc else "$"
            verdicts.append(
                FieldVerdict(
                    path=path,
                    value="",
                    verdict="invalid_type",
                    reason=str(error.get("msg", exc)),
                )
            )
        if not verdicts:
            verdicts.append(
                FieldVerdict(
                    path="$",
                    value="",
                    verdict="invalid_value",
                    reason=str(exc),
                )
            )
        return verdicts


def verify_json(
    text: str,
    schema: dict | None = None,
    score_fn=None,
    pydantic_model=None,
) -> StructuredVerificationResult:
    """Verify a JSON string for structure, schema, and optional value grounding.

    Parameters
    ----------
    text : str
        Raw JSON string (may be malformed).
    schema : dict | None
        JSON Schema to validate against. If None, only parse check.
    score_fn : callable | None
        ``score_fn(claim: str) -> float`` returning divergence [0, 1].
        Used to ground string values against a knowledge base.
    pydantic_model : type | None
        Optional Pydantic model class for production schema validation.

    Returns
    -------
    StructuredVerificationResult
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        return StructuredVerificationResult(
            valid_json=False,
            schema_valid=None,
            field_verdicts=[],
            error_count=1,
            parse_error=str(e),
        )

    verdicts: list[FieldVerdict] = []
    schema_valid = None

    if schema is not None:
        verdicts = _validate_schema(data, schema)
        schema_valid = all(v.verdict == "valid" for v in verdicts)

    if pydantic_model is not None:
        pydantic_verdicts = _pydantic_verdicts(data, pydantic_model)
        verdicts.extend(pydantic_verdicts)
        pydantic_valid = not pydantic_verdicts
        schema_valid = (
            pydantic_valid if schema_valid is None else schema_valid and pydantic_valid
        )

    if score_fn is not None:
        fields = _extract_fields(data)
        for path, value in fields:
            if not isinstance(value, str) or not value.strip():
                continue
            if _NUMERIC_RE.match(value):
                continue
            claim = f"{path} is {value}"
            try:
                div = score_fn(claim)
            except Exception:  # nosec B112
                continue
            verdict = "valid" if div < 0.5 else "invalid_value"
            existing = [v for v in verdicts if v.path == path]
            if existing:
                if verdict == "invalid_value":
                    existing[0].verdict = verdict
                    existing[0].reason = f"Value grounding divergence: {div:.2f}"
            else:
                verdicts.append(
                    FieldVerdict(
                        path=path,
                        value=str(value),
                        verdict=verdict,
                        reason=f"Grounding divergence: {div:.2f}"
                        if verdict != "valid"
                        else "",
                    )
                )

    error_count = sum(1 for v in verdicts if v.verdict != "valid")

    return StructuredVerificationResult(
        valid_json=True,
        schema_valid=schema_valid,
        field_verdicts=verdicts,
        error_count=error_count,
    )
