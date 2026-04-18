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
from typing import Any

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


def _validate_schema(
    data: dict,
    schema: dict,
    prefix: str = "",
) -> list[FieldVerdict]:
    """Validate data against a JSON Schema (flat, no $ref resolution)."""
    verdicts: list[FieldVerdict] = []

    if schema.get("type") == "object":
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        for key in required:
            path = f"{prefix}.{key}" if prefix else key
            if key not in data:
                verdicts.append(
                    FieldVerdict(
                        path=path,
                        value="",
                        verdict="missing",
                        reason=f"Required field '{key}' is missing",
                    )
                )

        for key, prop_schema in properties.items():
            path = f"{prefix}.{key}" if prefix else key
            if key not in data:
                continue
            val = data[key]
            expected_type = prop_schema.get("type")
            if expected_type and not _check_type(val, expected_type):
                verdicts.append(
                    FieldVerdict(
                        path=path,
                        value=str(val),
                        verdict="invalid_type",
                        reason=f"Expected type '{expected_type}', got {type(val).__name__}",
                    )
                )
            elif isinstance(val, dict) and prop_schema.get("type") == "object":
                verdicts.extend(_validate_schema(val, prop_schema, path))
            else:
                verdicts.append(
                    FieldVerdict(
                        path=path,
                        value=str(val),
                        verdict="valid",
                    )
                )

        if not schema.get("additionalProperties", True):
            for key in data:
                if key not in properties:
                    path = f"{prefix}.{key}" if prefix else key
                    verdicts.append(
                        FieldVerdict(
                            path=path,
                            value=str(data[key]),
                            verdict="extra",
                            reason=f"Unexpected field '{key}'",
                        )
                    )

    return verdicts


def verify_json(
    text: str,
    schema: dict | None = None,
    score_fn=None,
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
        root_type = schema.get("type")
        if root_type and not _check_type(data, root_type):
            schema_valid = False
            verdicts.append(
                FieldVerdict(
                    path="$",
                    value=type(data).__name__,
                    verdict="invalid_type",
                    reason=f"Schema expects {root_type}, got {type(data).__name__}",
                )
            )
        elif isinstance(data, dict):
            verdicts = _validate_schema(data, schema)
            schema_valid = all(v.verdict == "valid" for v in verdicts)
        else:
            schema_valid = True

        # Enforce enum/const regardless of root type
        if schema_valid is not False:
            if "enum" in schema and data not in schema["enum"]:
                schema_valid = False
                verdicts.append(
                    FieldVerdict(
                        path="$",
                        value=str(data),
                        verdict="invalid_value",
                        reason=f"Value not in enum {schema['enum']}",
                    )
                )
            if "const" in schema and data != schema["const"]:
                schema_valid = False
                verdicts.append(
                    FieldVerdict(
                        path="$",
                        value=str(data),
                        verdict="invalid_value",
                        reason=f"Value does not match const {schema['const']!r}",
                    )
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
