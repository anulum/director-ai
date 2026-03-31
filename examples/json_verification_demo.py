# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""JSON verification demo — validate LLM-generated JSON against a schema."""

from director_ai import verify_json

# Schema for a customer order response
schema = {
    "type": "object",
    "required": ["order_id", "status", "items"],
    "properties": {
        "order_id": {"type": "string"},
        "status": {"type": "string"},
        "items": {"type": "array"},
        "total": {"type": "number"},
    },
    "additionalProperties": False,
}

# Valid response
good = (
    '{"order_id": "ORD-123", "status": "shipped", "items": ["widget"], "total": 29.99}'
)
result = verify_json(good, schema=schema)
print(f"Valid JSON: {result.valid_json}, Schema valid: {result.schema_valid}")

# Missing required field
bad = '{"order_id": "ORD-456", "status": "pending"}'
result = verify_json(bad, schema=schema)
print(f"\nMissing 'items': Schema valid: {result.schema_valid}")
for v in result.field_verdicts:
    if v.verdict != "valid":
        print(f"  {v.path}: {v.verdict} — {v.reason}")

# Wrong type
wrong_type = (
    '{"order_id": "ORD-789", "status": "shipped", "items": ["x"], "total": "expensive"}'
)
result = verify_json(wrong_type, schema=schema)
print(f"\nWrong type for 'total': Errors: {result.error_count}")
for v in result.field_verdicts:
    if v.verdict != "valid":
        print(f"  {v.path}: {v.verdict} — {v.reason}")

# Malformed JSON
malformed = '{"order_id": "ORD-000", status: broken}'
result = verify_json(malformed)
print(f"\nMalformed JSON: valid={result.valid_json}, error={result.parse_error}")
