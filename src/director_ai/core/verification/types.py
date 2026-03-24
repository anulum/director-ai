# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Result types for structured output verification."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CodeCheckResult",
    "FieldVerdict",
    "StructuredVerificationResult",
    "ToolCallResult",
]


@dataclass
class FieldVerdict:
    """Verification result for a single JSON field or tool argument."""

    path: str  # dot-separated path (e.g. "order.status")
    value: str
    verdict: str  # "valid", "invalid_type", "invalid_value", "missing", "extra"
    reason: str = ""


@dataclass
class StructuredVerificationResult:
    """Result of verifying a JSON output."""

    valid_json: bool
    schema_valid: bool | None  # None if no schema provided
    field_verdicts: list[FieldVerdict]
    error_count: int
    parse_error: str = ""


@dataclass
class ToolCallResult:
    """Result of verifying a tool/function call."""

    function_exists: bool
    arguments_valid: bool
    result_plausible: bool
    fabrication_suspected: bool
    verdicts: list[FieldVerdict]
    reason: str = ""


@dataclass
class CodeCheckResult:
    """Result of verifying generated code."""

    syntax_valid: bool
    unknown_imports: list[str]
    hallucinated_apis: list[str]
    error_count: int
    parse_error: str = ""
