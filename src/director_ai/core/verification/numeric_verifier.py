# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Numeric and quantitative verification for LLM outputs.

Extracts numeric claims from text and checks internal consistency:
- Percentage arithmetic: "grew 15% from $10M" → verify the math
- Date logic: birth < death, founding < present
- Order of magnitude: Earth population ≠ 80 billion
- Cross-reference: numbers in different sentences that should agree
- Unit sanity: negative counts, probabilities > 100%

Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

__all__ = [
    "NumericClaim",
    "NumericIssue",
    "NumericVerificationResult",
    "verify_numeric",
]

_PERCENT_PATTERN = re.compile(
    r"(?:grew|increased|decreased|dropped|rose|fell|declined|changed|gained|lost)"
    r"\s+(?:by\s+)?(\d{1,10}(?:\.\d{1,10})?)\s*%"
    r".{0,80}?\b(?:from|of)\s+\$?([\d,]{1,20}(?:\.\d{1,10})?)\s*(?:million|billion|thousand|[MBKmk])?"
    r".{0,80}?\bto\s+\$?([\d,]{1,20}(?:\.\d{1,10})?)\s*(?:million|billion|thousand|[MBKmk])?",
    re.IGNORECASE,
)

_NUMBER_PATTERN = re.compile(
    r"(?<!\w)(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(million|billion|trillion|thousand|percent|%|km|mi|kg|lb|m|ft)?"
    r"(?!\w)",
    re.IGNORECASE,
)

_DATE_PATTERN = re.compile(
    r"\b((?:1[0-9]|20)\d{2})\b",
)

_PROB_PATTERN = re.compile(
    r"(-?\d{1,10}(?:\.\d{1,10})?)\s*(?:%|percent)\s+(?:probability|chance|likelihood|confidence)",
    re.IGNORECASE,
)


def _parse_number(s: str) -> float:
    return float(s.replace(",", ""))


@dataclass
class NumericClaim:
    """A numeric value extracted from text."""

    value: float
    unit: str
    context: str  # surrounding text
    position: int  # character offset


@dataclass
class NumericIssue:
    """A detected numeric inconsistency."""

    issue_type: (
        str  # "arithmetic", "date_logic", "magnitude", "probability", "internal"
    )
    description: str
    severity: str  # "error" or "warning"
    context: str


@dataclass
class NumericVerificationResult:
    """Result of numeric verification on a text."""

    claims_found: int
    issues: list[NumericIssue] = field(default_factory=list)
    valid: bool = True

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")


def verify_numeric(text: str) -> NumericVerificationResult:
    """Verify numeric consistency within a text.

    Checks percentage arithmetic, date logic, probability bounds,
    and internal number consistency. Returns issues found.
    """
    issues: list[NumericIssue] = []
    claims_found = 0

    # 1. Percentage arithmetic
    for m in _PERCENT_PATTERN.finditer(text):
        claims_found += 1
        pct = _parse_number(m.group(1))
        val_from = _parse_number(m.group(2))
        val_to = _parse_number(m.group(3))
        if val_from > 0:
            actual_pct = abs(val_to - val_from) / val_from * 100
            if abs(actual_pct - pct) > 1.0:
                issues.append(
                    NumericIssue(
                        issue_type="arithmetic",
                        description=(
                            f"Claimed {pct}% change from {val_from} to {val_to}, "
                            f"but actual change is {actual_pct:.1f}%"
                        ),
                        severity="error",
                        context=m.group(0),
                    )
                )

    # 2. Date logic
    dates = [int(m.group(1)) for m in _DATE_PATTERN.finditer(text)]
    claims_found += len(dates)
    current_year = datetime.now().year
    for d in dates:
        if d > current_year + 5:
            issues.append(
                NumericIssue(
                    issue_type="date_logic",
                    description=f"Year {d} is in the far future (current: {current_year})",
                    severity="warning",
                    context=str(d),
                )
            )

    # Check birth/death, founded/closed ordering via context
    _check_date_ordering(text, dates, issues)

    # 3. Probability bounds
    for m in _PROB_PATTERN.finditer(text):
        claims_found += 1
        prob = _parse_number(m.group(1))
        if prob > 100:
            issues.append(
                NumericIssue(
                    issue_type="probability",
                    description=f"Probability {prob}% exceeds 100%",
                    severity="error",
                    context=m.group(0),
                )
            )
        elif prob < 0:
            issues.append(
                NumericIssue(
                    issue_type="probability",
                    description=f"Negative probability {prob}%",
                    severity="error",
                    context=m.group(0),
                )
            )

    # 4. Order of magnitude for well-known quantities
    _check_magnitude(text, issues)

    # 5. Internal consistency: same number referenced differently
    _check_internal_consistency(text, issues)

    # Count raw numbers
    for _m in _NUMBER_PATTERN.finditer(text):
        claims_found += 1

    result = NumericVerificationResult(
        claims_found=claims_found,
        issues=issues,
        valid=all(i.severity != "error" for i in issues),
    )
    return result


def _check_date_ordering(
    text: str, dates: list[int], issues: list[NumericIssue]
) -> None:
    text_lower = text.lower()

    born_years = []
    died_years = []
    for m in re.finditer(r"born\s+(?:in\s+)?(\d{4})", text_lower):
        born_years.append(int(m.group(1)))
    for m in re.finditer(r"died\s+(?:in\s+)?(\d{4})", text_lower):
        died_years.append(int(m.group(1)))

    for b in born_years:
        for d in died_years:
            if d < b:
                issues.append(
                    NumericIssue(
                        issue_type="date_logic",
                        description=f"Death year {d} is before birth year {b}",
                        severity="error",
                        context=f"born {b}, died {d}",
                    )
                )

    founded_years = []
    for m in re.finditer(r"founded\s+(?:in\s+)?(\d{4})", text_lower):
        founded_years.append(int(m.group(1)))
    current_year = datetime.now().year
    for f in founded_years:
        if f > current_year:
            issues.append(
                NumericIssue(
                    issue_type="date_logic",
                    description=f"Founded in {f} is in the future",
                    severity="error",
                    context=f"founded {f}",
                )
            )


_MAGNITUDE_CHECKS = [
    (
        re.compile(r"earth.*?population.*?(\d+(?:\.\d+)?)\s*(billion|million)", re.I),
        "earth_population",
        6.0,
        12.0,
        "billion",
    ),
    (
        re.compile(r"speed\s+of\s+light.*?(\d+(?:,\d+)*(?:\.\d+)?)\s*(km/s|m/s)", re.I),
        "speed_of_light_km",
        200000,
        400000,
        "km/s",
    ),
]


def _check_magnitude(text: str, issues: list[NumericIssue]) -> None:
    for pattern, name, lo, hi, unit in _MAGNITUDE_CHECKS:
        m = pattern.search(text)
        if m:
            val = _parse_number(m.group(1))
            found_unit = m.group(2).lower()
            if found_unit == "million" and unit == "billion":
                val /= 1000
            if val < lo or val > hi:
                issues.append(
                    NumericIssue(
                        issue_type="magnitude",
                        description=(
                            f"{name}: {val} {found_unit} outside expected range "
                            f"[{lo}-{hi}] {unit}"
                        ),
                        severity="warning",
                        context=m.group(0),
                    )
                )


def _check_internal_consistency(text: str, issues: list[NumericIssue]) -> None:
    """Check if the same total is referenced with different values."""
    total_pattern = re.compile(
        r"total\s+(?:of\s+)?(\d+(?:,\d+)*(?:\.\d+)?)", re.IGNORECASE
    )
    totals = [_parse_number(m.group(1)) for m in total_pattern.finditer(text)]
    if len(totals) >= 2:
        for i in range(1, len(totals)):
            if abs(totals[i] - totals[0]) > 0.01 * max(totals[0], 1):
                issues.append(
                    NumericIssue(
                        issue_type="internal",
                        description=(
                            f"Inconsistent totals: {totals[0]} vs {totals[i]}"
                        ),
                        severity="error",
                        context=f"total {totals[0]} ... total {totals[i]}",
                    )
                )
