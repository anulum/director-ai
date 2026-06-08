# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — threat-intelligence indicators

"""Indicators of compromise (IOCs) matched against prompts and responses.

A :class:`ThreatIndicator` is one known-bad signal — a substring, a regex, or a
content hash — carrying the attribution (the intrusion set or actor it belongs to)
and severity that make a match actionable: not just "blocked" but "this matches
the APT29 phishing kit". Indicators are the local, STIX-aligned representation the
matcher works with; they can be authored directly or imported from a STIX bundle.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from typing import Any

__all__ = ["IndicatorType", "Severity", "ThreatIndicator"]


@lru_cache(maxsize=4096)
def _compiled_regex(pattern: str) -> re.Pattern[str]:
    """Compile and cache a regex indicator pattern (shared across matches)."""
    return re.compile(pattern)


class IndicatorType(StrEnum):
    """How an indicator's pattern is matched against text."""

    SUBSTRING = "substring"
    """Case-insensitive substring containment."""

    REGEX = "regex"
    """A regular expression searched anywhere in the text."""

    SHA256 = "sha256"
    """The SHA-256 of the whole text equals the (hex) pattern."""


class Severity(StrEnum):
    """Indicator severity, ordered low → critical."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_SEVERITY_RANK = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


def severity_rank(severity: Severity) -> int:
    """Return the ordinal rank of ``severity`` (low=0 … critical=3)."""
    return _SEVERITY_RANK[severity]


@dataclass(frozen=True)
class ThreatIndicator:
    """One indicator of compromise with attribution and severity."""

    id: str
    name: str
    indicator_type: IndicatorType
    pattern: str
    attribution: str = ""
    severity: Severity = Severity.MEDIUM
    labels: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("indicator id is required")
        if not self.pattern.strip():
            raise ValueError("indicator pattern is required")
        if self.indicator_type is IndicatorType.REGEX:
            try:
                _compiled_regex(self.pattern)
            except re.error as exc:
                raise ValueError(f"invalid regex pattern: {exc}") from exc
        if self.indicator_type is IndicatorType.SHA256 and not re.fullmatch(
            r"[0-9a-fA-F]{64}", self.pattern
        ):
            raise ValueError("sha256 indicator pattern must be 64 hex characters")

    def matches(self, text: str) -> bool:
        """Whether ``text`` triggers this indicator."""
        if self.indicator_type is IndicatorType.SUBSTRING:
            return self.pattern.lower() in text.lower()
        if self.indicator_type is IndicatorType.REGEX:
            return _compiled_regex(self.pattern).search(text) is not None
        return hashlib.sha256(text.encode("utf-8")).hexdigest() == self.pattern.lower()

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict (the indicator definition)."""
        return {
            "id": self.id,
            "name": self.name,
            "indicator_type": str(self.indicator_type),
            "pattern": self.pattern,
            "attribution": self.attribution,
            "severity": str(self.severity),
            "labels": list(self.labels),
        }
