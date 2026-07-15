# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — PII Redaction (core)

"""PII (Personally Identifiable Information) redaction.

Detects and masks sensitive information before processing or logging. The
structured report intentionally carries categories, offsets, replacements, and
counts only; raw matched values are never serialised into tenant-safe metadata.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from director_ai.core.safety.moderation import ModerationDetector

_REPLACEMENTS: dict[str, str] = {
    "card": "[CARD]",
    "credit_card": "[CARD]",
    "email": "[EMAIL]",
    "iban": "[IBAN]",
    "ipv4": "[IPV4]",
    "passport": "[PASSPORT]",
    "person": "[PERSON]",
    "phone": "[PHONE]",
    "phi": "[PHI]",
    "ssn": "[SSN]",
}


def _stable_category(category: str) -> str:
    """Normalise detector categories to stable public labels."""
    normalised = category.lower()
    if normalised == "credit_card":
        return "card"
    return normalised


def _replacement_for(category: str) -> str:
    """Return the redaction token for a detector category."""
    stable = _stable_category(category)
    return _REPLACEMENTS.get(stable, f"[{stable.upper()}]")


@dataclass(frozen=True)
class PIIRedactionFinding:
    """Tenant-safe description of one redacted PII span."""

    detector: str
    category: str
    start: int
    end: int
    replacement: str
    score: float = 1.0

    @classmethod
    def from_match(cls, match: Any) -> PIIRedactionFinding:
        """Build a redaction finding from a moderation detector match."""
        category = _stable_category(match.category)
        return cls(
            detector=match.detector,
            category=category,
            start=match.start,
            end=match.end,
            replacement=_replacement_for(category),
            score=float(match.score),
        )

    def to_dict(self) -> dict[str, object]:
        """Return metadata without the raw matched text."""
        return {
            "category": self.category,
            "detector": self.detector,
            "end": self.end,
            "replacement": self.replacement,
            "score": self.score,
            "start": self.start,
        }


@dataclass(frozen=True)
class PIIRedactionReport:
    """PII redaction result and tenant-safe metadata."""

    redacted_text: str
    findings: tuple[PIIRedactionFinding, ...]

    @property
    def redacted(self) -> bool:
        """Whether at least one PII span was replaced."""
        return bool(self.findings)

    @property
    def category_counts(self) -> dict[str, int]:
        """Return stable redaction counts grouped by category."""
        return dict(sorted(Counter(f.category for f in self.findings).items()))

    def to_dict(self) -> dict[str, object]:
        """Serialise a tenant-safe report suitable for audit logs."""
        return {
            "category_counts": self.category_counts,
            "findings": [finding.to_dict() for finding in self.findings],
            "privacy": {
                "payload_classification": "tenant_safe",
                "raw_payload_included": False,
            },
            "redacted": self.redacted,
            "redacted_text": self.redacted_text,
        }


def _select_non_overlapping(
    findings: Iterable[PIIRedactionFinding],
) -> tuple[PIIRedactionFinding, ...]:
    """Keep the earliest non-overlapping redaction spans.

    Candidates are visited in start order, so the selected spans are
    disjoint with strictly increasing ends — a candidate can only ever
    overlap the most recently selected span, and one ``last_end``
    comparison replaces a scan of every accepted span. The previous
    full-scan selection was quadratic: ~120 000 findings (a 2 MB input
    of dense PII) took 13 minutes; this selection is linear.
    """
    ordered = sorted(findings, key=lambda f: (f.start, -(f.end - f.start), f.category))
    selected: list[PIIRedactionFinding] = []
    last_end = 0
    for finding in ordered:
        if finding.start < 0 or finding.end <= finding.start:
            continue
        if finding.start < last_end:
            continue
        selected.append(finding)
        last_end = finding.end
    return tuple(selected)


class PIIRedactor:
    """Enterprise redaction pipeline for sensitive string values."""

    def __init__(
        self,
        enabled: bool = True,
        *,
        detectors: Sequence[ModerationDetector] | None = None,
        prefer_rust: bool = True,
    ) -> None:
        self.enabled = enabled
        self._prefer_rust = prefer_rust
        self._detectors = tuple(detectors) if detectors is not None else None

    def redact_with_report(self, text: str) -> PIIRedactionReport:
        """Redact PII and return a tenant-safe structured report."""
        if not self.enabled or not text:
            return PIIRedactionReport(redacted_text=text, findings=())

        findings: list[PIIRedactionFinding] = []
        for detector in self._resolve_detectors():
            result = detector.analyse(text)
            findings.extend(
                PIIRedactionFinding.from_match(match) for match in result.matches
            )
        selected = _select_non_overlapping(findings)
        if not selected:
            return PIIRedactionReport(redacted_text=text, findings=())

        pieces: list[str] = []
        cursor = 0
        for finding in selected:
            pieces.append(text[cursor : finding.start])
            pieces.append(finding.replacement)
            cursor = finding.end
        pieces.append(text[cursor:])
        return PIIRedactionReport(redacted_text="".join(pieces), findings=selected)

    def redact(self, text: str) -> str:
        """Redact known PII patterns from the text string."""
        return self.redact_with_report(text).redacted_text

    def __call__(self, text: str) -> str:
        """Redact the given text and return the sanitised result."""
        return self.redact(text)

    def _resolve_detectors(self) -> tuple[ModerationDetector, ...]:
        """Return configured detectors, building defaults only when needed."""
        if self._detectors is not None:
            return self._detectors
        return (_cached_default_regex_detector(prefer_rust=self._prefer_rust),)


def _build_default_regex_detector(*, prefer_rust: bool) -> ModerationDetector:
    """Build the default dependency-light regex PII detector."""
    from director_ai.core.safety.moderation import RegexPIIDetector

    return RegexPIIDetector(prefer_rust=prefer_rust)


@lru_cache(maxsize=2)
def _cached_default_regex_detector(*, prefer_rust: bool) -> ModerationDetector:
    """Return the process-wide default regex detector for redaction.

    Regex detectors are immutable after construction, so sharing the default
    instance avoids rebuilding the optional Rust scanner for every scorer or
    moderation pipeline object.
    """
    return _build_default_regex_detector(prefer_rust=prefer_rust)
