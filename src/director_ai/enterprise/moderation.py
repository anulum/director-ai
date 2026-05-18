# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Enterprise content moderation wrapper

"""Enterprise content moderation wrapper.

Combines PII redaction with toxicity detectors and returns a single
operator-facing allow, warn, redact, or block decision. The structured report is
tenant-safe: it contains detector/category/offset metadata and the safe text,
but never serialises raw matched values.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from .redactor import PIIRedactionFinding, PIIRedactor

if TYPE_CHECKING:
    from director_ai.core.safety.moderation import ModerationDetector, ModerationMatch


class ModerationAction(StrEnum):
    """Decision produced by :class:`ContentModerator`."""

    ALLOW = "allow"
    WARN = "warn"
    REDACT = "redact"
    BLOCK = "block"


@dataclass(frozen=True)
class ContentModerationFinding:
    """Tenant-safe metadata for one moderation finding."""

    detector: str
    category: str
    start: int
    end: int
    action: ModerationAction
    replacement: str | None = None
    score: float = 1.0

    @classmethod
    def from_redaction(
        cls,
        finding: PIIRedactionFinding,
    ) -> ContentModerationFinding:
        """Convert a PII redaction finding into moderation metadata."""
        return cls(
            detector=finding.detector,
            category=finding.category,
            start=finding.start,
            end=finding.end,
            action=ModerationAction.REDACT,
            replacement=finding.replacement,
            score=finding.score,
        )

    @classmethod
    def from_match(
        cls,
        match: ModerationMatch,
        *,
        action: ModerationAction,
    ) -> ContentModerationFinding:
        """Convert a detector match into a tenant-safe moderation finding."""
        return cls(
            detector=match.detector,
            category=match.category.lower(),
            start=match.start,
            end=match.end,
            action=action,
            score=float(match.score),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialise finding metadata without raw matched text."""
        payload: dict[str, object] = {
            "action": self.action.value,
            "category": self.category,
            "detector": self.detector,
            "end": self.end,
            "score": self.score,
            "start": self.start,
        }
        if self.replacement is not None:
            payload["replacement"] = self.replacement
        return payload


@dataclass(frozen=True)
class ContentModerationResult:
    """Result of a content moderation pass."""

    action: ModerationAction
    safe_text: str
    findings: tuple[ContentModerationFinding, ...]

    @property
    def blocked(self) -> bool:
        """Whether the moderation result blocks downstream release."""
        return self.action is ModerationAction.BLOCK

    @property
    def category_counts(self) -> dict[str, int]:
        """Return stable finding counts grouped by category."""
        return dict(sorted(Counter(f.category for f in self.findings).items()))

    def to_dict(self) -> dict[str, object]:
        """Serialise a tenant-safe moderation report."""
        return {
            "action": self.action.value,
            "blocked": self.blocked,
            "category_counts": self.category_counts,
            "findings": [finding.to_dict() for finding in self.findings],
            "privacy": {
                "payload_classification": "tenant_safe",
                "raw_input_included": False,
            },
            "safe_text": self.safe_text,
        }


class ContentModerator:
    """Production wrapper for PII redaction and toxicity moderation."""

    def __init__(
        self,
        *,
        enable_pii: bool = True,
        toxicity_action: ModerationAction = ModerationAction.BLOCK,
        toxicity_detectors: Sequence[ModerationDetector] | None = None,
        prefer_rust: bool = True,
    ) -> None:
        if toxicity_action not in (ModerationAction.BLOCK, ModerationAction.WARN):
            raise ValueError("toxicity_action must be block or warn")
        self._pii_redactor = (
            PIIRedactor(prefer_rust=prefer_rust) if enable_pii else None
        )
        self._toxicity_action = toxicity_action
        self._toxicity_detectors = (
            tuple(toxicity_detectors)
            if toxicity_detectors is not None
            else (_build_default_toxicity_detector(prefer_rust=prefer_rust),)
        )

    def moderate(self, text: str) -> ContentModerationResult:
        """Moderate text and return the safe text plus tenant-safe metadata."""
        safe_text = text
        findings: list[ContentModerationFinding] = []
        redacted = False

        if self._pii_redactor is not None:
            redaction = self._pii_redactor.redact_with_report(text)
            safe_text = redaction.redacted_text
            redacted = redaction.redacted
            findings.extend(
                ContentModerationFinding.from_redaction(finding)
                for finding in redaction.findings
            )

        toxicity_findings: list[ContentModerationFinding] = []
        for detector in self._toxicity_detectors:
            result = detector.analyse(safe_text)
            toxicity_findings.extend(
                ContentModerationFinding.from_match(
                    match,
                    action=self._toxicity_action,
                )
                for match in result.matches
            )
        findings.extend(toxicity_findings)

        if any(f.action is ModerationAction.BLOCK for f in toxicity_findings):
            action = ModerationAction.BLOCK
        elif any(f.action is ModerationAction.WARN for f in toxicity_findings):
            action = ModerationAction.WARN
        elif redacted:
            action = ModerationAction.REDACT
        else:
            action = ModerationAction.ALLOW

        return ContentModerationResult(
            action=action,
            safe_text=safe_text,
            findings=tuple(findings),
        )


def _build_default_toxicity_detector(*, prefer_rust: bool) -> ModerationDetector:
    """Build the default dependency-light toxicity detector."""
    from director_ai.core.safety.moderation import KeywordToxicityDetector

    return KeywordToxicityDetector(prefer_rust=prefer_rust)
