# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — moderation detector protocol

"""Shared protocol and result types for moderation detectors.

A detector consumes a piece of text and returns a
:class:`ModerationResult` containing zero or more
:class:`ModerationMatch` records. A match is purely informational —
deciding what to do with the match (block, redact, flag) is the
caller's responsibility. :class:`Policy.check` surfaces every match
as a :class:`Violation` so the existing enforcement pipeline does
not need a new branch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModerationMatch:
    """One finding from a detector."""

    detector: str
    category: str
    start: int
    end: int
    text: str
    score: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    def snippet(self, window: int = 40) -> str:
        """Return a short excerpt around the match for logging."""
        lo = max(0, self.start - window)
        hi = self.end + window
        return self.text[lo:hi]


@dataclass(frozen=True)
class ModerationResult:
    """Aggregated output of a single detector run."""

    detector: str
    matches: list[ModerationMatch] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return bool(self.matches)


class ModerationDetector(ABC):
    """Protocol every PII/toxicity detector implements."""

    name: str = "moderation"

    @abstractmethod
    def analyse(self, text: str) -> ModerationResult:
        """Return every match found in ``text``. Must be side-effect
        free: calling ``analyse`` twice on the same input must
        produce the same matches (model-based detectors that change
        weights between calls should expose that behaviour through
        a separate method)."""
        ...  # pragma: no cover
