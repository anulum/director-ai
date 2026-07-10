# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — citation-grounding judge

"""Decide whether each assertion in an answer is grounded in what it cites.

This is the core of the HalluHard groundedness metric. An answer is split into
sentence-level assertions; each is matched to the citations that occur within it
(the inline ``… claim [2].`` convention); and the cited sources' text is scored
against the assertion with the NLI scorer. An assertion counts as **grounded**
only when it carries a citation *and* the cited material entails it — an uncited
factual sentence, or one whose cited source fails to support it, is a
hallucination by HalluHard's definition.

The judge is backend-agnostic: it takes any object exposing
``score(premise, hypothesis) -> float`` (the :class:`Scorer` protocol, satisfied
by :class:`~director_ai.core.scoring.nli.NLIScorer`), so it is fully testable
with a stub and needs no model to exercise its logic.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from .citations import reference_section_start, resolve_citations

__all__ = [
    "CitationGroundingJudge",
    "ClaimGrounding",
    "GroundingReport",
    "Scorer",
]

# Sentence span: text up to and including its terminal punctuation.
_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]+|[^.!?]+$")
_MIN_CLAIM_WORDS = 4


class Scorer(Protocol):
    """Anything that scores divergence: ``0`` = entailment, ``1`` = contradiction."""

    def score(self, premise: str, hypothesis: str) -> float:
        """Return divergence for ``hypothesis`` against ``premise``."""
        ...


@dataclass(frozen=True)
class ClaimGrounding:
    """The grounding outcome for one assertion in the answer."""

    claim: str
    has_citation: bool
    grounded: bool
    support: float  # 0-1, NLI support (1 - divergence); 0.0 when uncited
    cited: tuple[str, ...] = ()  # resolved identifiers cited by this claim

    def to_dict(self) -> dict[str, object]:
        """Serialise this claim-grounding result for reports."""
        return {
            "claim": self.claim,
            "has_citation": self.has_citation,
            "grounded": self.grounded,
            "support": round(self.support, 4),
            "cited": list(self.cited),
        }


@dataclass(frozen=True)
class GroundingReport:
    """Per-answer grounding summary (no raw source text retained)."""

    claims: tuple[ClaimGrounding, ...] = field(default_factory=tuple)

    @property
    def total(self) -> int:
        """Return the number of assertions in the report."""
        return len(self.claims)

    @property
    def grounded_fraction(self) -> float:
        """Fraction of assertions that are cited *and* supported."""
        if not self.claims:
            return 1.0
        return sum(1 for c in self.claims if c.grounded) / len(self.claims)

    @property
    def citation_coverage(self) -> float:
        """Fraction of assertions that carry at least one citation."""
        if not self.claims:
            return 1.0
        return sum(1 for c in self.claims if c.has_citation) / len(self.claims)

    @property
    def hallucinated(self) -> tuple[ClaimGrounding, ...]:
        """Assertions that are not grounded (uncited or unsupported)."""
        return tuple(c for c in self.claims if not c.grounded)

    def to_dict(self) -> dict[str, object]:
        """Serialise this grounding report without source text."""
        return {
            "total": self.total,
            "grounded_fraction": round(self.grounded_fraction, 4),
            "citation_coverage": round(self.citation_coverage, 4),
            "hallucinated_count": len(self.hallucinated),
            "claims": [c.to_dict() for c in self.claims],
        }


def _sentences_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Split into ``(sentence, start, end)`` triples, dropping short fragments."""
    out: list[tuple[str, int, int]] = []
    for m in _SENTENCE_RE.finditer(text):
        sentence = m.group(0).strip()
        if len(sentence.split()) >= _MIN_CLAIM_WORDS:
            out.append((sentence, m.start(), m.end()))
    return out


class CitationGroundingJudge:
    """Score how well an answer's assertions are grounded in their citations.

    Parameters
    ----------
    scorer : Scorer
        Divergence scorer (``0`` = entailment). An
        :class:`~director_ai.core.scoring.nli.NLIScorer` is the production choice.
    support_threshold : float
        Minimum NLI support (``1 - divergence``) for a cited assertion to count
        as grounded (default 0.6).
    """

    def __init__(self, *, scorer: Scorer, support_threshold: float = 0.6) -> None:
        if not 0.0 < support_threshold <= 1.0:
            raise ValueError("support_threshold must be in (0, 1]")
        self._scorer = scorer
        self._threshold = support_threshold

    def assess(self, answer: str, sources: Mapping[str, str]) -> GroundingReport:
        """Assess every assertion in ``answer`` against its cited ``sources``.

        ``sources`` maps a resolved citation identifier (DOI / arXiv id / URL /
        author-year) to the fetched source text. A citation whose identifier is
        absent from ``sources`` (e.g. the fetch failed) contributes no evidence,
        so the assertion is judged ungrounded rather than silently passed.
        """
        citations = resolve_citations(answer)
        body = answer[: reference_section_start(answer)]
        results: list[ClaimGrounding] = []
        for sentence, start, end in _sentences_with_spans(body):
            cited = tuple(c.identifier for c in citations if start <= c.start < end)
            if not cited:
                results.append(ClaimGrounding(sentence, False, False, 0.0, ()))
                continue
            evidence = " ".join(
                sources[ident] for ident in cited if sources.get(ident)
            ).strip()
            if not evidence:
                results.append(ClaimGrounding(sentence, True, False, 0.0, cited))
                continue
            support = 1.0 - _clamp(self._scorer.score(evidence, sentence))
            grounded = support >= self._threshold
            results.append(ClaimGrounding(sentence, True, grounded, support, cited))
        return GroundingReport(tuple(results))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
