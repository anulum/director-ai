# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — citation tracing for chain-of-thought

"""Trace the claims in a reasoning chain to the citations that back them.

A grounded answer cites its sources; an ungrounded one asserts and moves on. This
module links the two: it segments the response body into claim sentences, attaches
each inline citation to the claim it sits in, and reports which claims carry a
citation and which do not.

It reuses the citation parser in
:mod:`director_ai.core.citation_grounding.citations` — DOIs, arXiv ids, URLs,
numeric markers resolved through the reference list, and author-year forms — and
maps each citation's character offset onto the sentence that contains it.
Citations inside the reference list are excluded (they are bibliography entries,
not citing markers).

Coverage is a coarse signal: not every sentence needs a citation, so an uncited
claim is a candidate for review, not a defect. The tracing itself is exact and
deterministic — character-offset interval mapping, no scoring or inference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # ``Citation`` is used only as a (PEP 563 string) annotation here, so the
    # advanced BUSL-1.1 citation_grounding package is not needed to import this
    # module — the Apache-2.0 core wheel excludes it. ``trace_citations`` imports
    # the runtime helpers lazily and raises a clear error if the tier is absent.
    from ..citation_grounding.citations import Citation

__all__ = [
    "ClaimCitation",
    "TraceResult",
    "trace_citations",
]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORDLIKE = re.compile(r"[A-Za-z0-9]")


def _sentence_spans(body: str) -> list[tuple[int, int, str]]:
    """Segment *body* into ``(start, end, sentence)`` spans with offsets."""
    spans: list[tuple[int, int, str]] = []
    start = 0
    for match in _SENT_SPLIT.finditer(body):
        spans.append((start, match.start(), body[start : match.start()]))
        start = match.end()
    if start < len(body):
        spans.append((start, len(body), body[start:]))
    return [(s, e, t) for s, e, t in spans if _WORDLIKE.search(t)]


@dataclass(frozen=True)
class ClaimCitation:
    """One claim sentence and the citations attached to it."""

    index: int
    claim: str
    citations: tuple[Citation, ...]

    @property
    def cited(self) -> bool:
        return bool(self.citations)


@dataclass
class TraceResult:
    """Citation tracing over a reasoning chain."""

    claims: list[ClaimCitation] = field(default_factory=list)

    @property
    def cited(self) -> list[ClaimCitation]:
        return [c for c in self.claims if c.cited]

    @property
    def uncited(self) -> list[ClaimCitation]:
        return [c for c in self.claims if not c.cited]

    @property
    def coverage(self) -> float:
        """Fraction of claim sentences that carry at least one citation."""
        if not self.claims:
            return 0.0
        return len(self.cited) / len(self.claims)


def trace_citations(text: str) -> TraceResult:
    """Link the claims in *text* to their inline citations.

    Returns a :class:`TraceResult` with one :class:`ClaimCitation` per claim
    sentence in the body (the reference section, if any, is excluded).
    """
    try:
        from ..citation_grounding.citations import (
            reference_section_start,
            resolve_citations,
        )
    except ImportError as exc:  # pragma: no cover - only without the advanced tier
        raise ImportError(
            "citation tracing requires the advanced citation-grounding module "
            "(BUSL-1.1), which is absent from the Apache-2.0 core wheel"
        ) from exc
    body_end = reference_section_start(text)
    citations = [c for c in resolve_citations(text) if c.start < body_end]
    spans = _sentence_spans(text[:body_end])

    claims = [
        ClaimCitation(
            index=index,
            claim=sentence.strip(),
            citations=tuple(c for c in citations if start <= c.start < end),
        )
        for index, (start, end, sentence) in enumerate(spans)
    ]
    return TraceResult(claims=claims)
