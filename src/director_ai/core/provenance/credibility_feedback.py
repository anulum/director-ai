# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — online credibility learning from human feedback

"""Turn human feedback into per-source credibility over time.

:class:`~director_ai.core.provenance.credibility.SourceCredibility` already
tracks a decaying trust score per source and already feeds
:class:`~director_ai.core.provenance.verifier.ProvenanceVerifier`'s composite
trust score. What was missing was the online-learning step: nothing fed the
tracker from operator feedback. :class:`CredibilityFeedbackLoop` closes that
loop. When a human approves or rejects a response, every source cited in that
response is observed with the corresponding signal, so a source whose cited
facts keep getting rejected drifts down while a consistently-approved source
drifts up. Because the tracker is shared with the verifier, the trust score a
later response receives is learned from the feedback the earlier ones drew.

The same credibility can re-rank retrieval candidates: :meth:`rerank` blends
each chunk's retrieval relevance with its source credibility, so a marginally
less relevant chunk from a highly trusted source can outrank a slightly closer
chunk from a source operators keep correcting.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence

from ..calibration.feedback_store import Correction
from ..types import EvidenceChunk
from .credibility import SourceCredibility, SourceScore

__all__ = ["CredibilityFeedbackLoop"]


def _unit_interval(name: str, value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]; got {value!r}")
    return float(value)


class CredibilityFeedbackLoop:
    """Update a :class:`SourceCredibility` tracker from human feedback.

    Parameters
    ----------
    credibility :
        The tracker to update. Share the same instance with the
        :class:`ProvenanceVerifier` whose trust scores should learn from the
        feedback.
    approve_signal :
        Signal folded in for an approved response. Default 1.0.
    reject_signal :
        Signal folded in for a rejected response. Default 0.0.
    """

    def __init__(
        self,
        *,
        credibility: SourceCredibility,
        approve_signal: float = 1.0,
        reject_signal: float = 0.0,
    ) -> None:
        self._credibility = credibility
        self._approve_signal = _unit_interval("approve_signal", approve_signal)
        self._reject_signal = _unit_interval("reject_signal", reject_signal)

    def observe(
        self,
        *,
        source_ids: Sequence[str],
        human_approved: bool,
    ) -> tuple[SourceScore, ...]:
        """Fold one human verdict into every cited source's credibility.

        Blank source ids are ignored; each distinct source is observed once.
        Returns the updated :class:`SourceScore` for each observed source.
        """
        signal = self._approve_signal if human_approved else self._reject_signal
        observed: list[SourceScore] = []
        for source_id in _unique_non_empty(source_ids):
            observed.append(self._credibility.observe(source_id, signal))
        return tuple(observed)

    def observe_correction(
        self,
        correction: Correction,
        *,
        source_ids: Sequence[str],
    ) -> tuple[SourceScore, ...]:
        """Fold a :class:`Correction`'s human verdict into cited sources.

        The feedback store records the response, not the citations, so the
        caller resolves which sources the corrected response cited.
        """
        return self.observe(
            source_ids=source_ids,
            human_approved=correction.human_approved,
        )

    def ingest_corrections(
        self,
        corrections: Iterable[Correction],
        *,
        source_resolver: Callable[[Correction], Sequence[str]],
    ) -> int:
        """Replay stored corrections through the loop.

        ``source_resolver`` maps each correction to the source ids its
        response cited. Corrections that resolve to no source are skipped.
        Returns the number of corrections that updated at least one source.
        """
        applied = 0
        for correction in corrections:
            source_ids = source_resolver(correction)
            if self.observe_correction(correction, source_ids=source_ids):
                applied += 1
        return applied

    def credibility_of(self, source_id: str) -> float:
        """Return the current decayed credibility of one source."""
        return self._credibility.score(source_id)

    def rerank(
        self,
        chunks: Sequence[EvidenceChunk],
        *,
        credibility_weight: float = 0.5,
    ) -> list[EvidenceChunk]:
        """Return ``chunks`` reordered by blended relevance and credibility.

        ``credibility_weight`` in ``[0, 1]`` is the share given to source
        credibility; the remainder weights retrieval relevance (derived from
        the chunk distance). A weight of 0 preserves the pure relevance order;
        a weight of 1 ranks purely by source credibility. The sort is stable,
        so ties keep their incoming order, and chunk distances are left
        unchanged.
        """
        weight = _unit_interval("credibility_weight", credibility_weight)
        scored = [
            (self._blended_score(chunk, weight), index, chunk)
            for index, chunk in enumerate(chunks)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [chunk for _score, _index, chunk in scored]

    def _blended_score(self, chunk: EvidenceChunk, weight: float) -> float:
        """Blend one chunk's relevance with its source credibility."""
        relevance = 1.0 / (1.0 + max(0.0, chunk.distance))
        credibility = self._credibility.score(chunk.source) if chunk.source else 0.5
        return (1.0 - weight) * relevance + weight * credibility


def _unique_non_empty(source_ids: Sequence[str]) -> list[str]:
    """Return distinct, non-empty source ids preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for source_id in source_ids:
        if source_id and source_id not in seen:
            seen.add(source_id)
            out.append(source_id)
    return out
