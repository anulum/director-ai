# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — counterfactual contradiction explainer

"""Explain which retrieved passage contradicts a claim, and why.

Grounding a claim is not only about finding support — it is about surfacing
the evidence that *refutes* it. Given a claim and the passages a retriever
returned, :class:`ContradictionExplainer` scores each passage for
contradiction and returns a human-readable account: *this claim contradicts
the passage from source X because the passage states "…" (contradiction
0.91)*.

The contradiction signal is injected, mirroring
:class:`~director_ai.core.retrieval.conflict_guard.ConflictAwareKnowledgeGuard`'s
``score_fn``: the caller supplies a ``scorer(passage, claim) -> probability``
backed by whatever verifier the deployment trusts — the NLI scorer in
:mod:`director_ai.core.scoring.nli`, a rule engine, or a domain model. Keeping
the model out of this module makes the selection-and-explanation logic
deterministic and testable on its own.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..types import EvidenceChunk

__all__ = [
    "ContradictionExplainer",
    "ContradictionExplanation",
    "ContradictionReport",
    "ContradictionScorer",
]

ContradictionScorer = Callable[[str, str], float]
"""``scorer(passage, claim) -> contradiction probability in [0, 1]``."""

_EXCERPT_LIMIT = 160


def _unit_interval(name: str, value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]; got {value!r}")
    return float(value)


def _excerpt(text: str, limit: int = _EXCERPT_LIMIT) -> str:
    stripped = " ".join(text.split())
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[: limit - 1]}…"


@dataclass(frozen=True)
class ContradictionExplanation:
    """One passage that contradicts the claim, with a stated rationale."""

    claim: str
    chunk_index: int
    chunk_source: str
    chunk_excerpt: str
    score: float
    rationale: str

    def __post_init__(self) -> None:
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        _unit_interval("score", self.score)


@dataclass(frozen=True)
class ContradictionReport:
    """The contradictions found for one claim, strongest first."""

    claim: str
    contradictions: tuple[ContradictionExplanation, ...]

    @property
    def has_contradiction(self) -> bool:
        """Return ``True`` when at least one passage contradicts the claim."""
        return bool(self.contradictions)

    @property
    def best(self) -> ContradictionExplanation | None:
        """Return the strongest contradiction, or ``None`` when there is none."""
        return self.contradictions[0] if self.contradictions else None


class ContradictionExplainer:
    """Find and explain passages that contradict a claim.

    Parameters
    ----------
    scorer :
        ``scorer(passage, claim) -> probability`` in ``[0, 1]`` that the
        passage contradicts the claim.
    threshold :
        Minimum contradiction probability for a passage to be reported.
        Default 0.5.
    """

    def __init__(
        self,
        *,
        scorer: ContradictionScorer,
        threshold: float = 0.5,
    ) -> None:
        if not callable(scorer):
            raise TypeError("scorer must be callable")
        self._scorer = scorer
        self._threshold = _unit_interval("threshold", threshold)

    def explain(
        self,
        claim: str,
        chunks: Sequence[EvidenceChunk],
    ) -> ContradictionReport:
        """Return the contradictions for ``claim`` across ``chunks``.

        Each passage is scored; passages at or above the threshold become
        :class:`ContradictionExplanation` entries sorted by descending score.
        Empty-text passages are skipped. Raises :class:`ValueError` for an
        empty claim.
        """
        if not claim or not claim.strip():
            raise ValueError("claim must be a non-empty string")
        explanations: list[ContradictionExplanation] = []
        for index, chunk in enumerate(chunks):
            if not chunk.text or not chunk.text.strip():
                continue
            score = _unit_interval(
                "contradiction score", float(self._scorer(chunk.text, claim))
            )
            if score < self._threshold:
                continue
            explanations.append(self._build_explanation(claim, index, chunk, score))
        explanations.sort(key=lambda item: (-item.score, item.chunk_index))
        return ContradictionReport(claim=claim, contradictions=tuple(explanations))

    def _build_explanation(
        self,
        claim: str,
        index: int,
        chunk: EvidenceChunk,
        score: float,
    ) -> ContradictionExplanation:
        """Assemble one contradiction explanation for a scored passage."""
        excerpt = _excerpt(chunk.text)
        source = chunk.source or "an unattributed source"
        rationale = (
            f"This claim contradicts the passage from {source} because the "
            f'passage states: "{excerpt}" (contradiction {score:.2f}).'
        )
        return ContradictionExplanation(
            claim=claim,
            chunk_index=index,
            chunk_source=chunk.source,
            chunk_excerpt=excerpt,
            score=score,
            rationale=rationale,
        )
