# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Lite Scorer (No-NLI Fast Path)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Lightweight divergence scorer using word overlap, length ratio,
named entity heuristics, and negation asymmetry. ~0.5ms/pair,
~65% accuracy on AggreFact subset.

Usage::

    scorer = LiteScorer()
    score = scorer.score("The sky is blue.", "The sky is green.")
"""

from __future__ import annotations

from ._heuristics import ENTITY_RE as _ENTITY_RE
from ._heuristics import NEGATION_WORDS as _NEGATION_WORDS
from ._heuristics import WORD_RE as _WORD_RE

__all__ = ["LiteScorer"]


class LiteScorer:
    """Fast divergence scorer without any ML model dependency."""

    def score(self, premise: str, hypothesis: str) -> float:
        """Compute divergence in [0, 1]. 0 = aligned, 1 = contradicted."""
        if not premise or not hypothesis:
            return 0.5

        p_words = set(_WORD_RE.findall(premise.lower()))
        h_words = set(_WORD_RE.findall(hypothesis.lower()))

        if not p_words or not h_words:
            return 0.5

        # Jaccard overlap
        intersection = len(p_words & h_words)
        union = len(p_words | h_words)
        jaccard = intersection / union

        # Length ratio penalty
        len_ratio = min(len(premise), len(hypothesis)) / max(
            len(premise), len(hypothesis)
        )

        # Named entity overlap
        p_ents = set(_ENTITY_RE.findall(premise))
        h_ents = set(_ENTITY_RE.findall(hypothesis))
        if p_ents and h_ents:
            ent_overlap = len(p_ents & h_ents) / len(p_ents | h_ents)
        elif p_ents or h_ents:
            ent_overlap = 0.0
        else:
            ent_overlap = 0.5

        # Negation asymmetry
        p_neg = len(p_words & _NEGATION_WORDS)
        h_neg = len(h_words & _NEGATION_WORDS)
        neg_penalty = 0.3 if (p_neg == 0) != (h_neg == 0) else 0.0

        similarity = (
            0.4 * jaccard
            + 0.2 * len_ratio
            + 0.2 * ent_overlap
            + 0.2 * (1.0 - neg_penalty)
        )
        divergence = max(0.0, min(1.0, 1.0 - similarity))
        return divergence

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score multiple (premise, hypothesis) pairs."""
        return [self.score(p, h) for p, h in pairs]
