# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — claim-boundary gating for streaming coherence scores

"""Gate coherence re-scoring to complete claims during token streaming.

Scoring an *incomplete* sentence is unreliable: an NLI or RAG coherence model
asked whether "Water boils at 100 degrees" is entailed by the knowledge base
returns a low score for the dangling fragment, even though the finished
sentence is perfectly grounded. Feeding that per-token dip straight into a
hard-limit halt makes the guard stop on correct text mid-sentence — a false
halt. Measured on the false-halt benchmark, ungated per-token NLI scoring halts
~97% of factually-correct passages.

:class:`StreamingCoherenceGate` re-scores only when the accumulated text reaches
a claim boundary (sentence-ending punctuation) or a runaway word cap, and holds
the previous score between those points. The model therefore always judges a
complete claim, and the halt acts on claim-level coherence rather than on the
noise of a half-finished sentence. Short opening fragments pass on a grace score
until the first claim completes.

This gate changes *when* the score is taken, not the score itself — a genuinely
incoherent completed claim still scores low and still halts.
"""

from __future__ import annotations

from collections.abc import Callable

__all__ = ["StreamingCoherenceGate", "ends_claim"]

# Sentence / claim terminators after which a re-score is meaningful.
_CLAIM_ENDINGS: tuple[str, ...] = (".", "!", "?", ";", ":")


def ends_claim(text: str) -> bool:
    """True when *text* ends on sentence/claim punctuation (a claim boundary)."""
    return text.rstrip().endswith(_CLAIM_ENDINGS)


class StreamingCoherenceGate:
    """Re-score coherence only at claim boundaries; hold the score otherwise.

    Parameters
    ----------
    score_fn:
        ``score_fn(accumulated_text) -> float`` — the underlying coherence
        score for the full accumulated text (typically a closure over
        ``scorer.review(prompt, text)``).
    min_words:
        Below this accumulated word count the gate returns ``pass_score``
        rather than scoring an unreliable opening fragment.
    rescore_cap:
        Force a re-score after this many words without a claim boundary, so a
        run-on stream with no punctuation is still checked.
    pass_score:
        The score held before the first claim completes (a passing value, so an
        opening fragment is never the reason for a halt).
    """

    def __init__(
        self,
        score_fn: Callable[[str], float],
        *,
        min_words: int = 4,
        rescore_cap: int = 24,
        pass_score: float = 1.0,
    ) -> None:
        if min_words < 1:
            raise ValueError(f"min_words must be >= 1, got {min_words}")
        if rescore_cap < 1:
            raise ValueError(f"rescore_cap must be >= 1, got {rescore_cap}")
        self._score_fn = score_fn
        self._min_words = min_words
        self._rescore_cap = rescore_cap
        self._last_score = float(pass_score)
        self._scored_at_words = 0

    def reset(self, pass_score: float = 1.0) -> None:
        """Reset the held score and counter for a fresh stream."""
        self._last_score = float(pass_score)
        self._scored_at_words = 0

    @property
    def last_score(self) -> float:
        return self._last_score

    @staticmethod
    def _ends_claim(text: str) -> bool:
        return text.rstrip().endswith(_CLAIM_ENDINGS)

    def update(self, accumulated_text: str) -> float:
        """Return the coherence to act on for the current accumulated text.

        Re-scores when the text completes a claim (ends with sentence
        punctuation) or the word cap since the last score is reached; otherwise
        returns the held score.
        """
        word_count = len(accumulated_text.split())
        words_since = word_count - self._scored_at_words

        should_score = word_count >= self._min_words and (
            self._ends_claim(accumulated_text) or words_since >= self._rescore_cap
        )
        if not should_score:
            return self._last_score

        self._last_score = float(self._score_fn(accumulated_text))
        self._scored_at_words = word_count
        return self._last_score
