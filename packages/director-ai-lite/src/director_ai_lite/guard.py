# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Director-Lite standalone streaming halt
"""The streaming-halt guard, standalone and dependency-free.

Director-Lite ships the one differentiator — stopping a token stream *before* a
hallucination finishes generating — as a small, model-free guard that installs
with zero heavy dependencies and no ``director-ai`` requirement. The grounding
heuristic and the coherence calibration match the full package's no-model path,
so upgrading to ``director-ai`` (NLI/RAG scoring) does not change the call site:

    from director_ai_lite import guard

    result = guard(token_stream, facts={"fr": "Paris is the capital of France."},
                   prompt="What is the capital of France?")
    print(result.output, result.halted, result.halt_reason)

For model-backed accuracy install the full package (``pip install
director-ai-lite[full]``) and pass its scorer to :class:`StreamGuard`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from ._coherence import combine_weighted_coherence

_WORD = re.compile(r"[a-z0-9]+")
# fmt: off
_STOP = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "that", "the", "to", "was", "were", "with", "this",
    "these", "those", "which", "who", "what", "when", "where", "why", "how",
})
# fmt: on


def _content_words(text: str) -> set[str]:
    """Return lowercased content words (stop-words and short tokens removed)."""
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def _factual_divergence(accumulated: str, fact_words: set[str]) -> float:
    """Return factual divergence in [0, 1] for ``accumulated`` against the facts.

    Grounding support is the share of the accumulated content words that are
    attested in the supplied facts; divergence is ``1 - support``. With no facts
    there is nothing to ground against, so the score is the neutral ``0.5``.
    """
    if not fact_words:
        return 0.5
    words = _content_words(accumulated)
    if not words:
        return 0.0
    grounded = len(words & fact_words)
    return 1.0 - grounded / len(words)


@dataclass
class StreamResult:
    """Outcome of a guarded token stream."""

    tokens: list[str] = field(default_factory=list)
    coherence_history: list[float] = field(default_factory=list)
    halted: bool = False
    halt_index: int = -1
    halt_reason: str = ""

    @property
    def output(self) -> str:
        """Surviving text: the tokens accepted before any hard halt."""
        end = (
            self.halt_index
            if self.halted and self.halt_index >= 0
            else len(self.tokens)
        )
        return "".join(self.tokens[:end])


class StreamGuard:
    """Zero-config, dependency-free streaming-halt guard.

    Parameters
    ----------
    facts:
        Optional mapping of key → grounded statement scored against the stream.
    threshold:
        Coherence floor in [0, 1]; the stream hard-halts on the first token whose
        accumulated coherence drops below it.
    scorer:
        Optional ``review(prompt, text) -> (_, score)`` object (e.g. the full
        package's NLI scorer) that overrides the heuristic for higher accuracy.
    """

    def __init__(
        self,
        facts: Mapping[str, str] | None = None,
        *,
        threshold: float = 0.5,
        scorer: object | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self._facts = dict(facts or {})
        self._fact_words = _content_words(" ".join(self._facts.values()))
        self._threshold = threshold
        self._scorer = scorer

    def _coherence(self, prompt: str, accumulated: str) -> float:
        if self._scorer is not None:
            _, score = self._scorer.review(prompt, accumulated)  # type: ignore[attr-defined]
            return float(score.score)
        divergence = _factual_divergence(accumulated, self._fact_words)
        return combine_weighted_coherence(
            h_logic=0.0,
            h_factual=divergence,
            w_logic=0.0,
            w_fact=1.0,
            nli_available=False,
            evidence_present=bool(self._fact_words),
            dialogue_route=False,
        )

    def guard(self, tokens: Iterable[str], prompt: str = "") -> StreamResult:
        """Stream ``tokens`` under oversight; halt before an ungrounded completion."""
        result = StreamResult()
        accumulated = ""
        for index, token in enumerate(tokens):
            result.tokens.append(token)
            accumulated += token
            coherence = self._coherence(prompt, accumulated)
            result.coherence_history.append(coherence)
            if coherence < self._threshold:
                result.halted = True
                result.halt_index = index
                result.halt_reason = (
                    f"coherence {coherence:.2f} below threshold {self._threshold:.2f}"
                )
                break
        return result

    def safe_text(self, tokens: Iterable[str], prompt: str = "") -> str:
        """Return only the surviving (non-halted) output text."""
        return self.guard(tokens, prompt=prompt).output


def streaming_guard(
    tokens: Iterable[str],
    *,
    facts: Mapping[str, str] | None = None,
    prompt: str = "",
    threshold: float = 0.5,
    scorer: object | None = None,
) -> StreamResult:
    """One-call streaming halt: guard ``tokens`` and return the result."""
    return StreamGuard(facts, threshold=threshold, scorer=scorer).guard(
        tokens, prompt=prompt
    )


def guard(
    tokens: Iterable[str],
    *,
    facts: Mapping[str, str] | None = None,
    prompt: str = "",
    threshold: float = 0.5,
    scorer: object | None = None,
) -> StreamResult:
    """Three-line front door to the streaming halt (alias of :func:`streaming_guard`)."""
    return streaming_guard(
        tokens, facts=facts, prompt=prompt, threshold=threshold, scorer=scorer
    )


__all__ = ["StreamGuard", "StreamResult", "guard", "streaming_guard"]
