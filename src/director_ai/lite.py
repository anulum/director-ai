# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Lite: streaming-halt in three lines
"""Director-Lite — the token-level streaming halt, with zero configuration.

The full Director-AI surface is large; this module is the 30-second front door
to its one differentiator: stopping a token stream *before* a hallucination
finishes generating. No model download, no knowledge-base wiring, no config.

    from director_ai.lite import StreamGuard

    guard = StreamGuard(facts={"capital": "Paris is the capital of France."})
    result = guard.guard(token_stream, prompt="What is the capital of France?")
    print(result.output, result.halted, result.halt_reason)

It wraps the same Apache-2.0 ``StreamingKernel`` and ``CoherenceScorer`` the full
product uses, in heuristic (no-NLI) mode by default so it runs anywhere. Pass a
configured ``scorer`` to upgrade to model-backed NLI scoring without changing the
call site.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from director_ai.core import StreamSession


class StreamGuard:
    """Zero-config streaming-halt guard.

    Parameters
    ----------
    facts : optional mapping of key → grounded statement used for factual scoring.
    threshold : coherence floor; the stream halts when coherence drops below it.
    window_size : sliding-window length for trend-based halting.
    scorer : optional pre-built scorer (e.g. model-backed NLI) — overrides the
        default heuristic scorer for higher accuracy.
    """

    def __init__(
        self,
        facts: Mapping[str, str] | None = None,
        *,
        threshold: float = 0.5,
        window_size: int = 10,
        scorer: Any = None,
    ) -> None:
        self._facts = dict(facts or {})
        self._threshold = threshold
        self._window_size = window_size
        self._scorer = scorer

    def _ensure_scorer(self) -> Any:
        if self._scorer is None:
            from director_ai.core import CoherenceScorer, GroundTruthStore

            store = GroundTruthStore()
            for key, value in self._facts.items():
                store.add(key, value)
            self._scorer = CoherenceScorer(
                threshold=self._threshold,
                ground_truth_store=store,
                use_nli=False,
            )
        return self._scorer

    def guard(self, tokens: Iterable[str], prompt: str = "") -> StreamSession:
        """Stream ``tokens`` under oversight; halt before an incoherent completion.

        Returns the :class:`StreamSession`: ``output`` is the surviving text
        (hard-halted tokens removed), ``halted``/``halt_reason`` say whether and
        why the stream was stopped.
        """
        from director_ai.core import StreamingKernel

        scorer = self._ensure_scorer()

        def _coherence(accumulated: str) -> float:
            _, score = scorer.review(prompt, accumulated)
            return float(score.score)

        kernel = StreamingKernel(
            hard_limit=self._threshold, window_size=self._window_size
        )
        return kernel.stream_tokens(iter(tokens), _coherence, prompt=prompt)

    def safe_text(self, tokens: Iterable[str], prompt: str = "") -> str:
        """Convenience: return only the surviving (non-halted) output text."""
        return self.guard(tokens, prompt=prompt).output


def streaming_guard(
    tokens: Iterable[str],
    *,
    facts: Mapping[str, str] | None = None,
    prompt: str = "",
    threshold: float = 0.5,
    scorer: Any = None,
) -> StreamSession:
    """One-call streaming halt: guard ``tokens`` and return the session.

    Equivalent to ``StreamGuard(facts, threshold=...).guard(tokens, prompt)`` —
    the shortest path to the streaming halt.
    """
    return StreamGuard(facts, threshold=threshold, scorer=scorer).guard(
        tokens, prompt=prompt
    )


__all__ = ["StreamGuard", "streaming_guard"]
