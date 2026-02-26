# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — LangChain Integration (Callback Handler)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
LangChain callback handler for coherence scoring.

Scores every LLM response through ``CoherenceScorer`` and optionally
raises on low coherence.  Add it to any LangChain chain with one line::

    from director_ai.integrations.langchain_callback import CoherenceCallbackHandler

    handler = CoherenceCallbackHandler(threshold=0.6)
    chain = LLMChain(llm=llm, callbacks=[handler])

Requires ``pip install director-ai[langchain]``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from director_ai.core.types import CoherenceScore

logger = logging.getLogger("DirectorAI.LangChain")

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    # Provide a usable base when langchain-core is not installed,
    # so the module can still be imported for type checking.
    class BaseCallbackHandler:  # type: ignore[no-redef]
        """Stub base when langchain-core is not installed."""

        def __init__(self, **kwargs: Any) -> None:
            pass


class CoherenceCallbackHandler(BaseCallbackHandler):
    """LangChain callback that scores LLM outputs for coherence.

    Parameters
    ----------
    threshold : float — coherence threshold (default 0.6).
    use_nli : bool — use DeBERTa NLI model for logical divergence.
    raise_on_failure : bool — raise ``CoherenceError`` on low coherence
        instead of just logging a warning.
    ground_truth_store : optional ``GroundTruthStore`` for factual checks.

    Attributes
    ----------
    last_score : the most recent ``CoherenceScore`` (or None).
    scores : list of all scores from the session.
    """

    def __init__(
        self,
        threshold: float = 0.6,
        use_nli: bool = False,
        raise_on_failure: bool = False,
        ground_truth_store: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        from director_ai.core.scorer import CoherenceScorer

        self.scorer = CoherenceScorer(
            threshold=threshold,
            use_nli=use_nli,
            ground_truth_store=ground_truth_store,
        )
        self.raise_on_failure = raise_on_failure
        self.last_score: CoherenceScore | None = None
        self.scores: list = []
        self._current_prompt: str = ""

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """Capture the prompt for use in scoring."""
        if prompts:
            self._current_prompt = prompts[0]

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Score the LLM response for coherence."""
        # Extract text from LangChain LLMResult
        text = ""
        try:
            generations = response.generations
            if generations and generations[0]:
                text = generations[0][0].text
        except (AttributeError, IndexError):
            logger.debug("Could not extract text from LLM response")
            return

        if not text:
            return

        approved, score = self.scorer.review(self._current_prompt, text)
        self.last_score = score
        self.scores.append(score)

        logger.info(
            "Coherence: %.4f (H_logic=%.2f, H_fact=%.2f) — %s",
            score.score,
            score.h_logical,
            score.h_factual,
            "APPROVED" if approved else "REJECTED",
        )

        if not approved and self.raise_on_failure:
            from director_ai.core.exceptions import CoherenceError

            raise CoherenceError(
                f"LLM output below coherence threshold: "
                f"{score.score:.4f} < {self.scorer.threshold}"
            )

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Capture input from chain for context."""
        # Try common input key names
        for key in ("input", "query", "question", "prompt"):
            if key in inputs:
                self._current_prompt = str(inputs[key])
                break
