"""
Director-AI LangChain integration.

Requires: pip install director-ai[langchain]

Usage::

    from director_ai.integrations.langchain import DirectorAIGuard

    guard = DirectorAIGuard(facts={"capital": "Paris is the capital of France."})
    chain = llm | guard
    result = chain.invoke("What is the capital of France?")
"""

from __future__ import annotations

from typing import Any

from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.core.types import CoherenceScore


class DirectorAIGuard:
    """LangChain-compatible output guard.

    Works as a Runnable: pipe it after your LLM in a chain.
    Raises ``HallucinationError`` when coherence is below threshold.

    Parameters
    ----------
    facts : dict[str, str] | None — key-value facts for the knowledge base.
    store : GroundTruthStore | None — pre-built store (overrides facts).
    threshold : float — minimum coherence to pass.
    use_nli : bool | None — NLI mode (None=auto-detect).
    raise_on_fail : bool — if True, raise on failure; if False, return
        the result dict with ``approved=False``.
    """

    def __init__(
        self,
        facts: dict[str, str] | None = None,
        store: GroundTruthStore | None = None,
        threshold: float = 0.6,
        use_nli: bool | None = None,
        raise_on_fail: bool = False,
    ):
        self.store = store or GroundTruthStore()
        if facts:
            for k, v in facts.items():
                self.store.add(k, v)
        self.scorer = CoherenceScorer(
            threshold=threshold,
            ground_truth_store=self.store,
            use_nli=use_nli,
        )
        self.raise_on_fail = raise_on_fail

    def check(self, query: str, response: str) -> dict[str, Any]:
        """Score a response against the knowledge base.

        Returns dict with keys: approved, score, h_logical, h_factual,
        response, coherence_score.
        """
        approved, cs = self.scorer.review(query, response)
        result = {
            "approved": approved,
            "score": cs.score,
            "h_logical": cs.h_logical,
            "h_factual": cs.h_factual,
            "response": response,
            "coherence_score": cs,
        }
        if not approved and self.raise_on_fail:
            raise HallucinationError(query, response, cs)
        return result

    async def acheck(self, query: str, response: str) -> dict[str, Any]:
        """Async version of check()."""
        approved, cs = await self.scorer.areview(query, response)
        result = {
            "approved": approved,
            "score": cs.score,
            "h_logical": cs.h_logical,
            "h_factual": cs.h_factual,
            "response": response,
            "coherence_score": cs,
        }
        if not approved and self.raise_on_fail:
            raise HallucinationError(query, response, cs)
        return result

    def invoke(self, input: Any, **kwargs) -> dict[str, Any]:
        """LangChain Runnable interface.

        Accepts str or dict with 'query' and 'response' keys.
        When receiving a plain string (typical LLM output), uses
        kwargs.get('query', '') as the query.
        """
        if isinstance(input, dict):
            query = input.get("query", input.get("input", ""))
            response = input.get("response", input.get("output", ""))
        else:
            query = kwargs.get("query", "")
            response = str(input)
        return self.check(query, response)

    async def ainvoke(self, input: Any, **kwargs) -> dict[str, Any]:
        """Async LangChain Runnable interface."""
        if isinstance(input, dict):
            query = input.get("query", input.get("input", ""))
            response = input.get("response", input.get("output", ""))
        else:
            query = kwargs.get("query", "")
            response = str(input)
        return await self.acheck(query, response)


class HallucinationError(Exception):
    """Raised when DirectorAIGuard detects hallucination."""

    def __init__(
        self, query: str, response: str, score: CoherenceScore
    ):
        self.query = query
        self.response = response
        self.score = score
        super().__init__(
            f"Hallucination detected (coherence={score.score:.3f}): "
            f"{response[:100]}"
        )
