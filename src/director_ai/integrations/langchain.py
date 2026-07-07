# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Director-AI LangChain integration.

Requires: pip install director-ai[langchain]

Usage::

    from director_ai.integrations.langchain import DirectorAIGuard

    guard = DirectorAIGuard(facts={"capital": "Paris is the capital of France."})
    chain = llm | guard
    result = chain.invoke("What is the capital of France?")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.core.exceptions import HallucinationError

if TYPE_CHECKING:
    # Present the real LangChain base to the type checker so this guard is
    # checked against the Runnable contract.
    from langchain_core.runnables import Runnable
else:
    # Runtime resolution: subclass the real Runnable when langchain-core is
    # installed (so ``llm | guard`` composes), else fall back to a minimal base
    # that keeps the standalone ``.check()`` API importable without the extra.
    try:
        from langchain_core.runnables import Runnable
    except ImportError:

        class Runnable:
            """Fallback base used when langchain-core is not installed."""

            def __class_getitem__(cls, _item: Any) -> type:
                """Return this fallback class for generic subscription."""
                # Mimic the real generic so ``Runnable[Any, Any]`` resolves.
                return cls


class DirectorAIGuard(
    Runnable[Any, Any]  # type: ignore[misc,unused-ignore] # LangChain Runnable base may be Any when optional stubs are absent.
):
    """LangChain Runnable output guard.

    Subclasses ``langchain_core.runnables.Runnable`` so it composes directly in
    a chain — ``llm | guard`` — and exposes the full ``invoke``/``ainvoke``
    (plus inherited ``batch``/``stream``) surface.
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

    def invoke(
        self,
        input: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """LangChain Runnable interface.

        Accepts str or dict with 'query' and 'response' keys.
        When receiving a plain string (typical LLM output), uses
        kwargs.get('query', '') as the query. ``config`` is accepted for the
        Runnable contract (a chain passes it positionally) and is unused.
        """
        if isinstance(input, dict):
            query = input.get("query", input.get("input", ""))
            response = input.get("response", input.get("output", ""))
        else:
            response = str(input)
            query = kwargs.get("query", response)
        return self.check(str(query), str(response))

    async def ainvoke(
        self,
        input: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async LangChain Runnable interface."""
        if isinstance(input, dict):
            query = input.get("query", input.get("input", ""))
            response = input.get("response", input.get("output", ""))
        else:
            response = str(input)
            query = kwargs.get("query", response)
        return await self.acheck(str(query), str(response))
