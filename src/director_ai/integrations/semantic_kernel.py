# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Microsoft Semantic Kernel Integration
"""Director-AI filter for Microsoft Semantic Kernel.

Requires: pip install semantic-kernel director-ai

Usage::

    from semantic_kernel import Kernel
    from director_ai.integrations.semantic_kernel import DirectorAIFilter

    kernel = Kernel()
    kernel.add_filter("function_invocation", DirectorAIFilter(
        facts={"pricing": "Team plan costs $19/user/month."},
    ))
"""

from __future__ import annotations

from typing import Any

from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.core.exceptions import HallucinationError


class DirectorAIFilter:
    """Semantic Kernel function invocation filter.

    Checks LLM output coherence after each function invocation.
    Attach via ``kernel.add_filter("function_invocation", filter)``.

    Parameters
    ----------
    facts : dict[str, str] | None — key-value facts for the knowledge base.
    store : GroundTruthStore | None — pre-built store (overrides facts).
    threshold : float — minimum coherence to pass.
    use_nli : bool | None — NLI mode (None=auto-detect).
    raise_on_fail : bool — raise HallucinationError on failure.
    """

    def __init__(
        self,
        facts: dict[str, str] | None = None,
        store: GroundTruthStore | None = None,
        threshold: float = 0.5,
        use_nli: bool | None = None,
        raise_on_fail: bool = True,
    ) -> None:
        self._store = store or GroundTruthStore()
        if facts and store is None:
            for k, v in facts.items():
                self._store.add(k, v)
        self._scorer = CoherenceScorer(
            threshold=threshold,
            ground_truth_store=self._store,
            use_nli=use_nli,
        )
        self._raise = raise_on_fail

    async def __call__(self, context: Any, next_fn: Any) -> None:
        """Filter hook: runs after function invocation, checks output."""
        await next_fn(context)

        result = str(context.result) if context.result else ""
        if not result:
            return

        prompt = ""
        if hasattr(context, "arguments") and context.arguments:
            prompt = str(context.arguments.get("input", ""))

        approved, score = self._scorer.review(prompt, result)

        if not approved:
            if self._raise:
                raise HallucinationError(prompt, result, score)
            context.result = {
                "approved": False,
                "score": score.score,
                "original": result,
            }

    @property
    def scorer(self) -> CoherenceScorer:
        return self._scorer
