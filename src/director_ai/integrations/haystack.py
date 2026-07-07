# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Director-AI Haystack integration.

Requires: pip install director-ai[haystack]

Usage::

    from director_ai.integrations.haystack import DirectorAIChecker
    from haystack import Pipeline

    pipeline = Pipeline()
    pipeline.add_component("checker", DirectorAIChecker(
        facts={"capital": "Paris is the capital of France."}
    ))
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from director_ai.core import CoherenceScorer, GroundTruthStore

if TYPE_CHECKING:
    # Present the real Haystack decorator to the type checker so this component
    # is checked against the Haystack contract.
    from haystack import component
else:
    # Runtime resolution: use the real ``@component`` decorator when haystack-ai
    # is installed (so ``pipeline.add_component`` accepts this class), else fall
    # back to a no-op that keeps the module importable without the extra.
    try:
        from haystack import component
    except ImportError:

        class _ComponentFallback:
            """No-op stand-in for ``haystack.component`` when it is absent."""

            def __call__(self, cls: type) -> type:
                return cls

            def output_types(self, **types: Any) -> Callable[[Any], Any]:
                def _decorate(func: Any) -> Any:
                    return func

                return _decorate

        component = _ComponentFallback()


@component
class DirectorAIChecker:
    """Haystack 2.x component for coherence checking.

    Decorated with ``@component`` so it registers input/output sockets and can
    be added to a ``haystack.Pipeline`` via ``add_component``. Validates LLM
    responses against a knowledge base and annotates results with coherence
    scores and approval status.
    """

    def __init__(
        self,
        facts: dict[str, str] | None = None,
        store: GroundTruthStore | None = None,
        threshold: float = 0.6,
        use_nli: bool | None = None,
        filter_rejected: bool = False,
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
        self.filter_rejected = filter_rejected

    @component.output_types(
        replies=list[str],
        scores=list[dict[str, Any]],
        approved=list[bool],
    )
    def run(
        self,
        query: str = "",
        replies: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Haystack 2.x component interface.

        Parameters
        ----------
        query : the user question.
        replies : list of LLM response strings.

        Returns
        -------
        dict with "replies" (filtered if configured), "scores", "approved" lists.

        """
        if not replies:
            return {"replies": [], "scores": [], "approved": []}

        scored_replies = []
        scores = []
        approved_list = []

        for reply in replies:
            approved, cs = self.scorer.review(query, reply)
            scores.append(
                {
                    "score": cs.score,
                    "h_logical": cs.h_logical,
                    "h_factual": cs.h_factual,
                    "approved": approved,
                    "warning": cs.warning,
                },
            )
            approved_list.append(approved)
            if not self.filter_rejected or approved:
                scored_replies.append(reply)

        return {
            "replies": scored_replies,
            "scores": scores,
            "approved": approved_list,
        }

    def to_dict(self) -> dict[str, Any]:
        """Haystack serialization."""
        return {
            "type": "director_ai.integrations.haystack.DirectorAIChecker",
            "init_parameters": {
                "threshold": self.scorer.threshold,
                "filter_rejected": self.filter_rejected,
            },
        }
