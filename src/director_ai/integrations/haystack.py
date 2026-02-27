"""
Director-AI Haystack integration.

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

from typing import Any

from director_ai.core import CoherenceScorer, GroundTruthStore


class DirectorAIChecker:
    """Haystack 2.x component for coherence checking.

    Validates LLM responses against a knowledge base and annotates
    results with coherence scores and approval status.
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
            scores.append({
                "score": cs.score,
                "h_logical": cs.h_logical,
                "h_factual": cs.h_factual,
                "approved": approved,
                "warning": cs.warning,
            })
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
