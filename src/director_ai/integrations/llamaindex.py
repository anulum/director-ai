"""
Director-AI LlamaIndex integration.

Requires: pip install director-ai[llamaindex]

Usage::

    from director_ai.integrations.llamaindex import DirectorAIPostprocessor

    postprocessor = DirectorAIPostprocessor(
        facts={"capital": "Paris is the capital of France."}
    )
    # Use as a node postprocessor in a query engine
    query_engine = index.as_query_engine(
        node_postprocessors=[postprocessor]
    )
"""

from __future__ import annotations

from typing import Any

from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.core.types import CoherenceScore


class DirectorAIPostprocessor:
    """LlamaIndex-compatible node postprocessor.

    Scores each retrieved node's text for coherence against the
    Director-AI knowledge base. Nodes below threshold are filtered out.

    Can also be used standalone via ``check()`` for response validation.

    Parameters
    ----------
    facts : dict[str, str] | None — key-value facts for the knowledge base.
    store : GroundTruthStore | None — pre-built store (overrides facts).
    threshold : float — minimum coherence to keep a node.
    use_nli : bool | None — NLI mode (None=auto-detect).
    """

    def __init__(
        self,
        facts: dict[str, str] | None = None,
        store: GroundTruthStore | None = None,
        threshold: float = 0.6,
        use_nli: bool | None = None,
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
        self.threshold = threshold

    def check(self, query: str, response: str) -> dict[str, Any]:
        """Score a response against the knowledge base."""
        approved, cs = self.scorer.review(query, response)
        return {
            "approved": approved,
            "score": cs.score,
            "h_logical": cs.h_logical,
            "h_factual": cs.h_factual,
            "response": response,
            "coherence_score": cs,
        }

    def postprocess_nodes(self, nodes, query_bundle=None, **kwargs):
        """LlamaIndex NodePostprocessor interface.

        Filters out nodes whose text scores below threshold.
        Adds coherence metadata to surviving nodes.
        """
        query_str = ""
        if query_bundle is not None:
            query_str = getattr(query_bundle, "query_str", str(query_bundle))

        filtered = []
        for node in nodes:
            text = getattr(node, "text", "") or str(node)
            approved, cs = self.scorer.review(query_str, text)
            if approved:
                if hasattr(node, "metadata") and isinstance(node.metadata, dict):
                    node.metadata["director_ai_score"] = cs.score
                    node.metadata["director_ai_approved"] = True
                filtered.append(node)
        return filtered

    def validate_response(
        self, query: str, response: str
    ) -> tuple[bool, CoherenceScore]:
        """Validate a final response (not node-level)."""
        return self.scorer.review(query, response)
