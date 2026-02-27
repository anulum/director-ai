"""
Director-AI CrewAI integration.

Requires: pip install director-ai[crewai]

Usage::

    from director_ai.integrations.crewai import DirectorAITool
    from crewai import Agent

    guard_tool = DirectorAITool(facts={"company": "Founded in 2020"})
    agent = Agent(tools=[guard_tool], ...)
"""

from __future__ import annotations

from typing import Any

from director_ai.core import CoherenceScorer, GroundTruthStore


class DirectorAITool:
    """CrewAI-compatible tool for fact-checking agent outputs.

    Can be added to any CrewAI Agent's tool list. The agent can invoke
    it to verify claims before including them in its final answer.

    Implements the CrewAI tool protocol (name, description, _run).
    """

    name: str = "director_ai_fact_check"
    description: str = (
        "Verify a claim against a knowledge base. "
        "Input: 'query | claim' separated by pipe. "
        "Returns coherence score and approval status."
    )

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

    def _run(self, input_text: str) -> str:
        """CrewAI tool execution interface."""
        if "|" in input_text:
            query, claim = input_text.split("|", 1)
        else:
            query = ""
            claim = input_text

        approved, cs = self.scorer.review(query.strip(), claim.strip())

        status = "APPROVED" if approved else "REJECTED"
        warning = " (low confidence)" if cs.warning else ""
        return (
            f"[{status}{warning}] Coherence: {cs.score:.3f} "
            f"(logical: {cs.h_logical:.3f}, factual: {cs.h_factual:.3f})"
        )

    def run(self, input_text: str) -> str:
        """Alias for _run (some CrewAI versions use run())."""
        return self._run(input_text)

    def check(self, query: str, response: str) -> dict[str, Any]:
        """Direct API for programmatic use."""
        approved, cs = self.scorer.review(query, response)
        return {
            "approved": approved,
            "score": cs.score,
            "h_logical": cs.h_logical,
            "h_factual": cs.h_factual,
            "warning": cs.warning,
        }
