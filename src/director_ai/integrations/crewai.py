# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Director-AI CrewAI integration.

Requires: pip install director-ai[crewai]

Usage::

    from director_ai.integrations.crewai import DirectorAITool
    from crewai import Agent

    guard_tool = DirectorAITool(facts={"company": "Founded in 2020"})
    agent = Agent(tools=[guard_tool], ...)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from director_ai.core import CoherenceScorer, GroundTruthStore

if TYPE_CHECKING:
    # Present the real CrewAI base to the type checker so this tool is checked
    # against the BaseTool contract.
    from crewai.tools import BaseTool
else:
    # Runtime resolution: subclass the real BaseTool when crewai is installed
    # (so ``Agent(tools=[tool])`` accepts it), else fall back to a minimal base
    # that keeps the standalone ``.check()``/``.run()`` API usable without it.
    try:
        from crewai.tools import BaseTool
    except ImportError:

        class BaseTool:
            """Fallback base used when crewai is not installed."""

            name: str = ""
            description: str = ""

            def __init__(self, **kwargs: Any) -> None:
                pass

            def run(self, *args: Any, **kwargs: Any) -> Any:
                """Mirror crewai BaseTool.run: forward to the concrete _run."""
                return self._run(*args, **kwargs)


class DirectorAITool(
    BaseTool  # type: ignore[misc,unused-ignore] # crewai BaseTool may be Any when optional stubs are absent.
):
    """CrewAI ``BaseTool`` for fact-checking agent outputs.

    Subclasses ``crewai.tools.BaseTool`` so it can be added directly to a
    CrewAI ``Agent(tools=[...])`` list; the agent invokes it to verify claims
    before including them in its final answer. The inherited ``run`` drives the
    concrete ``_run``.
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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        resolved_store = store or GroundTruthStore()
        if facts:
            for k, v in facts.items():
                resolved_store.add(k, v)
        # Stored as private attrs so they coexist with crewai's Pydantic
        # BaseTool (undeclared public attributes are rejected there).
        self._store = resolved_store
        self._scorer = CoherenceScorer(
            threshold=threshold,
            ground_truth_store=resolved_store,
            use_nli=use_nli,
        )

    @property
    def scorer(self) -> CoherenceScorer:
        """The underlying coherence scorer (read-only)."""
        return self._scorer

    @property
    def store(self) -> GroundTruthStore:
        """The underlying knowledge store (read-only)."""
        return self._store

    def _run(self, input_text: str) -> str:
        """CrewAI tool execution interface."""
        if "|" in input_text:
            query, claim = input_text.split("|", 1)
        else:
            claim = input_text
            query = claim

        approved, cs = self._scorer.review(query.strip(), claim.strip())

        status = "APPROVED" if approved else "REJECTED"
        warning = " (low confidence)" if cs.warning else ""
        return (
            f"[{status}{warning}] Coherence: {cs.score:.3f} "
            f"(logical: {cs.h_logical:.3f}, factual: {cs.h_factual:.3f})"
        )

    def check(self, query: str, response: str) -> dict[str, Any]:
        """Direct API for programmatic use."""
        approved, cs = self._scorer.review(query, response)
        return {
            "approved": approved,
            "score": cs.score,
            "h_logical": cs.h_logical,
            "h_factual": cs.h_factual,
            "warning": cs.warning,
        }
