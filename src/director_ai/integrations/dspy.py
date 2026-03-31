# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — DSPy / Instructor Integration
"""Director-AI assertion module for DSPy and Instructor.

Works as a DSPy assertion or a standalone validation function
for Instructor-style structured output pipelines.

Requires: pip install director-ai

Usage with DSPy::

    import dspy
    from director_ai.integrations.dspy import director_assert

    class FactCheckedQA(dspy.Module):
        def forward(self, question):
            answer = self.generate(question=question)
            director_assert(
                answer.response,
                facts={"pricing": "Team plan costs $19/user/month."},
            )
            return answer

Usage with Instructor / standalone::

    from director_ai.integrations.dspy import coherence_check

    result = coherence_check(
        response="The team plan costs $29/month.",
        facts={"pricing": "Team plan costs $19/user/month."},
    )
    if not result["approved"]:
        print(f"Hallucination detected: {result['score']:.3f}")
"""

from __future__ import annotations

from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.core.exceptions import HallucinationError


def coherence_check(
    response: str,
    prompt: str = "",
    facts: dict[str, str] | None = None,
    store: GroundTruthStore | None = None,
    threshold: float = 0.5,
    use_nli: bool | None = None,
) -> dict:
    """Check response coherence and return a result dict.

    Parameters
    ----------
    response : str — LLM output to verify.
    prompt : str — original prompt/question for context.
    facts : dict[str, str] — key-value facts for grounding.
    store : GroundTruthStore — pre-built store (overrides facts).
    threshold : float — minimum coherence to pass.
    use_nli : bool | None — NLI mode.

    Returns
    -------
    dict with keys: approved, score, evidence.
    """
    gt = store or GroundTruthStore()
    if facts and store is None:
        for k, v in facts.items():
            gt.add(k, v)

    scorer = CoherenceScorer(
        threshold=threshold,
        ground_truth_store=gt,
        use_nli=use_nli,
    )
    approved, cs = scorer.review(prompt, response)
    return {
        "approved": approved,
        "score": cs.score,
        "evidence": cs.evidence,
    }


def director_assert(
    response: str,
    prompt: str = "",
    facts: dict[str, str] | None = None,
    store: GroundTruthStore | None = None,
    threshold: float = 0.5,
    use_nli: bool | None = None,
    message: str = "",
) -> None:
    """DSPy-compatible assertion: raises on hallucination.

    Use inside a ``dspy.Module.forward()`` to enforce factual grounding.
    Also works standalone for Instructor or any pipeline.

    Raises
    ------
    HallucinationError if coherence is below threshold.
    """
    gt = store or GroundTruthStore()
    if facts and store is None:
        for k, v in facts.items():
            gt.add(k, v)

    scorer = CoherenceScorer(
        threshold=threshold,
        ground_truth_store=gt,
        use_nli=use_nli,
    )
    approved, cs = scorer.review(prompt, response)
    if not approved:
        raise HallucinationError(prompt, response, cs)
