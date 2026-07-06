# SPDX-License-Identifier: Apache-2.0
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

from typing import TypedDict

from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.core.exceptions import HallucinationError
from director_ai.core.types import CoherenceScore, EvidenceChunk, ScoringEvidence


class CoherenceCheckResult(TypedDict):
    """JSON-safe result returned by the DSPy and Instructor adapter."""

    approved: bool
    score: float
    evidence: dict[str, object]


def coherence_check(
    response: str,
    prompt: str = "",
    facts: dict[str, str] | None = None,
    store: GroundTruthStore | None = None,
    threshold: float = 0.5,
    use_nli: bool | None = None,
) -> CoherenceCheckResult:
    """Check response coherence and return a JSON-safe result.

    Parameters
    ----------
    response : str
        LLM output to verify.
    prompt : str
        Original prompt or question for context.
    facts : dict[str, str] | None
        Key-value facts for grounding. Ignored when ``store`` is supplied.
    store : GroundTruthStore | None
        Pre-built store that overrides ``facts``.
    threshold : float
        Minimum coherence score required to approve the response.
    use_nli : bool | None
        NLI mode forwarded to :class:`~director_ai.core.CoherenceScorer`.

    Returns
    -------
    CoherenceCheckResult
        JSON-safe approval flag, score, and evidence payload.

    Raises
    ------
    ValueError
        If a supplied fact key is blank or the scorer rejects its configuration.
    """
    gt = _build_ground_truth_store(facts=facts, store=store)
    scorer = CoherenceScorer(
        threshold=threshold,
        ground_truth_store=gt,
        use_nli=use_nli,
    )
    prompt_context = prompt.strip() or response
    approved, cs = scorer.review(prompt_context, response)
    return {
        "approved": approved,
        "score": cs.score,
        "evidence": _coherence_evidence_payload(cs),
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

    Parameters
    ----------
    response : str
        LLM output to verify.
    prompt : str
        Original prompt or question for context.
    facts : dict[str, str] | None
        Key-value facts for grounding. Ignored when ``store`` is supplied.
    store : GroundTruthStore | None
        Pre-built store that overrides ``facts``.
    threshold : float
        Minimum coherence score required to approve the response.
    use_nli : bool | None
        NLI mode forwarded to :class:`~director_ai.core.CoherenceScorer`.
    message : str
        Reserved for DSPy-compatible call sites; the structured
        :class:`HallucinationError` carries the rejection details.

    Raises
    ------
    ValueError
        If a supplied fact key is blank or the scorer rejects its configuration.
    HallucinationError
        If coherence is below ``threshold``.
    """
    _ = message
    gt = _build_ground_truth_store(facts=facts, store=store)
    scorer = CoherenceScorer(
        threshold=threshold,
        ground_truth_store=gt,
        use_nli=use_nli,
    )
    prompt_context = prompt.strip() or response
    approved, cs = scorer.review(prompt_context, response)
    if not approved:
        raise HallucinationError(prompt_context, response, cs)


def _build_ground_truth_store(
    *,
    facts: dict[str, str] | None,
    store: GroundTruthStore | None,
) -> GroundTruthStore:
    gt = store or GroundTruthStore()
    if facts and store is None:
        for key, value in facts.items():
            gt.add(_fact_key(key), value)
    return gt


def _fact_key(key: str) -> str:
    stripped = key.strip()
    if not stripped:
        raise ValueError("fact key must not be blank")
    return stripped


def _coherence_evidence_payload(cs: CoherenceScore) -> dict[str, object]:
    payload: dict[str, object] = {
        "h_logical": cs.h_logical,
        "h_factual": cs.h_factual,
    }
    evidence = cs.evidence
    if evidence is not None:
        payload.update(_scoring_evidence_payload(evidence))
    return payload


def _scoring_evidence_payload(evidence: ScoringEvidence) -> dict[str, object]:
    return {
        "chunks": [_evidence_chunk_payload(chunk) for chunk in evidence.chunks],
        "nli_premise": evidence.nli_premise,
        "nli_hypothesis": evidence.nli_hypothesis,
        "nli_score": evidence.nli_score,
        "chunk_scores": evidence.chunk_scores,
        "premise_chunk_count": evidence.premise_chunk_count,
        "hypothesis_chunk_count": evidence.hypothesis_chunk_count,
        "claim_coverage": evidence.claim_coverage,
        "per_claim_divergences": evidence.per_claim_divergences,
        "claims": evidence.claims,
        "token_count": evidence.token_count,
        "estimated_cost_usd": evidence.estimated_cost_usd,
    }


def _evidence_chunk_payload(chunk: EvidenceChunk) -> dict[str, object]:
    return {
        "text": chunk.text,
        "distance": chunk.distance,
        "source": chunk.source,
    }
