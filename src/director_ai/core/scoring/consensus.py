# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Cross-model consensus scoring — multi-model factual agreement.

Queries the same prompt to multiple models, then scores pairwise
factual agreement via NLI. High disagreement → low confidence.

Usage::

    scorer = ConsensusScorer(
        models=["gpt-4o", "claude-3.5-sonnet", "gemini-1.5-pro"],
        generate_fn=my_generate_function,
    )
    result = scorer.score("What is the capital of France?")
    print(result.agreement_score)  # 0.95 = high consensus
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ConsensusScorer", "ConsensusResult", "ModelResponse", "PairwiseAgreement"]


@dataclass
class ModelResponse:
    """Response from a single model."""

    model: str
    response: str


@dataclass
class PairwiseAgreement:
    """Agreement between two model responses."""

    model_a: str
    model_b: str
    divergence: float  # 0 = agree, 1 = contradict
    agreed: bool


@dataclass
class ConsensusResult:
    """Result of cross-model consensus check."""

    responses: list[ModelResponse]
    pairs: list[PairwiseAgreement] = field(default_factory=list)
    agreement_score: float = 1.0  # 0 = complete disagreement, 1 = consensus
    lowest_pair_agreement: float = 1.0
    disagreement_pairs: list[PairwiseAgreement] = field(default_factory=list)
    num_models: int = 0

    @property
    def has_consensus(self) -> bool:
        return self.agreement_score > 0.7


class ConsensusScorer:
    """Score factual agreement across multiple LLM responses.

    Parameters
    ----------
    models : list[str]
        Model identifiers to query.
    generate_fn : callable
        Function(prompt: str, model: str) -> str. Generates a response.
    score_fn : callable | None
        Function(text_a: str, text_b: str) -> float (divergence 0-1).
        If None, uses Jaccard word overlap heuristic.
    agreement_threshold : float
        Divergence below which a pair is considered in agreement.
    """

    def __init__(
        self,
        models: list[str],
        generate_fn=None,
        score_fn=None,
        agreement_threshold: float = 0.5,
    ):
        if len(models) < 2:
            raise ValueError("Need at least 2 models for consensus scoring")
        self._models = models
        self._generate = generate_fn
        self._score_fn = score_fn or self._jaccard_divergence
        self._threshold = agreement_threshold

    def score(self, prompt: str) -> ConsensusResult:
        """Query all models and compute pairwise agreement.

        Parameters
        ----------
        prompt : str
            The prompt to send to all models.

        Returns
        -------
        ConsensusResult
            Pairwise agreement scores and overall consensus.
        """
        responses = self._gather_responses(prompt)
        return self.score_responses(responses)

    def score_responses(self, responses: list[ModelResponse]) -> ConsensusResult:
        """Score agreement across pre-generated responses.

        Useful when you already have responses from multiple models
        and want to check consensus without re-generating.
        """
        if len(responses) < 2:
            return ConsensusResult(
                responses=responses,
                agreement_score=1.0,
                num_models=len(responses),
            )

        pairs: list[PairwiseAgreement] = []
        disagreements: list[PairwiseAgreement] = []

        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                div = self._score_fn(responses[i].response, responses[j].response)
                agreed = div < self._threshold
                pa = PairwiseAgreement(
                    model_a=responses[i].model,
                    model_b=responses[j].model,
                    divergence=div,
                    agreed=agreed,
                )
                pairs.append(pa)
                if not agreed:
                    disagreements.append(pa)

        if pairs:
            avg_agreement = 1.0 - sum(p.divergence for p in pairs) / len(pairs)
            lowest = 1.0 - max(p.divergence for p in pairs)
        else:
            avg_agreement = 1.0
            lowest = 1.0

        return ConsensusResult(
            responses=responses,
            pairs=pairs,
            agreement_score=max(0.0, min(1.0, avg_agreement)),
            lowest_pair_agreement=max(0.0, min(1.0, lowest)),
            disagreement_pairs=disagreements,
            num_models=len(responses),
        )

    def _gather_responses(self, prompt: str) -> list[ModelResponse]:
        if self._generate is None:
            raise ValueError("generate_fn is required for score()")
        return [
            ModelResponse(model=m, response=self._generate(prompt, m))
            for m in self._models
        ]

    @staticmethod
    def _jaccard_divergence(a: str, b: str) -> float:
        wa = set(a.lower().split())
        wb = set(b.lower().split())
        if not wa or not wb:
            return 1.0
        return 1.0 - len(wa & wb) / len(wa | wb)
