# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — cross-model consensus and divergence explanation

"""Measure agreement across several models' answers to one prompt.

Every other guard in this package scores a *single* response. This one scores a
*panel*: given the same prompt answered by GPT, Claude, Gemini, a local Llama, or
any other set of models, it quantifies how much the answers agree, names the
specific claims where they diverge, and recommends whether to accept the majority
answer, send it for review, or escalate to a stronger model or a human.

The consensus rests on two measurable signals over every unordered pair of
answers:

* **semantic agreement** — when an NLI contradiction scorer is supplied, the
  agreement of a pair is ``1 - max(P(contradiction) a→b, P(contradiction) b→a)``;
  the directional maximum is taken because "A contradicts B" and "B contradicts
  A" are scored separately and either is enough to break agreement;
* **lexical agreement** — the fallback when no scorer is supplied (and a fast
  cheap pre-filter when one is): the Jaccard word overlap of the two answers via
  the Rust ``rust_word_overlap`` kernel, with a bit-exact pure-Python fallback.

The panel consensus is the mean pairwise agreement. Divergences are reported at
the *claim* level: each answer is split into claims and every cross-model claim
pair whose contradiction probability clears ``divergence_threshold`` is returned
as evidence (which two models, which two claims, how strongly they conflict), so
"the models disagree" always comes with the sentence that caused it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..text_overlap import word_overlap

__all__ = [
    "ContradictionEngine",
    "ModelResponse",
    "Divergence",
    "ConsensusResult",
    "CrossModelConsensus",
]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-zA-Z0-9']+")


@runtime_checkable
class ContradictionEngine(Protocol):
    """An NLI contradiction scorer.

    Duck-typed against
    :class:`director_ai.core.scoring.contradiction.ContradictionScorer`: a
    ``contradiction(premise, hypothesis)`` method returning ``P(contradiction)``
    in ``[0, 1]`` and a ``threshold`` for the halt/flag decision.
    """

    def contradiction(self, premise: str, hypothesis: str) -> float:
        """Return the directional contradiction probability for two texts."""
        ...

    @property
    def threshold(self) -> float:
        """Return the contradiction threshold used for flagging."""
        ...


def _lexical_overlap(text_a: str, text_b: str) -> float:
    """Lexical Jaccard overlap in ``[0, 1]``.

    Delegates to the shared measured-fast-path helper (pure Python below a large
    -input threshold, Rust above it). See :mod:`director_ai.core.text_overlap`.
    """
    return word_overlap(text_a, text_b, logger_name=__name__)


def _split_claims(text: str, *, min_words: int = 3, cap: int = 24) -> list[str]:
    """Split *text* into claim-sized sentences, dropping fragments and capping count.

    Sentences shorter than ``min_words`` content tokens are noise (greetings,
    list bullets, "Yes.") and are dropped; at most ``cap`` claims are kept so a
    pathologically long answer cannot blow up the O(n²) cross-claim comparison.
    """
    claims: list[str] = []
    for raw in _SENTENCE_SPLIT.split(text.strip()):
        sentence = raw.strip()
        if len(_WORD_RE.findall(sentence)) >= min_words:
            claims.append(sentence)
        if len(claims) >= cap:
            break
    return claims


@dataclass(frozen=True)
class ModelResponse:
    """One model's answer to the shared prompt."""

    model_id: str
    text: str


@dataclass(frozen=True)
class Divergence:
    """A specific claim-level disagreement between two models, with evidence."""

    model_a: str
    model_b: str
    claim_a: str
    claim_b: str
    contradiction: float


@dataclass(frozen=True)
class ConsensusResult:
    """The consensus verdict over a panel of model answers."""

    consensus: float
    recommendation: str
    n_models: int
    agreement_matrix: tuple[tuple[float, ...], ...]
    divergences: tuple[Divergence, ...]
    rationale: tuple[str, ...]


@dataclass
class CrossModelConsensus:
    """Quantify agreement across a panel of model answers to one prompt.

    Parameters
    ----------
    nli:
        Optional :class:`ContradictionEngine`. When supplied, pairwise agreement
        is semantic (``1 - max`` directional contradiction) and divergences are
        reported at claim level with contradiction evidence. When ``None``,
        agreement is the lexical Jaccard overlap and divergences fall back to the
        lowest-agreement answer pair.
    accept_threshold / escalate_threshold:
        Consensus band edges. At or above ``accept_threshold`` recommend
        ``accept``; at or above ``escalate_threshold`` recommend ``review``;
        below it recommend ``escalate`` (route to a stronger model or a human).
    divergence_threshold:
        Minimum ``P(contradiction)`` for a claim pair to be reported as a
        divergence. Defaults to the injected scorer's own ``threshold`` when not
        set explicitly.
    max_divergences:
        Cap on reported divergences, strongest first, to keep the result bounded.
    """

    nli: ContradictionEngine | None = None
    accept_threshold: float = 0.7
    escalate_threshold: float = 0.45
    divergence_threshold: float | None = None
    max_divergences: int = 12

    def __post_init__(self) -> None:
        """Validate consensus thresholds and result bounds."""
        if not 0.0 <= self.escalate_threshold <= self.accept_threshold <= 1.0:
            raise ValueError("require 0 <= escalate_threshold <= accept_threshold <= 1")
        if self.divergence_threshold is not None and not (
            0.0 <= self.divergence_threshold <= 1.0
        ):
            raise ValueError("divergence_threshold must be in [0, 1]")
        if self.max_divergences < 0:
            raise ValueError("max_divergences must be non-negative")

    @property
    def _flag_threshold(self) -> float:
        if self.divergence_threshold is not None:
            return self.divergence_threshold
        if self.nli is not None:
            return float(self.nli.threshold)
        return 0.5

    def agreement(self, text_a: str, text_b: str) -> float:
        """Pairwise agreement of two answers in ``[0, 1]`` (1 = fully agree)."""
        if self.nli is not None:
            contra = max(
                self.nli.contradiction(text_a, text_b),
                self.nli.contradiction(text_b, text_a),
            )
            return max(0.0, min(1.0, 1.0 - contra))
        return _lexical_overlap(text_a, text_b)

    def _divergences(
        self, responses: Sequence[ModelResponse]
    ) -> tuple[Divergence, ...]:
        """Claim-level contradictions across models, strongest first."""
        if self.nli is None:
            return ()
        flag = self._flag_threshold
        found: list[Divergence] = []
        claims = [(_split_claims(r.text), r.model_id) for r in responses]
        for i in range(len(claims)):
            claims_i, model_i = claims[i]
            for j in range(i + 1, len(claims)):
                claims_j, model_j = claims[j]
                for claim_a in claims_i:
                    for claim_b in claims_j:
                        contra = max(
                            self.nli.contradiction(claim_a, claim_b),
                            self.nli.contradiction(claim_b, claim_a),
                        )
                        if contra >= flag:
                            found.append(
                                Divergence(
                                    model_a=model_i,
                                    model_b=model_j,
                                    claim_a=claim_a,
                                    claim_b=claim_b,
                                    contradiction=round(contra, 4),
                                )
                            )
        found.sort(key=lambda d: d.contradiction, reverse=True)
        return tuple(found[: self.max_divergences])

    def consensus(self, responses: Sequence[ModelResponse]) -> ConsensusResult:
        """Return the :class:`ConsensusResult` for a panel of model answers."""
        n = len(responses)
        if n < 2:
            raise ValueError("consensus needs at least two model responses")

        matrix = [[1.0] * n for _ in range(n)]
        pair_scores: list[float] = []
        weakest = (1.0, 0, 1)
        for i in range(n):
            for j in range(i + 1, n):
                score = round(self.agreement(responses[i].text, responses[j].text), 4)
                matrix[i][j] = matrix[j][i] = score
                pair_scores.append(score)
                if score < weakest[0]:
                    weakest = (score, i, j)

        consensus = round(sum(pair_scores) / len(pair_scores), 4)

        divergences = self._divergences(responses)
        if not divergences and self.nli is None:
            # No semantic scorer: surface the single least-agreeing pair as the
            # divergence evidence so the caller still sees who disagreed most.
            _, i, j = weakest
            divergences = (
                Divergence(
                    model_a=responses[i].model_id,
                    model_b=responses[j].model_id,
                    claim_a=responses[i].text,
                    claim_b=responses[j].text,
                    contradiction=round(1.0 - weakest[0], 4),
                ),
            )

        if consensus >= self.accept_threshold:
            recommendation = "accept"
        elif consensus >= self.escalate_threshold:
            recommendation = "review"
        else:
            recommendation = "escalate"

        rationale: list[str] = []
        if consensus >= self.accept_threshold:
            rationale.append("models broadly agree")
        elif consensus < self.escalate_threshold:
            rationale.append("models broadly disagree")
        else:
            rationale.append("partial agreement")
        if divergences and self.nli is not None:
            rationale.append(f"{len(divergences)} contradicting claim pair(s)")
        if self.nli is None:
            rationale.append("lexical agreement only (no NLI scorer supplied)")

        return ConsensusResult(
            consensus=consensus,
            recommendation=recommendation,
            n_models=n,
            agreement_matrix=tuple(tuple(row) for row in matrix),
            divergences=divergences,
            rationale=tuple(rationale),
        )
