# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Grounded Logical-Premise Regression (KIMI2-K)
"""Regression tests for the KIMI2-K false-halt fix.

The GPU reproduction of the KIMI red-team (2026-07-16) established that the
detector catches false claims robustly but *false-halts true answers to
questions*. Root cause: ``h_logical`` ran NLI with ``premise=question``, and a
true declarative answer does not entail the question that prompted it, so
``h_logical`` was spuriously high for EVERY true answer. When a grounding store
is present the retrieved context is the correct premise for the logical signal
too — these tests pin that behaviour and prove the fix does not reopen the
evasion path (a false claim contradicted by the context is still caught).
"""

from __future__ import annotations

from typing import Any

from director_ai.core.scoring.scorer import CoherenceScorer

_QUESTION = "What is the capital of France?"
_CONTEXT = "The capital of France is Paris."
_TRUE_CLAIM = "France's capital city is Paris."
_FALSE_CLAIM = "France's capital city is Berlin."


class _PremiseSensingNLI:
    """NLI double that records every premise and models the KIMI2-K defect.

    A bare interrogative premise (ends in ``?``) yields a spuriously high
    divergence regardless of the claim's truth — the exact degeneracy that
    false-halted true inputs. A declarative context premise yields a low
    divergence when the claim agrees with it and a high one when it conflicts.
    """

    model_available = True

    def __init__(self) -> None:
        self.premises: list[str] = []
        self.last_token_count = 0
        self.last_estimated_cost = 0.0
        self._cost_per_token = 0.0

    def _ensure_model(self) -> bool:
        return True

    def reset_token_counter(self) -> None:
        self.last_token_count = 0

    def _divergence(self, premise: str, hypothesis: str) -> float:
        self.premises.append(premise)
        if premise.rstrip().endswith("?"):
            # Degenerate: a declarative answer never entails a question.
            return 0.85
        return 0.15 if "Paris" in hypothesis else 0.85

    def score_chunked(
        self, premise: str, hypothesis: str, **_kwargs: Any
    ) -> tuple[float, None]:
        return self._divergence(premise, hypothesis), None

    def _score_chunked_with_counts(
        self, premise: str, hypothesis: str, **_kwargs: Any
    ) -> tuple[float, None, int, int]:
        return self._divergence(premise, hypothesis), None, 1, 1


class _FactStore:
    """Minimal non-vector grounding store returning a fixed context."""

    def __init__(self, context: str) -> None:
        self._context = context

    def retrieve_context(self, prompt: str, top_k: int = 3, tenant_id: str = "") -> str:
        del prompt, top_k, tenant_id
        return self._context


def _grounded_scorer(context: str) -> CoherenceScorer:
    scorer = CoherenceScorer(use_nli=False, threshold=0.5)
    scorer._rust_scorer = None
    scorer._nli = _PremiseSensingNLI()
    scorer.ground_truth_store = _FactStore(context)
    return scorer


def test_grounded_true_answer_to_a_question_is_not_false_halted() -> None:
    scorer = _grounded_scorer(_CONTEXT)

    approved, score = scorer.review(_QUESTION, _TRUE_CLAIM)

    # Before the fix the question was the logical premise → h_logical ≈ 0.85
    # dragged the score below 0.5 and blocked a true answer.
    assert approved
    assert score.score >= 0.5


def test_grounded_logical_premise_is_the_context_not_the_question() -> None:
    scorer = _grounded_scorer(_CONTEXT)
    nli = scorer._nli
    assert isinstance(nli, _PremiseSensingNLI)

    scorer.review(_QUESTION, _TRUE_CLAIM)

    # The logical NLI scored against the retrieved context, never the bare
    # interrogative prompt.
    assert _CONTEXT in nli.premises
    assert _QUESTION not in nli.premises


def test_grounded_false_answer_to_a_question_is_still_caught() -> None:
    scorer = _grounded_scorer(_CONTEXT)

    approved, score = scorer.review(_QUESTION, _FALSE_CLAIM)

    # The context contradicts the claim → both signals high → still blocked.
    # This proves the false-halt fix does not reopen the evasion path.
    assert not approved
    assert score.score < 0.5


def test_ungrounded_review_keeps_the_prompt_premise() -> None:
    # No grounding store: the logical premise cannot improve on the prompt, so
    # the original prompt-premise path is preserved (degraded — see KIMI2-F).
    scorer = CoherenceScorer(use_nli=False, threshold=0.5)
    scorer._rust_scorer = None
    scorer._nli = _PremiseSensingNLI()
    scorer.ground_truth_store = None

    scorer.review(_QUESTION, _TRUE_CLAIM)

    nli = scorer._nli
    assert isinstance(nli, _PremiseSensingNLI)
    assert _QUESTION in nli.premises
