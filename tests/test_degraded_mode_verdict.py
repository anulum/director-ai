# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Degraded-Mode Verdict Flag (KIMI2-F)
"""Contracts for the first-class ``degraded_mode`` verdict field.

KIMI's DX finding: a quickstart deployment without the ``[nli]`` extra scores
with a weak word-overlap heuristic that false-blocks true claims, and the only
signal was a log line (``reason="nli_unavailable_using_heuristic"``). The
verdict itself now carries ``degraded_mode`` so a caller can detect — in
code, not in logs — that the score came from heuristic/lite scoring with no
model-backed contradiction path behind it.
"""

from __future__ import annotations

from typing import Any

from director_ai.core import GroundTruthStore
from director_ai.core.scoring.scorer import CoherenceScorer


class _ModelNLI:
    """Minimal model-backed NLI double (non-lite backend)."""

    model_available = True
    backend = "deberta"

    def _ensure_model(self) -> bool:
        return True

    def score_chunked(self, premise: str, hypothesis: str, **_kw: Any):
        del premise, hypothesis
        return 0.1, None

    def _score_chunked_with_counts(self, premise: str, hypothesis: str, **_kw: Any):
        del premise, hypothesis
        return 0.1, None, 1, 1


def test_heuristic_review_reports_degraded_mode() -> None:
    scorer = CoherenceScorer(use_nli=False, threshold=0.5)
    scorer._rust_scorer = None

    _approved, score = scorer.review("What color is the sky?", "The sky is blue.")

    assert score.degraded_mode is True


def test_grounded_heuristic_review_reports_degraded_mode() -> None:
    store = GroundTruthStore()
    store.add("sky color", "The sky is blue.")
    scorer = CoherenceScorer(use_nli=False, threshold=0.5, ground_truth_store=store)
    scorer._rust_scorer = None

    _approved, score = scorer.review("What color is the sky?", "The sky is blue.")

    assert score.degraded_mode is True


def test_model_backed_review_is_not_degraded() -> None:
    scorer = CoherenceScorer(use_nli=False, threshold=0.5)
    scorer._rust_scorer = None
    scorer._nli = _ModelNLI()

    _approved, score = scorer.review("prompt", "response")

    assert score.degraded_mode is False


def test_cached_review_carries_the_same_degraded_flag() -> None:
    scorer = CoherenceScorer(use_nli=False, threshold=0.5)
    scorer._rust_scorer = None

    _a1, first = scorer.review("cache me", "The sky is blue.")
    _a2, second = scorer.review("cache me", "The sky is blue.")

    assert first.degraded_mode is True
    assert second.degraded_mode is True


def test_default_coherence_score_is_not_degraded() -> None:
    from director_ai.core.types import CoherenceScore

    score = CoherenceScore(score=1.0, approved=True, h_logical=0.0, h_factual=0.0)

    assert score.degraded_mode is False
