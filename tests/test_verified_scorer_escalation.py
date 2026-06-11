# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# Director-Class AI -- default-path verified scorer escalation tests

from __future__ import annotations

import pytest

from director_ai.core.scorer import CoherenceScorer
from director_ai.core.types import EvidenceChunk, ScoringEvidence


def _evidence(source: str, response: str) -> ScoringEvidence:
    return ScoringEvidence(
        chunks=[EvidenceChunk(text=source, distance=0.0, source="test")],
        nli_premise=source,
        nli_hypothesis=response,
        nli_score=0.2,
    )


def test_low_confidence_review_runs_atomic_verified_scorer(monkeypatch) -> None:
    source = "Paris is the capital of France. Berlin is the capital of Germany."
    response = "Paris is the capital of France and Berlin is the capital of Germany."
    scorer = CoherenceScorer(threshold=0.6, use_nli=False)
    scorer._verified_scorer_enabled = True

    monkeypatch.setattr(
        scorer,
        "_heuristic_coherence",
        lambda *_args, **_kwargs: (0.1, 0.3, 0.61, _evidence(source, response)),
    )

    approved, score = scorer.review("What does the source say?", response)

    assert approved is True
    assert score.verified_approved is True
    assert score.verified_coverage == pytest.approx(1.0)
    assert score.verified_claim_count == 2
    assert score.verified_result is not None
    assert score.verified_result["claims"][0]["is_atomic"] is True


def test_rag_task_verification_can_fail_closed(monkeypatch) -> None:
    source = "Notifications support Slack and Microsoft Teams."
    response = "Notifications support WhatsApp Business approvals."
    scorer = CoherenceScorer(threshold=0.6, use_nli=False)
    scorer._verified_scorer_enabled = True

    monkeypatch.setattr(
        scorer,
        "_heuristic_coherence",
        lambda *_args, **_kwargs: (0.1, 0.1, 0.93, _evidence(source, response)),
    )

    approved, score = scorer.review(
        "Based on the following source document, answer the support question.",
        response,
    )

    assert approved is False
    assert score.approved is False
    assert score.verified_approved is False
    assert score.verified_coverage == pytest.approx(0.0)
    assert score.verified_result is not None
    assert score.verified_result["claims"][0]["verdict"] == "unverifiable"


def test_high_confidence_non_grounded_review_skips_verified_scorer(monkeypatch) -> None:
    source = "Paris is the capital of France."
    response = "Paris is the capital of France."
    scorer = CoherenceScorer(threshold=0.6, use_nli=False)
    scorer._verified_scorer_enabled = True

    monkeypatch.setattr(
        scorer,
        "_heuristic_coherence",
        lambda *_args, **_kwargs: (0.0, 0.0, 0.99, _evidence(source, response)),
    )

    approved, score = scorer.review("Write a short neutral paragraph.", response)

    assert approved is True
    assert score.verified_result is None
    assert score.verified_approved is None
