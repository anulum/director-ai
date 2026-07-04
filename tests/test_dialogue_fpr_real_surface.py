# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Dialogue FPR real-surface tests
"""Real public-surface coverage for dialogue false-positive routing."""

from __future__ import annotations

import pytest

from director_ai.core.config import DirectorConfig
from director_ai.core.scoring.scorer import CoherenceScorer
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def _build_public_lite_scorer() -> CoherenceScorer:
    """Return the public scorer configuration used for dialogue routing."""
    return DirectorConfig(
        mode="general",
        use_nli=False,
        scorer_backend="lite",
        coherence_threshold=0.2,
        hard_limit=0.2,
        soft_limit=0.2,
        adaptive_threshold_enabled=False,
        cache_size=0,
        w_logic=0.0,
        w_fact=1.0,
    ).build_scorer()


def test_dialogue_fpr_unit_guard_has_real_surface_companion() -> None:
    """The helper-heavy dialogue FPR guard needs public scorer coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_dialogue_fpr.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_dialogue_fpr_real_surface.py" in category


def test_public_review_routes_multi_turn_dialogue_without_false_reject() -> None:
    """A real scorer review should accept aligned multi-turn dialogue."""
    scorer = _build_public_lite_scorer()

    approved, score = scorer.review(
        "User: What color is the sky?\nAssistant: It is blue.\nUser: Are you sure?",
        "Yes, the sky is typically blue on a clear day.",
        tenant_id="tenant-a",
    )

    assert approved is True
    assert score.approved is True
    assert score.detected_task_type == "dialogue"
    assert score.h_logical == pytest.approx(0.0)
    assert score.h_factual == pytest.approx(0.0)
    assert score.score == pytest.approx(1.0)


def test_public_review_keeps_single_speaker_question_out_of_dialogue() -> None:
    """A single speaker marker should route as QA, not dialogue."""
    scorer = _build_public_lite_scorer()

    approved, score = scorer.review(
        "User: What is the capital of France?",
        "Paris is the capital of France.",
        tenant_id="tenant-a",
    )

    assert approved is True
    assert score.approved is True
    assert score.detected_task_type == "qa"
    assert score.detected_task_type != "dialogue"
    assert score.h_logical == pytest.approx(0.0)
    assert score.h_factual == pytest.approx(0.5)
    assert score.score == pytest.approx(0.5)
