# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - production guard runtime real-surface tests
"""Real-surface tests for remaining ProductionGuard runtime branches."""

from __future__ import annotations

import pytest

from director_ai.core.config import DirectorConfig
from director_ai.core.types import CoherenceScore
from director_ai.guard import GuardResult, ProductionGuard


class _FeedbackStore:
    """Small feedback sink matching the production feedback-store protocol."""

    def __init__(self) -> None:
        """Initialise the recorded feedback buffer."""
        self.rows: list[tuple[float, bool]] = []

    def add(self, score: float, correct_label: bool) -> None:
        """Record one score/correction pair."""
        self.rows.append((score, correct_label))


class _ThresholdCalibrator:
    """Small calibrator sink matching the production calibrator protocol."""

    def __init__(self) -> None:
        """Initialise the recorded calibrator update buffer."""
        self.updates: list[tuple[float, bool]] = []

    def update(self, score: float, correct_label: bool) -> None:
        """Record one calibrator update."""
        self.updates.append((score, correct_label))


def test_record_feedback_updates_calibrator_without_conformal_predictor() -> None:
    """Feedback recording should not require conformal intervals to be enabled."""
    guard = ProductionGuard(config=DirectorConfig(use_nli=False))
    feedback = _FeedbackStore()
    calibrator = _ThresholdCalibrator()
    guard._feedback = feedback
    guard._calibrator = calibrator
    guard._conformal = None
    result = GuardResult(
        approved=True,
        score=0.87,
        coherence=CoherenceScore(
            score=0.87,
            approved=True,
            h_logical=0.1,
            h_factual=0.1,
        ),
    )

    guard.record_feedback(result, correct_label=False)

    assert feedback.rows == [(0.87, False)]
    assert calibrator.updates == [(0.87, False)]


def test_new_threshold_governor_accepts_explicit_thresholds_without_router() -> None:
    """Runtime governor construction should allow explicit thresholds and no router."""
    guard = ProductionGuard(
        config=DirectorConfig(use_nli=False, coherence_threshold=0.67),
    )

    governor = guard.new_threshold_governor(
        candidate_thresholds=(0.25, 0.67, 0.9),
        max_step=0.03,
        auto_apply=True,
        with_uncertainty_router=False,
    )

    assert governor.current_threshold == pytest.approx(0.67)
    assert governor.max_step == pytest.approx(0.03)
    assert governor.auto_apply is True
    assert governor.uncertainty_router is None
    assert [arm.threshold for arm in governor.learner.report().arms] == [
        0.25,
        0.67,
        0.9,
    ]
