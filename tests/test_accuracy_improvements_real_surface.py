# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - accuracy-improvement real-surface tests
"""Real public-surface coverage for accuracy-improvement routing."""

from __future__ import annotations

from pathlib import Path

import pytest

from director_ai.core.config import DirectorConfig
from director_ai.core.scoring.scorer import CoherenceScorer
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def _adaptive_lite_scorer(
    *,
    threshold_dialogue: float = 0.68,
    threshold_qa: float = 0.69,
) -> CoherenceScorer:
    """Build the public scorer surface with dependency-free adaptive routing."""
    config = DirectorConfig(
        mode="general",
        use_nli=False,
        scorer_backend="lite",
        cache_size=0,
        adaptive_threshold_enabled=True,
        coherence_threshold=1.0,
        hard_limit=0.1,
        soft_limit=1.0,
        w_logic=0.0,
        w_fact=1.0,
        threshold_dialogue=threshold_dialogue,
        threshold_qa=threshold_qa,
        # Adaptive per-task thresholds gate the composite-coherence scale;
        # the WCS-2a raw-support dialogue route gates its own matched-FPR
        # operating point instead, so this surface pins the squeeze mode.
        nli_dialogue_scoring="baseline_squeeze",
    )
    return config.build_scorer()


def test_accuracy_improvements_unit_guard_declares_real_surface_companion() -> None:
    """The helper-heavy accuracy guard should name this public companion."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_accuracy_improvements.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_accuracy_improvements_real_surface.py" in category


def test_adaptive_thresholds_change_public_review_outcome_by_task_type() -> None:
    """Task-specific thresholds should affect real ``review`` decisions."""
    scorer = _adaptive_lite_scorer(threshold_dialogue=0.1, threshold_qa=1.0)

    dialogue_approved, dialogue_score = scorer.review(
        "User: What color is the sky?\nAssistant: Blue.\nUser: Are you sure?",
        "Yes, the sky is typically blue on a clear day.",
        tenant_id="tenant-a",
    )
    qa_approved, qa_score = scorer.review(
        "What is the deployment status?",
        "The deployment status is green.",
        tenant_id="tenant-a",
    )

    assert dialogue_approved is True
    assert dialogue_score.approved is True
    assert dialogue_score.detected_task_type == "dialogue"
    assert dialogue_score.score == pytest.approx(1.0)

    assert qa_approved is False
    assert qa_score.approved is False
    assert qa_score.detected_task_type == "qa"
    assert qa_score.score == pytest.approx(0.5)


def test_yaml_configured_fail_closed_meta_classifier_blocks_startup(
    tmp_path: Path,
) -> None:
    """A real YAML config should fail closed when its classifier is missing."""
    config_path = tmp_path / "director.yaml"
    missing_classifier = tmp_path / "missing-meta-classifier.pkl"
    config_path.write_text(
        "\n".join(
            [
                "mode: general",
                "use_nli: false",
                "scorer_backend: lite",
                "cache_size: 0",
                "adaptive_threshold_enabled: true",
                "adaptive_threshold_fail_closed: true",
                f"meta_classifier_path: {missing_classifier}",
            ]
        ),
        encoding="utf-8",
    )

    config = DirectorConfig.from_yaml(str(config_path))

    with pytest.raises(RuntimeError, match="Adaptive threshold classifier unavailable"):
        config.build_scorer()
