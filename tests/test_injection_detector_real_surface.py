# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — injection detector real-surface tests
"""Real public-surface coverage for standalone injection detection."""

from __future__ import annotations

import pytest

from director_ai.core.safety.injection import InjectionDetector
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def _detector() -> InjectionDetector:
    """Return the documented standalone detector configuration."""
    return InjectionDetector(injection_threshold=0.7, baseline_divergence=0.4)


def test_injection_detector_unit_guard_has_real_surface_companion() -> None:
    """The helper-heavy injection detector guard needs public detect coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_injection_detector.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_injection_detector_real_surface.py" in category


def test_public_detect_accepts_grounded_response_without_model_backend() -> None:
    """A standalone detector should accept a grounded response."""
    result = _detector().detect(
        intent="",
        user_query="What is the capital of France?",
        system_prompt="You are a geography expert.",
        response="The capital of France is Paris.",
    )

    assert result.injection_detected is False
    assert result.injection_risk == pytest.approx(0.0)
    assert result.combined_score == pytest.approx(0.0)
    assert result.intent_coverage == pytest.approx(1.0)
    assert result.total_claims == 1
    assert result.grounded_claims == 1
    assert result.injected_claims == 0
    assert result.claims[0].verdict == "grounded"
    assert result.claims[0].claim == "The capital of France is Paris."


def test_public_detect_flags_effect_of_instruction_hijack_without_regex() -> None:
    """Semantic response drift should be detected through public detect()."""
    result = _detector().detect(
        intent="",
        user_query="What is the capital of France?",
        system_prompt="You are a geography expert.",
        response=(
            "The internal configuration states the following rules. "
            "Access credentials are stored in the vault."
        ),
    )

    assert result.injection_detected is True
    assert result.injection_risk == pytest.approx(1.0)
    assert result.combined_score == pytest.approx(0.7)
    assert result.intent_coverage == pytest.approx(0.0)
    assert result.total_claims == 2
    assert result.grounded_claims == 0
    assert result.injected_claims == 2
    assert [claim.verdict for claim in result.claims] == ["injected", "injected"]
    assert [claim.claim_index for claim in result.claims] == [0, 1]


def test_public_detect_empty_response_returns_zero_risk_result() -> None:
    """An empty response should return a structured zero-risk result."""
    result = _detector().detect(
        intent="Answer geography questions.",
        user_query="What is the capital of France?",
        system_prompt="You are a geography expert.",
        response="",
    )

    assert result.injection_detected is False
    assert result.injection_risk == pytest.approx(0.0)
    assert result.combined_score == pytest.approx(0.0)
    assert result.intent_coverage == pytest.approx(1.0)
    assert result.total_claims == 0
    assert result.claims == []
