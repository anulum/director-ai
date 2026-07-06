# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - DSPy integration real-surface tests
"""Production-surface coverage for the DSPy and Instructor adapter."""

from __future__ import annotations

import pytest

from director_ai.core.exceptions import HallucinationError
from director_ai.integrations.dspy import coherence_check, director_assert
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def test_dspy_unit_guard_declares_real_surface_companion() -> None:
    """The DSPy unit guard should declare this companion."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_integrations_dspy.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_integrations_dspy_real_surface.py" in category


@pytest.mark.parametrize("bad_key", ["", "   "])
def test_coherence_check_rejects_blank_fact_key_before_scoring(bad_key: str) -> None:
    """Standalone validation should fail closed before scorer construction."""
    with pytest.raises(ValueError, match="fact key must not be blank"):
        coherence_check(
            "Refunds are available within 30 days.",
            prompt="What is the refund window?",
            facts={bad_key: "Refunds are available within 30 days."},
            threshold=0.1,
            use_nli=False,
        )


@pytest.mark.parametrize("bad_key", ["", "   "])
def test_director_assert_rejects_blank_fact_key_before_scoring(bad_key: str) -> None:
    """DSPy assertion wiring should reject invalid fact dictionaries."""
    with pytest.raises(ValueError, match="fact key must not be blank"):
        director_assert(
            "Refunds are available within 30 days.",
            prompt="What is the refund window?",
            facts={bad_key: "Refunds are available within 30 days."},
            threshold=0.1,
            use_nli=False,
        )


def test_standalone_pipeline_approves_grounded_answer() -> None:
    """The public adapter should approve grounded output through real scoring."""
    result = coherence_check(
        "Refunds are available within 30 days.",
        prompt="What is the refund window?",
        facts={"refund window": "Refunds are available within 30 days."},
        threshold=0.1,
        use_nli=False,
    )

    assert result["approved"] is True
    assert result["score"] >= 0.1
    assert "h_logical" in result["evidence"]
    assert "h_factual" in result["evidence"]


def test_director_assert_raises_hallucination_error_on_public_rejection() -> None:
    """The assertion surface should preserve rejection details for callers."""
    with pytest.raises(HallucinationError) as exc_info:
        director_assert(
            "The answer is unsupported.",
            prompt="What is the refund window?",
            facts={"refund window": "Refunds are available within 30 days."},
            threshold=0.99,
            use_nli=False,
        )

    err = exc_info.value
    assert err.query == "What is the refund window?"
    assert err.response == "The answer is unsupported."
    assert err.score.score < 0.99
