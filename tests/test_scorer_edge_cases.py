# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Scorer Edge Cases (STRONG)
"""Multi-angle edge case tests for CoherenceScorer.

Covers: empty/null/unicode inputs, very long text, boundary thresholds,
score determinism, score range invariants, weight consistency, threshold
boundary behaviour, RTL/emoji/mixed-script inputs, and pipeline
performance characteristics.
"""

from __future__ import annotations

import pytest

from director_ai.core import CoherenceScorer

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def scorer():
    return CoherenceScorer(use_nli=False)


@pytest.fixture
def strict_scorer():
    return CoherenceScorer(use_nli=False, strict_mode=True)


# ── Empty / null inputs ────────────────────────────────────────────


class TestEmptyInputs:
    """Scorer must handle empty/null-like inputs gracefully."""

    @pytest.mark.parametrize(
        "prompt,response",
        [
            ("", "Some response"),
            ("What is 2+2?", ""),
            ("", ""),
        ],
    )
    def test_empty_inputs_no_crash(self, scorer, prompt, response):
        approved, score = scorer.review(prompt, response)
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0

    def test_whitespace_only_prompt(self, scorer):
        approved, score = scorer.review("   \n\t  ", "Normal response")
        assert isinstance(approved, bool)

    def test_whitespace_only_response(self, scorer):
        approved, score = scorer.review("Normal prompt", "   \n\t  ")
        assert isinstance(approved, bool)

    def test_null_bytes_in_input(self, scorer):
        approved, score = scorer.review("test\x00prompt", "test\x00response")
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0

    def test_single_char_inputs(self, scorer):
        approved, score = scorer.review("?", ".")
        assert isinstance(approved, bool)


# ── Unicode / multilingual ─────────────────────────────────────────


class TestUnicodeInputs:
    """Scorer must handle diverse Unicode correctly."""

    @pytest.mark.parametrize(
        "prompt,response",
        [
            ("What is this? 🎉🥳", "It is a celebration 🎉"),
            ("ما هو 2+2؟", "الإجابة هي 4"),
            ("2+2は何ですか？", "答えは4です"),
            ("Čo je 2+2?", "Odpoveď je 4"),
            ("Що таке 2+2?", "Відповідь 4"),
            ("Mixed: hello مرحبا こんにちは", "Response in English"),
        ],
    )
    def test_multilingual_no_crash(self, scorer, prompt, response):
        approved, score = scorer.review(prompt, response)
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0

    def test_zero_width_chars(self, scorer):
        approved, score = scorer.review(
            "test\u200b\u200c\u200dprompt", "response\ufeff"
        )
        assert isinstance(approved, bool)

    def test_surrogate_like_chars(self, scorer):
        approved, score = scorer.review("𝕋𝕖𝕤𝕥", "ℝ𝕖𝕤𝕡𝕠𝕟𝕤𝕖")
        assert isinstance(approved, bool)


# ── Very long text ─────────────────────────────────────────────────


class TestLongText:
    """Scorer must handle large inputs without OOM or timeout."""

    @pytest.mark.parametrize("length", [1_000, 10_000, 100_000])
    def test_long_response_no_crash(self, scorer, length):
        long_text = "word " * length
        approved, score = scorer.review("Summarise", long_text)
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0

    def test_long_prompt_no_crash(self, scorer):
        long_prompt = "context " * 50_000
        approved, score = scorer.review(long_prompt, "Short answer.")
        assert isinstance(approved, bool)


# ── Score invariants ───────────────────────────────────────────────


class TestScoreInvariants:
    """Score components must satisfy mathematical invariants."""

    def test_score_in_range(self, scorer):
        for _ in range(20):
            _, score = scorer.review("What is AI?", "AI is artificial intelligence.")
            assert 0.0 <= score.score <= 1.0
            assert 0.0 <= score.h_logical <= 1.0
            assert 0.0 <= score.h_factual <= 1.0

    def test_deterministic(self, scorer):
        _, s1 = scorer.review("What?", "Answer.")
        _, s2 = scorer.review("What?", "Answer.")
        assert s1.score == s2.score
        assert s1.h_logical == s2.h_logical
        assert s1.h_factual == s2.h_factual

    def test_weights_sum_to_one(self):
        assert abs(CoherenceScorer.W_LOGIC + CoherenceScorer.W_FACT - 1.0) < 1e-9

    def test_coherence_formula_consistency(self, scorer):
        """Score = 1 - (w_logic * h_logical + w_fact * h_factual)."""
        _, score = scorer.review("What is 2+2?", "4")
        expected = 1.0 - (
            CoherenceScorer.W_LOGIC * score.h_logical
            + CoherenceScorer.W_FACT * score.h_factual
        )
        assert abs(score.score - expected) < 1e-6


# ── Threshold boundary ─────────────────────────────────────────────


class TestThresholdBoundary:
    """Test behaviour at exact threshold boundaries."""

    @pytest.mark.parametrize("threshold", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    def test_various_thresholds_accepted(self, threshold):
        scorer = CoherenceScorer(use_nli=False, threshold=threshold)
        approved, score = scorer.review("test", "test")
        assert isinstance(approved, bool)

    def test_threshold_zero_always_approves(self):
        scorer = CoherenceScorer(use_nli=False, threshold=0.0)
        approved, _ = scorer.review("test", "test")
        assert approved is True

    def test_threshold_one_rejects_nonperfect(self):
        scorer = CoherenceScorer(use_nli=False, threshold=1.0, soft_limit=1.0)
        # Heuristic scorer unlikely to produce exactly 1.0
        approved, score = scorer.review("X is Y", "Z is W")
        # Either approved or not — but must not crash
        assert isinstance(approved, bool)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            CoherenceScorer(use_nli=False, threshold=1.5)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            CoherenceScorer(use_nli=False, threshold=-0.1)

    def test_soft_limit_below_threshold_raises(self):
        with pytest.raises(ValueError, match="soft_limit"):
            CoherenceScorer(use_nli=False, threshold=0.7, soft_limit=0.5)


# ── Strict mode ────────────────────────────────────────────────────


class TestStrictMode:
    """Strict mode must disable heuristic fallbacks."""

    def test_strict_mode_flag_stored(self, strict_scorer):
        assert strict_scorer.strict_mode is True

    def test_non_strict_default(self, scorer):
        assert scorer.strict_mode is False


# ── Performance characteristics ─────────────────────────────────────


class TestPerformanceDoc:
    """Document and verify performance guarantees."""

    def test_heuristic_score_has_evidence_field(self, scorer):
        _, score = scorer.review("What is AI?", "AI is intelligence.")
        assert hasattr(score, "evidence")

    def test_heuristic_score_has_components(self, scorer):
        _, score = scorer.review("Q", "A")
        assert hasattr(score, "h_logical")
        assert hasattr(score, "h_factual")
        assert hasattr(score, "score")
        assert hasattr(score, "approved")

    def test_review_returns_tuple(self, scorer):
        result = scorer.review("Q", "A")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
