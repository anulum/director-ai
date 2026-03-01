# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — DeBERTa NLI Integration Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Integration tests that actually load the DeBERTa-v3-base-mnli model.

These require a network connection on first run (model is cached by
HuggingFace). Mark with @slow so they're skipped in fast CI runs:

    pytest -m slow        # run only slow tests
    pytest -m "not slow"  # skip slow tests (default CI)
"""

import pytest

from director_ai.core.nli import NLIScorer

# Skip if model can't load (no torch, no network, etc.)
_scorer = NLIScorer(use_model=True)
_model_ok = _scorer.model_available


@pytest.mark.slow
@pytest.mark.consumer
@pytest.mark.skipif(not _model_ok, reason="DeBERTa model not available")
class TestDeBERTaIntegration:
    """Integration tests using the real DeBERTa NLI model."""

    @pytest.fixture(scope="class")
    def scorer(self):
        return NLIScorer(use_model=True)

    def test_model_loads(self, scorer):
        assert scorer.model_available

    def test_entailment_low_score(self, scorer):
        """Entailing pairs should score low (near 0)."""
        s = scorer.score(
            "The sky is blue during the day.",
            "The sky has a blue color when the sun is up.",
        )
        assert s < 0.4

    def test_contradiction_high_score(self, scorer):
        """Contradicting pairs should score high (near 1)."""
        s = scorer.score(
            "The Earth revolves around the Sun.",
            "The Sun revolves around the Earth.",
        )
        assert s > 0.5

    def test_neutral_mid_score(self, scorer):
        """Neutral pairs should score mid-range."""
        s = scorer.score(
            "The cat sat on the mat.",
            "It was raining outside.",
        )
        assert 0.2 < s < 0.8

    def test_score_range(self, scorer):
        """Score always in [0, 1]."""
        pairs = [
            ("Water is wet.", "Water is dry."),
            ("Dogs bark.", "Dogs are animals."),
            ("Two plus two is four.", "Math is abstract."),
        ]
        for p, h in pairs:
            s = scorer.score(p, h)
            assert 0.0 <= s <= 1.0, f"Score {s} out of range for ({p}, {h})"

    def test_batch_scoring(self, scorer):
        """Batched forward pass returns correct results."""
        pairs = [
            ("A is B.", "A is B."),
            ("A is B.", "A is not B."),
        ]
        scores = scorer.score_batch(pairs)
        assert len(scores) == 2
        assert scores[1] > scores[0]

    def test_batch_matches_sequential(self, scorer):
        """Batched results match sequential scoring."""
        pairs = [
            ("Water is wet.", "Water is a liquid."),
            ("The sky is blue.", "The sky is green."),
            ("Dogs bark.", "Cats meow."),
        ]
        batch = scorer.score_batch(pairs)
        sequential = [scorer.score(p, h) for p, h in pairs]
        for b, s in zip(batch, sequential, strict=True):
            assert abs(b - s) < 0.02

    def test_long_input_truncation(self, scorer):
        """Very long inputs are truncated without error."""
        long_text = "word " * 1000
        s = scorer.score(long_text, "Short hypothesis.")
        assert 0.0 <= s <= 1.0

    def test_empty_input(self, scorer):
        """Empty strings don't crash the model."""
        s = scorer.score("", "")
        assert 0.0 <= s <= 1.0
