# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Lite Scorer Tests
"""Multi-angle tests for LiteScorer heuristic NLI.

Covers: score range invariants, empty inputs, long text, batch scoring,
entity overlap boost, identical text, CoherenceScorer pipeline integration,
parametrised inputs, determinism, and performance documentation.
"""

import pytest

from director_ai.core.lite_scorer import LiteScorer


class TestLiteScorer:
    def setup_method(self):
        self.scorer = LiteScorer()

    def test_score_returns_float_in_range(self):
        s = self.scorer.score("The sky is blue.", "The sky is blue.")
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0

    def test_aligned_pair_low_divergence(self):
        s = self.scorer.score(
            "The capital of France is Paris.",
            "Paris is the capital of France.",
        )
        assert s < 0.5

    def test_contradicted_pair_high_divergence(self):
        s = self.scorer.score(
            "Water boils at 100 degrees.",
            "Quantum entanglement in black holes produces Hawking radiation.",
        )
        assert s > 0.4

    def test_negation_penalty(self):
        base = self.scorer.score("The test passed.", "The test passed.")
        negated = self.scorer.score("The test passed.", "The test did not pass.")
        assert negated > base

    def test_empty_premise_returns_neutral(self):
        assert self.scorer.score("", "anything") == 0.5

    def test_empty_hypothesis_returns_neutral(self):
        assert self.scorer.score("anything", "") == 0.5

    def test_both_empty_returns_neutral(self):
        assert self.scorer.score("", "") == 0.5

    def test_batch_consistency(self):
        pairs = [
            ("The sky is blue.", "The sky is blue."),
            ("Dogs are animals.", "Cats are robots."),
        ]
        batch_scores = self.scorer.score_batch(pairs)
        individual = [self.scorer.score(p, h) for p, h in pairs]
        assert batch_scores == individual

    def test_batch_empty(self):
        assert self.scorer.score_batch([]) == []

    def test_entity_overlap_boosts_similarity(self):
        s_with = self.scorer.score(
            "Albert Einstein developed relativity.",
            "Albert Einstein was a physicist.",
        )
        s_without = self.scorer.score(
            "A scientist developed relativity.",
            "A different person was a physicist.",
        )
        assert s_with < s_without

    def test_identical_text_low_divergence(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert self.scorer.score(text, text) < 0.3


class TestLiteScorerIntegration:
    def test_coherence_scorer_lite_backend(self):
        from director_ai.core.scorer import CoherenceScorer

        scorer = CoherenceScorer(
            threshold=0.3,
            use_nli=False,
            scorer_backend="lite",
        )
        approved, score = scorer.review("The sky is blue.", "The sky is blue.")
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0


class TestLiteScorerParametrised:
    """Parametrised LiteScorer tests."""

    @pytest.mark.parametrize(
        "premise,hypothesis",
        [
            ("The sky is blue", "The sky is blue"),
            ("", "response"),
            ("prompt", ""),
            ("日本語テスト", "日本語レスポンス"),
            ("a" * 10000, "b" * 10000),
        ],
    )
    def test_various_inputs(self, premise, hypothesis):
        scorer = LiteScorer()
        result = scorer.score(premise, hypothesis)
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize("batch_size", [0, 1, 5, 10])
    def test_batch_sizes(self, batch_size):
        scorer = LiteScorer()
        pairs = [("p", "h")] * batch_size
        results = scorer.score_batch(pairs)
        assert len(results) == batch_size


class TestLiteScorerPerformanceDoc:
    """Document LiteScorer pipeline performance."""

    def test_score_fast(self):
        import time

        scorer = LiteScorer()
        t0 = time.perf_counter()
        for _ in range(1000):
            scorer.score("What is AI?", "AI is intelligence.")
        per_call_us = (time.perf_counter() - t0) / 1000 * 1_000_000
        assert per_call_us < 500, f"LiteScorer took {per_call_us:.0f}µs"

    def test_deterministic(self):
        scorer = LiteScorer()
        s1 = scorer.score("X", "Y")
        s2 = scorer.score("X", "Y")
        assert s1 == s2
