# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — NLI Scorer Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import pytest

from director_ai.core.nli import NLIScorer


@pytest.mark.consumer
class TestNLIScorer:
    def test_heuristic_fallback_consistent(self):
        scorer = NLIScorer(use_model=False)
        h = scorer.score("test", "This is consistent with reality.")
        assert h == pytest.approx(0.1)

    def test_heuristic_fallback_contradiction(self):
        scorer = NLIScorer(use_model=False)
        h = scorer.score("test", "The opposite is true.")
        assert h == pytest.approx(0.9)

    def test_heuristic_fallback_neutral(self):
        scorer = NLIScorer(use_model=False)
        h = scorer.score("test", "The answer depends on your perspective.")
        assert h == pytest.approx(0.5)

    def test_heuristic_fallback_overlap(self):
        scorer = NLIScorer(use_model=False)
        h = scorer.score("The sky is blue", "The sky is blue and clear")
        assert 0.0 <= h <= 1.0

    def test_model_available_is_false_without_model(self):
        scorer = NLIScorer(use_model=False)
        assert scorer.model_available is False

    def test_score_batch(self):
        scorer = NLIScorer(use_model=False)
        pairs = [
            ("test", "consistent with reality"),
            ("test", "opposite is true"),
        ]
        results = scorer.score_batch(pairs)
        assert len(results) == 2
        assert results[0] < results[1]

    def test_score_range(self):
        scorer = NLIScorer(use_model=False)
        for text in ["hello world", "anything", "random noise xyz"]:
            h = scorer.score("test prompt", text)
            assert 0.0 <= h <= 1.0
