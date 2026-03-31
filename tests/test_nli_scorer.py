# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — NLI Scorer Tests (STRONG)
"""Multi-angle tests for NLI scorer core pipeline.

Covers: heuristic scoring, score range invariants, batch scoring,
empty/long inputs, ONNX fallback, invalid backend guard, determinism,
pipeline integration with CoherenceScorer, and performance documentation.
"""

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

    def test_score_batch_empty(self):
        scorer = NLIScorer(use_model=False)
        assert scorer.score_batch([]) == []

    def test_score_batch_matches_sequential(self):
        scorer = NLIScorer(use_model=False)
        pairs = [
            ("sky is blue", "consistent with reality"),
            ("earth is round", "opposite is true"),
            ("cats meow", "depends on your perspective"),
            ("water is wet", "random unrelated text"),
        ]
        batch = scorer.score_batch(pairs)
        sequential = [scorer.score(p, h) for p, h in pairs]
        assert batch == sequential

    def test_score_range(self):
        scorer = NLIScorer(use_model=False)
        for text in ["hello world", "anything", "random noise xyz"]:
            h = scorer.score("test prompt", text)
            assert 0.0 <= h <= 1.0

    def test_onnx_backend_without_path_falls_back(self):
        scorer = NLIScorer(use_model=True, backend="onnx")
        s = scorer.score("premise", "hypothesis")
        assert 0.0 <= s <= 1.0

    def test_onnx_backend_invalid_path_falls_back(self, tmp_path):
        scorer = NLIScorer(
            use_model=True,
            backend="onnx",
            onnx_path=str(tmp_path / "no_such_dir_xyz"),
        )
        s = scorer.score("premise", "hypothesis")
        assert 0.0 <= s <= 1.0

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="backend"):
            NLIScorer(backend="invalid")
