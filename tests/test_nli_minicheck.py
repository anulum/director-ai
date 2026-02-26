from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from director_ai.core.nli import NLIScorer


class TestMiniCheckBackend:
    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="backend must be one of"):
            NLIScorer(backend="nonexistent")

    def test_deberta_backend_default(self):
        scorer = NLIScorer(use_model=False)
        assert scorer.backend == "deberta"

    def test_minicheck_fallback_to_heuristic(self):
        scorer = NLIScorer(backend="minicheck")
        result = scorer.score("The sky is blue.", "The sky is blue.")
        assert 0.0 <= result <= 1.0

    @patch("director_ai.core.nli.NLIScorer._ensure_minicheck")
    def test_minicheck_dispatches_correctly(self, mock_ensure):
        mock_ensure.return_value = False
        scorer = NLIScorer(backend="minicheck")
        result = scorer.score("A", "B")
        mock_ensure.assert_called_once()
        assert 0.0 <= result <= 1.0

    def test_minicheck_with_mock_package(self):
        mock_mc = MagicMock()
        mock_mc.score.return_value = [0.8]
        scorer = NLIScorer(backend="minicheck")
        scorer._minicheck = mock_mc
        scorer._minicheck_loaded = True
        result = scorer.score("The sky is blue.", "The sky is blue.")
        assert result == pytest.approx(0.2)
        mock_mc.score.assert_called_once()

    def test_minicheck_high_contradiction(self):
        mock_mc = MagicMock()
        mock_mc.score.return_value = [0.1]
        scorer = NLIScorer(backend="minicheck")
        scorer._minicheck = mock_mc
        scorer._minicheck_loaded = True
        result = scorer.score("Earth orbits Sun.", "Sun orbits Earth.")
        assert result == pytest.approx(0.9)

    def test_score_batch_uses_backend(self):
        mock_mc = MagicMock()
        mock_mc.score.return_value = [0.5]
        scorer = NLIScorer(backend="minicheck")
        scorer._minicheck = mock_mc
        scorer._minicheck_loaded = True
        results = scorer.score_batch([("A", "B"), ("C", "D")])
        assert len(results) == 2
        assert all(0.0 <= r <= 1.0 for r in results)
