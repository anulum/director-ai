# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — MiniCheck NLI Backend Tests (STRONG)
"""Multi-angle tests for MiniCheck NLI backend.

Covers: invalid backend guard, default backend, fallback to heuristic,
mock dispatch, mock package scoring, batch scoring, high contradiction,
parametrised backends, score invariants, pipeline integration, and
performance documentation.
"""

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
        mock_mc.score.return_value = [0.5, 0.7]
        scorer = NLIScorer(backend="minicheck")
        scorer._minicheck = mock_mc
        scorer._minicheck_loaded = True
        results = scorer.score_batch([("A", "B"), ("C", "D")])
        assert len(results) == 2
        assert all(0.0 <= r <= 1.0 for r in results)
        mock_mc.score.assert_called_once_with(docs=["A", "C"], claims=["B", "D"])

    @pytest.mark.parametrize("backend", ["deberta", "onnx", "minicheck"])
    def test_valid_backends_accepted(self, backend):
        scorer = NLIScorer(backend=backend, use_model=False)
        assert scorer.backend == backend

    @pytest.mark.parametrize(
        "mc_score,expected_divergence",
        [
            (1.0, 0.0),
            (0.5, 0.5),
            (0.0, 1.0),
        ],
    )
    def test_minicheck_score_to_divergence(self, mc_score, expected_divergence):
        mock_mc = MagicMock()
        mock_mc.score.return_value = [mc_score]
        scorer = NLIScorer(backend="minicheck")
        scorer._minicheck = mock_mc
        scorer._minicheck_loaded = True
        result = scorer.score("A", "B")
        assert result == pytest.approx(expected_divergence)


class TestMiniCheckPipelineIntegration:
    """Verify MiniCheck wires into CoherenceScorer pipeline."""

    def test_scorer_with_minicheck_backend(self):
        from director_ai.core import CoherenceScorer

        scorer = CoherenceScorer(use_nli=False, scorer_backend="minicheck")
        approved, score = scorer.review("test", "test")
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0


class TestMiniCheckPerformanceDoc:
    """Document MiniCheck backend performance."""

    def test_heuristic_fallback_fast(self):
        import time

        scorer = NLIScorer(backend="minicheck", use_model=False)
        t0 = time.perf_counter()
        for _ in range(100):
            scorer.score("test", "test")
        per_call_ms = (time.perf_counter() - t0) / 100 * 1000
        assert per_call_ms < 1.0
