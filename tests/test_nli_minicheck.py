# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — MiniCheck NLI Backend Tests
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


class TestGetMinicheckScorer:
    """Coverage for CoherenceScorer._get_minicheck_scorer (scorer.py:832-851)."""

    def test_returns_none_when_minicheck_unavailable(self):
        from director_ai.core.scoring.scorer import CoherenceScorer

        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        result = scorer._get_minicheck_scorer()
        assert result is None
        assert scorer._minicheck_nli is None

    def test_caches_none_on_failure(self):
        from director_ai.core.scoring.scorer import CoherenceScorer

        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        scorer._get_minicheck_scorer()
        assert hasattr(scorer, "_minicheck_nli")
        assert scorer._minicheck_nli is None
        # Second call returns cached None without retry
        result = scorer._get_minicheck_scorer()
        assert result is None

    def test_returns_cached_value_on_subsequent_calls(self):
        from director_ai.core.scoring.scorer import CoherenceScorer

        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        sentinel = MagicMock()
        scorer._minicheck_nli = sentinel
        assert scorer._get_minicheck_scorer() is sentinel

    def test_caches_successful_scorer(self):
        from director_ai.core.scoring.scorer import CoherenceScorer

        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        mock_mc = MagicMock(spec=NLIScorer)
        mock_mc._ensure_minicheck.return_value = True
        with patch("director_ai.core.scoring.scorer.NLIScorer", return_value=mock_mc):
            result = scorer._get_minicheck_scorer()
        assert result is mock_mc
        assert scorer._minicheck_nli is mock_mc


class TestMinicheckClaimCoverage:
    """Coverage for CoherenceScorer._minicheck_claim_coverage (scorer.py:853-879)."""

    def test_all_supported(self):
        from director_ai.core.scoring.scorer import CoherenceScorer

        mock_scorer = MagicMock(spec=NLIScorer)
        mock_scorer.score.return_value = 0.1  # low divergence = supported
        coverage, divs, sents = CoherenceScorer._minicheck_claim_coverage(
            mock_scorer, "Source text.", "First sentence. Second sentence."
        )
        assert len(sents) >= 2
        assert coverage == 1.0
        assert all(d == 0.1 for d in divs)

    def test_none_supported(self):
        from director_ai.core.scoring.scorer import CoherenceScorer

        mock_scorer = MagicMock(spec=NLIScorer)
        mock_scorer.score.return_value = 0.9  # high divergence = unsupported
        coverage, divs, sents = CoherenceScorer._minicheck_claim_coverage(
            mock_scorer, "Source text.", "First sentence. Second sentence."
        )
        assert coverage == 0.0
        assert all(d == 0.9 for d in divs)

    def test_partial_support(self):
        from director_ai.core.scoring.scorer import CoherenceScorer

        mock_scorer = MagicMock(spec=NLIScorer)
        mock_scorer.score.side_effect = [0.1, 0.9]
        coverage, divs, sents = CoherenceScorer._minicheck_claim_coverage(
            mock_scorer, "Source text.", "Good claim. Bad claim."
        )
        assert coverage == pytest.approx(0.5)
        assert divs == [0.1, 0.9]

    def test_empty_summary(self):
        from director_ai.core.scoring.scorer import CoherenceScorer

        mock_scorer = MagicMock(spec=NLIScorer)
        coverage, divs, sents = CoherenceScorer._minicheck_claim_coverage(
            mock_scorer, "Source text.", ""
        )
        assert coverage == 1.0
        assert divs == []
        assert sents == []

    def test_nltk_import_fallback(self):
        """When nltk is unavailable, falls back to period-splitting."""
        import sys

        from director_ai.core.scoring.scorer import CoherenceScorer

        mock_scorer = MagicMock(spec=NLIScorer)
        mock_scorer.score.return_value = 0.2
        with patch.dict(sys.modules, {"nltk": None, "nltk.tokenize": None}):
            coverage, divs, sents = CoherenceScorer._minicheck_claim_coverage(
                mock_scorer, "Source.", "First. Second."
            )
        assert len(sents) == 2
        assert coverage == 1.0
