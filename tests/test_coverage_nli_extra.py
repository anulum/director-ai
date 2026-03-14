"""Coverage tests for nli.py — minicheck, model loading branches."""

from __future__ import annotations

import sys
from unittest.mock import patch

from director_ai.core.nli import NLIScorer


class TestNLIScorerMinicheck:
    def test_minicheck_not_installed(self):
        scorer = NLIScorer(use_model=False, backend="minicheck")
        with patch.dict(sys.modules, {"minicheck": None}):
            scorer._minicheck_loaded = False
            scorer._minicheck = None
            result = scorer._minicheck_score("a", "b")
            assert 0.0 <= result <= 1.0

    def test_minicheck_batch_not_installed(self):
        scorer = NLIScorer(use_model=False, backend="minicheck")
        with patch.dict(sys.modules, {"minicheck": None}):
            scorer._minicheck_loaded = False
            scorer._minicheck = None
            result = scorer._minicheck_score_batch([("a", "b")])
            assert len(result) == 1


class TestNLIScorerEnsureModel:
    def test_ensure_model_no_use(self):
        scorer = NLIScorer(use_model=False)
        assert not scorer._ensure_model()

    def test_ensure_model_onnx_no_path(self):
        scorer = NLIScorer(use_model=True, backend="onnx")
        ready = scorer._ensure_model()
        assert not ready

    def test_backend_ready_custom(self):
        from director_ai.core.backends import ScorerBackend

        class Stub(ScorerBackend):
            def score(self, p, h):
                return 0.5

            def score_batch(self, pairs):
                return [0.5] * len(pairs)

        scorer = NLIScorer(backend=Stub())
        assert scorer._backend_ready

    def test_backend_ready_lite(self):
        scorer = NLIScorer(use_model=False, backend="lite")
        assert scorer._backend_ready

    def test_model_available_property(self):
        scorer = NLIScorer(use_model=False)
        assert not scorer.model_available

    def test_is_factcg(self):
        scorer = NLIScorer(use_model=False, model_name="yaxili96/FactCG-DeBERTa")
        assert scorer._is_factcg

    def test_is_not_factcg(self):
        scorer = NLIScorer(use_model=False, model_name="other/model")
        assert not scorer._is_factcg


class TestNLIScorerScoreRouting:
    def test_score_lite(self):
        scorer = NLIScorer(use_model=False, backend="lite")
        result = scorer.score("premise", "hypothesis")
        assert 0.0 <= result <= 1.0

    def test_score_batch_lite(self):
        scorer = NLIScorer(use_model=False, backend="lite")
        results = scorer.score_batch([("a", "b")])
        assert len(results) == 1

    def test_score_batch_minicheck_fallback(self):
        scorer = NLIScorer(use_model=False, backend="minicheck")
        scorer._minicheck_loaded = True
        scorer._minicheck = None
        results = scorer.score_batch([("a", "b")])
        assert len(results) == 1


class TestNLIChunkedScoring:
    def test_chunked_with_counts_short_input(self):
        scorer = NLIScorer(use_model=False)
        agg, per_hyp, pc, hc = scorer._score_chunked_with_counts("short", "short")
        assert pc == 1
        assert hc == 1

    def test_chunked_long_hypothesis(self):
        scorer = NLIScorer(use_model=False, max_length=20)
        long_hyp = ". ".join([f"Sentence {i}" for i in range(50)]) + "."
        agg, per_hyp = scorer.score_chunked("Short premise.", long_hyp)
        assert len(per_hyp) >= 1

    def test_chunked_mean_agg(self):
        scorer = NLIScorer(use_model=False, max_length=20)
        long_hyp = ". ".join([f"Sentence {i}" for i in range(20)]) + "."
        agg, per_hyp = scorer.score_chunked(
            "Short premise.",
            long_hyp,
            outer_agg="mean",
        )
        assert 0.0 <= agg <= 1.0

    def test_build_chunks_empty(self):
        scorer = NLIScorer(use_model=False)
        chunks = scorer._build_chunks([], budget=100)
        assert len(chunks) == 1
