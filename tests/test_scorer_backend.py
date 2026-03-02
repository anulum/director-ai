# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Scorer Backend Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

from director_ai.core import CoherenceScorer
from director_ai.core.nli import NLIScorer


class TestScorerBackendForwarding:
    def test_default_backend_is_deberta(self):
        scorer = CoherenceScorer(threshold=0.5, use_nli=False)
        assert scorer.scorer_backend == "deberta"

    def test_backend_param_forwarded(self):
        scorer = CoherenceScorer(
            threshold=0.5, use_nli=True, scorer_backend="minicheck"
        )
        assert scorer.scorer_backend == "minicheck"
        assert scorer._nli is not None
        assert scorer._nli.backend == "minicheck"

    def test_onnx_path_forwarded(self):
        scorer = CoherenceScorer(
            threshold=0.5,
            use_nli=True,
            scorer_backend="onnx",
            onnx_path="/tmp/fake_onnx",
        )
        assert scorer.onnx_path == "/tmp/fake_onnx"
        assert scorer._nli is not None
        assert scorer._nli._onnx_path == "/tmp/fake_onnx"


class TestHybridBackend:
    def test_hybrid_backend_requires_provider(self):
        import pytest

        with pytest.raises(
            ValueError, match="hybrid backend requires llm_judge_provider"
        ):
            CoherenceScorer(threshold=0.5, use_nli=False, scorer_backend="hybrid")

    def test_hybrid_backend_auto_enables_judge(self):
        scorer = CoherenceScorer(
            threshold=0.5,
            use_nli=False,
            scorer_backend="hybrid",
            llm_judge_provider="openai",
        )
        assert scorer._llm_judge_enabled is True

    def test_hybrid_review_calls_judge(self):
        from unittest.mock import patch

        scorer = CoherenceScorer(
            threshold=0.5,
            use_nli=False,
            scorer_backend="hybrid",
            llm_judge_provider="openai",
        )
        with patch.object(scorer, "_llm_judge_check", return_value=0.3):
            scorer.review("What color is the sky?", "The sky is blue.")
            assert scorer._llm_judge_enabled is True
            assert scorer.scorer_backend == "hybrid"


class TestNLIBatchLength:
    def test_batch_returns_correct_length(self):
        nli = NLIScorer(use_model=False, backend="deberta")
        pairs = [("premise", "hypothesis")] * 5
        results = nli.score_batch(pairs)
        assert len(results) == 5

    def test_empty_batch(self):
        nli = NLIScorer(use_model=False, backend="deberta")
        assert nli.score_batch([]) == []

    def test_minicheck_batch_fallback_length(self):
        nli = NLIScorer(use_model=False, backend="minicheck")
        pairs = [("a", "b"), ("c", "d"), ("e", "f")]
        results = nli.score_batch(pairs)
        assert len(results) == 3
