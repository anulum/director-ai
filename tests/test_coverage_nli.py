"""Coverage tests for nli.py — NLIScorer, OnnxDynamicBatcher, helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from director_ai.core.nli import (
    NLIScorer,
    OnnxDynamicBatcher,
    _probs_to_divergence,
    _softmax_np,
    nli_available,
)


class TestNLIHelpers:
    def test_softmax_np(self):
        x = np.array([[1.0, 2.0, 3.0]])
        result = _softmax_np(x)
        assert abs(result.sum() - 1.0) < 1e-6

    def test_probs_to_divergence_2class(self):
        probs = np.array([[0.3, 0.7]])
        result = _probs_to_divergence(probs)
        assert abs(result[0] - 0.3) < 1e-6

    def test_probs_to_divergence_3class(self):
        probs = np.array([[0.1, 0.3, 0.6]])
        result = _probs_to_divergence(probs)
        # P(contradiction) + 0.5*P(neutral) = 0.6 + 0.15 = 0.75
        assert abs(result[0] - 0.75) < 1e-6

    def test_nli_available(self):
        result = nli_available()
        assert isinstance(result, bool)


class TestNLIScorerInit:
    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="backend"):
            NLIScorer(backend="nonexistent")

    def test_invalid_backend_type(self):
        with pytest.raises(TypeError, match="backend"):
            NLIScorer(backend=42)

    def test_custom_backend(self):
        from director_ai.core.backends import ScorerBackend

        class Stub(ScorerBackend):
            def score(self, p, h):
                return 0.5

            def score_batch(self, pairs):
                return [0.5] * len(pairs)

        scorer = NLIScorer(backend=Stub())
        assert scorer.score("a", "b") == 0.5

    def test_custom_backend_batch(self):
        from director_ai.core.backends import ScorerBackend

        class Stub(ScorerBackend):
            def score(self, p, h):
                return 0.5

            def score_batch(self, pairs):
                return [0.5] * len(pairs)

        scorer = NLIScorer(backend=Stub())
        assert scorer.score_batch([("a", "b")]) == [0.5]


class TestNLIScorerHeuristic:
    def test_heuristic_fallback(self):
        scorer = NLIScorer(use_model=False)
        result = scorer.score("sky is blue", "sky is blue")
        assert 0.0 <= result <= 1.0

    def test_heuristic_batch(self):
        scorer = NLIScorer(use_model=False)
        results = scorer.score_batch([("a", "b"), ("c", "d")])
        assert len(results) == 2

    def test_score_batch_empty(self):
        scorer = NLIScorer(use_model=False)
        assert scorer.score_batch([]) == []

    def test_consistent_keyword(self):
        score = NLIScorer._heuristic_score("anything", "consistent with reality here")
        assert score < 0.3

    def test_contradicted_keyword(self):
        score = NLIScorer._heuristic_score("anything", "the opposite is true")
        assert score > 0.8

    def test_perspective_keyword(self):
        score = NLIScorer._heuristic_score("anything", "it depends on your perspective")
        assert abs(score - 0.5) < 0.01

    def test_empty_premise(self):
        score = NLIScorer._heuristic_score("", "something")
        assert abs(score - 0.5) < 0.01


class TestNLIScorerLite:
    def test_lite_score(self):
        scorer = NLIScorer(use_model=False, backend="lite")
        result = scorer.score("sky is blue", "sky is blue")
        assert 0.0 <= result <= 1.0

    def test_lite_batch(self):
        scorer = NLIScorer(use_model=False, backend="lite")
        results = scorer.score_batch([("a", "b")])
        assert len(results) == 1


class TestNLIScorerChunked:
    def test_score_chunked_short(self):
        scorer = NLIScorer(use_model=False)
        agg, chunks = scorer.score_chunked("sky is blue", "sky is blue")
        assert 0.0 <= agg <= 1.0

    def test_score_decomposed_single(self):
        scorer = NLIScorer(use_model=False)
        agg, scores = scorer.score_decomposed("sky is blue", "sky is blue")
        assert len(scores) == 1

    def test_score_decomposed_multi(self):
        scorer = NLIScorer(use_model=False)
        agg, scores = scorer.score_decomposed(
            "sky is blue",
            "The sky is blue. Water is wet.",
        )
        assert len(scores) == 2

    def test_decompose_claims(self):
        scorer = NLIScorer(use_model=False)
        claims = scorer.decompose_claims("First claim. Second claim! Third?")
        assert len(claims) == 3

    def test_split_sentences(self):
        result = NLIScorer._split_sentences("Hello. World! Yes?")
        assert len(result) == 3

    def test_estimate_tokens(self):
        assert NLIScorer._estimate_tokens("hello world") > 0

    def test_build_chunks(self):
        scorer = NLIScorer(use_model=False)
        sentences = ["Sentence one.", "Sentence two.", "Sentence three."]
        chunks = scorer._build_chunks(sentences, budget=5)
        assert len(chunks) >= 1


class TestOnnxDynamicBatcher:
    def test_submit_below_threshold(self):
        fn = MagicMock(return_value=[0.5])
        batcher = OnnxDynamicBatcher(fn, max_batch=10)
        result = batcher.submit([("a", "b")])
        assert result == []

    def test_submit_triggers_flush(self):
        fn = MagicMock(return_value=[0.5] * 2)
        batcher = OnnxDynamicBatcher(fn, max_batch=2)
        result = batcher.submit([("a", "b"), ("c", "d")])
        assert result == [0.5, 0.5]
        fn.assert_called_once()

    def test_explicit_flush(self):
        fn = MagicMock(return_value=[0.5])
        batcher = OnnxDynamicBatcher(fn, max_batch=10)
        batcher.submit([("a", "b")])
        result = batcher.flush()
        assert result == [0.5]

    def test_flush_empty(self):
        fn = MagicMock()
        batcher = OnnxDynamicBatcher(fn)
        assert batcher.flush() == []

    def test_uses_io_binding_no_cuda(self):
        batcher = OnnxDynamicBatcher(lambda x: x)
        assert not batcher.uses_io_binding

    def test_uses_io_binding_with_cuda(self):
        session = MagicMock()
        session.get_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        batcher = OnnxDynamicBatcher(lambda x: x, session=session)
        assert batcher.uses_io_binding


class TestNLIScorerAsync:
    def test_ascore(self):
        import asyncio

        scorer = NLIScorer(use_model=False)

        async def run():
            return await scorer.ascore("a", "b")

        result = asyncio.get_event_loop().run_until_complete(run())
        assert 0.0 <= result <= 1.0

    def test_ascore_batch(self):
        import asyncio

        scorer = NLIScorer(use_model=False)

        async def run():
            return await scorer.ascore_batch([("a", "b")])

        result = asyncio.get_event_loop().run_until_complete(run())
        assert len(result) == 1
