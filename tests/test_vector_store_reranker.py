# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vector Store Reranker Tests
"""Multi-angle tests for VectorStore RerankedBackend.

Covers: reranking order, top_k multiplier, delegation, empty query,
parametrised n_results, multiplier values, scorer pipeline integration,
and performance documentation.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

import director_ai.core.retrieval.vector_store.composite as composite_mod
from director_ai.core.vector_store import InMemoryBackend, RerankedBackend


class _MockCrossEncoder:
    """Simulates sentence_transformers.CrossEncoder.predict()."""

    def predict(self, pairs):
        return [float(len(pairs) - i) for i in range(len(pairs))]


def _make_reranked(base, top_k_multiplier=3, reranker_model="test-reranker"):
    """Build RerankedBackend with mocked sentence-transformers import."""
    mock_st = MagicMock()
    mock_st.CrossEncoder.return_value = _MockCrossEncoder()
    with (
        patch.dict("os.environ", {"DIRECTOR_FORCE_CPU": "1"}),
        patch.dict("sys.modules", {"sentence_transformers": mock_st}),
    ):
        return RerankedBackend(
            base,
            reranker_model=reranker_model,
            top_k_multiplier=top_k_multiplier,
        )


class TestRerankedBackend:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"base": None}, "base"),
            ({"reranker_model": ""}, "reranker_model"),
            ({"reranker_model": "   "}, "reranker_model"),
            ({"top_k_multiplier": 0}, "top_k_multiplier"),
            ({"top_k_multiplier": 1.5}, "top_k_multiplier"),
        ],
    )
    def test_rejects_invalid_constructor_values(self, kwargs, message):
        base = kwargs.pop("base", InMemoryBackend())
        with pytest.raises(ValueError, match=message):
            _make_reranked(base, **kwargs)

    def test_reranking_reverses_order(self):
        base = InMemoryBackend()
        base.add("d1", "first doc about cats")
        base.add("d2", "second doc about cats")
        base.add("d3", "third doc about cats")

        reranker = _make_reranked(base, top_k_multiplier=3)
        results = reranker.query("doc about cats", n_results=3)
        assert len(results) == 3
        texts = [r["text"] for r in results]
        assert texts[0] == "first doc about cats"

    def test_top_k_multiplier_fetches_more(self):
        base = MagicMock()
        base.query.return_value = [
            {"id": f"d{i}", "text": f"doc{i}", "distance": float(i)} for i in range(6)
        ]
        base.count.return_value = 6

        reranker = _make_reranked(base, top_k_multiplier=3)
        results = reranker.query("test", n_results=2)
        base.query.assert_called_once_with("test", n_results=6, tenant_id="")
        assert len(results) == 2

    def test_add_delegates_to_base(self):
        base = MagicMock()
        reranker = _make_reranked(base)
        reranker.add("id1", "text1", {"key": "val"})
        base.add.assert_called_once_with("id1", "text1", {"key": "val"})

    def test_count_delegates_to_base(self):
        base = MagicMock()
        base.count.return_value = 42
        reranker = _make_reranked(base)
        assert reranker.count() == 42

    def test_cuda_oom_retries_reranker_load_on_cpu(self, monkeypatch):
        calls = []
        released = []

        class FakeCrossEncoder:
            def __init__(self, _model, *, device, revision):
                calls.append((device, revision))
                if device == "cuda:0":
                    raise RuntimeError("CUDA out of memory while allocating")

            def predict(self, pairs):
                return [1.0 for _pair in pairs]

        monkeypatch.setitem(
            sys.modules,
            "sentence_transformers",
            types.SimpleNamespace(CrossEncoder=FakeCrossEncoder),
        )
        import director_ai.core._device as device_mod

        monkeypatch.setattr(device_mod, "select_torch_device", lambda: "cuda:0")
        monkeypatch.setattr(
            device_mod,
            "release_torch_cuda",
            lambda: released.append(True),
        )

        reranker = RerankedBackend(
            InMemoryBackend(),
            reranker_model="test-reranker",
            reranker_revision="rev-1",
        )

        assert reranker.count() == 0
        assert calls == [("cuda:0", "rev-1"), ("cpu", "rev-1")]
        assert released == [True]

    def test_non_oom_reranker_load_error_propagates(self, monkeypatch):
        class FakeCrossEncoder:
            def __init__(self, _model, *, device, revision):
                raise RuntimeError(f"bad weights on {device} {revision}")

        monkeypatch.setitem(
            sys.modules,
            "sentence_transformers",
            types.SimpleNamespace(CrossEncoder=FakeCrossEncoder),
        )
        import director_ai.core._device as device_mod

        monkeypatch.setattr(device_mod, "select_torch_device", lambda: "cuda:0")

        with pytest.raises(RuntimeError, match="bad weights"):
            RerankedBackend(
                InMemoryBackend(),
                reranker_model="test-reranker",
                reranker_revision="rev-1",
            )

    def test_empty_query_returns_empty(self):
        base = MagicMock()
        base.query.return_value = []
        reranker = _make_reranked(base)
        results = reranker.query("test", n_results=3)
        assert results == []

    @pytest.mark.parametrize("n_results", [1, 2, 3, 5])
    def test_parametrised_n_results(self, n_results):
        base = MagicMock()
        base.query.return_value = [
            {"id": f"d{i}", "text": f"doc{i}", "distance": float(i)} for i in range(10)
        ]
        base.count.return_value = 10
        reranker = _make_reranked(base)
        results = reranker.query("test", n_results=n_results)
        assert len(results) == n_results

    @pytest.mark.parametrize("multiplier", [1, 2, 3, 5])
    def test_parametrised_multiplier(self, multiplier):
        base = MagicMock()
        base.query.return_value = [
            {"id": f"d{i}", "text": f"doc{i}", "distance": float(i)}
            for i in range(multiplier * 2)
        ]
        base.count.return_value = multiplier * 2
        reranker = _make_reranked(base, top_k_multiplier=multiplier)
        results = reranker.query("test", n_results=2)
        assert len(results) == 2

    def test_cuda_oom_helper_matches_supported_error_forms(self):
        assert composite_mod._is_cuda_oom(RuntimeError("CUDA out of memory")) is True
        assert (
            composite_mod._is_cuda_oom(RuntimeError("torch.OutOfMemoryError")) is True
        )
        assert composite_mod._is_cuda_oom(RuntimeError("other runtime error")) is False


class TestRerankerPipelineIntegration:
    """Verify reranker integrates into scorer pipeline."""

    def test_reranked_in_ground_truth_store(self):
        from director_ai.core.vector_store import VectorGroundTruthStore

        base = InMemoryBackend()
        base.add("d1", "Paris is the capital of France")
        reranker = _make_reranked(base)
        store = VectorGroundTruthStore(backend=reranker)
        result = store.retrieve_context("capital of France")
        assert result is not None
        assert "Paris" in result


class TestRerankerPerformanceDoc:
    """Document reranker performance characteristics."""

    def test_base_backend_count_stable(self):
        base = InMemoryBackend()
        base.add("d1", "test")
        reranker = _make_reranked(base)
        assert reranker.count() == 1
        reranker.query("test")
        assert reranker.count() == 1  # query must not modify store

    def test_add_increments_count(self):
        base = InMemoryBackend()
        reranker = _make_reranked(base)
        assert reranker.count() == 0
        reranker.add("d1", "test")
        assert reranker.count() == 1
