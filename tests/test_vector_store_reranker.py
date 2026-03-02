# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Vector Store Reranker Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

from unittest.mock import MagicMock, patch

from director_ai.core.vector_store import InMemoryBackend, RerankedBackend


class _MockCrossEncoder:
    """Simulates sentence_transformers.CrossEncoder.predict()."""

    def predict(self, pairs):
        return [float(len(pairs) - i) for i in range(len(pairs))]


def _make_reranked(base, top_k_multiplier=3):
    """Build RerankedBackend with mocked sentence-transformers import."""
    mock_st = MagicMock()
    mock_st.CrossEncoder.return_value = _MockCrossEncoder()
    with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
        return RerankedBackend(base, top_k_multiplier=top_k_multiplier)


class TestRerankedBackend:
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
        base.query.assert_called_once_with("test", n_results=6)
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

    def test_empty_query_returns_empty(self):
        base = MagicMock()
        base.query.return_value = []
        reranker = _make_reranked(base)
        results = reranker.query("test", n_results=3)
        assert results == []
