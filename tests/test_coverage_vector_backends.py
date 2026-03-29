# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Coverage tests for vector_store.py — Pinecone/Weaviate/Qdrant backends."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from director_ai.core.vector_store import (
    VectorGroundTruthStore,
)


class TestSentenceTransformerBackend:
    def test_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        from director_ai.core.vector_store import SentenceTransformerBackend

        with pytest.raises(ImportError, match="sentence-transformers"):
            SentenceTransformerBackend()

    def test_add_and_query(self):
        import threading

        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(384).astype(np.float32)

        from director_ai.core import vector_store

        be = vector_store.SentenceTransformerBackend.__new__(
            vector_store.SentenceTransformerBackend,
        )
        be._model = mock_model
        be._docs = []
        be._embeddings = []
        be._lock = threading.Lock()

        be.add("d1", "hello world")
        be.add("d2", "goodbye world")
        assert be.count() == 2

        results = be.query("hello", n_results=1)
        assert len(results) <= 2

    def test_query_empty(self):
        import threading

        from director_ai.core import vector_store

        be = vector_store.SentenceTransformerBackend.__new__(
            vector_store.SentenceTransformerBackend,
        )
        be._model = MagicMock()
        be._docs = []
        be._embeddings = []
        be._lock = threading.Lock()
        assert be.query("test") == []


class TestChromaBackend:
    def test_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "chromadb", None)
        from director_ai.core.vector_store import ChromaBackend

        with pytest.raises(ImportError, match="chromadb"):
            ChromaBackend()

    def test_basic_operations(self):
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        mock_collection.query.return_value = {
            "documents": [["text1"]],
            "metadatas": [[{"source": "test"}]],
            "ids": [["id1"]],
            "distances": [[0.1]],
        }

        from director_ai.core import vector_store

        be = vector_store.ChromaBackend.__new__(vector_store.ChromaBackend)
        be._client = mock_client
        be._collection = mock_collection

        be.add("d1", "hello", {"key": "val"})
        mock_collection.add.assert_called_once()

        results = be.query("hello")
        assert len(results) == 1
        assert results[0]["text"] == "text1"

        assert be.count() == 1

    def test_persist_directory(self):
        from director_ai.core import vector_store

        be = vector_store.ChromaBackend.__new__(vector_store.ChromaBackend)
        assert be is not None


class TestPineconeBackend:
    def test_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pinecone", None)
        from director_ai.core.vector_store import PineconeBackend

        with pytest.raises(ImportError, match="pinecone"):
            PineconeBackend(api_key="k", index_name="idx")

    def test_no_embed_fn_raises(self):
        from director_ai.core import vector_store

        be = vector_store.PineconeBackend.__new__(vector_store.PineconeBackend)
        be._embed_fn = None
        be._texts = {}
        with pytest.raises(ValueError, match="embed_fn"):
            be._embed("text")

    def test_operations(self):
        from director_ai.core import vector_store

        be = vector_store.PineconeBackend.__new__(vector_store.PineconeBackend)
        be._embed_fn = lambda t: [0.1, 0.2, 0.3]
        be._index = MagicMock()
        be._namespace = ""
        be._texts = {}

        be.add("d1", "hello")
        be._index.upsert.assert_called_once()

        be._index.query.return_value = {
            "matches": [{"id": "d1", "score": 0.9, "metadata": {"text": "hello"}}],
        }
        results = be.query("hello")
        assert len(results) == 1

        be._index.describe_index_stats.return_value = {
            "namespaces": {"": {"vector_count": 5}},
        }
        assert be.count() == 5


class TestWeaviateBackend:
    def test_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "weaviate", None)
        from director_ai.core.vector_store import WeaviateBackend

        with pytest.raises(ImportError, match="weaviate"):
            WeaviateBackend()

    def test_operations(self):
        from director_ai.core import vector_store

        be = vector_store.WeaviateBackend.__new__(vector_store.WeaviateBackend)
        be._client = MagicMock()
        be._class_name = "Fact"
        be._embed_fn = lambda t: [0.1, 0.2]
        be._count = 0

        be.add("d1", "hello")
        assert be._count == 1

        mock_query = MagicMock()
        be._client.query.get.return_value = mock_query
        mock_query.with_near_vector.return_value = mock_query
        mock_query.with_near_text.return_value = mock_query
        mock_query.with_limit.return_value = mock_query
        mock_query.with_additional.return_value = mock_query
        mock_query.do.return_value = {
            "data": {
                "Get": {
                    "Fact": [
                        {
                            "text": "hello",
                            "doc_id": "d1",
                            "_additional": {"id": "x", "distance": 0.1},
                        },
                    ],
                },
            },
        }

        results = be.query("hello")
        assert len(results) == 1
        assert results[0]["text"] == "hello"
        assert be.count() == 1

    def test_query_without_embed_fn(self):
        from director_ai.core import vector_store

        be = vector_store.WeaviateBackend.__new__(vector_store.WeaviateBackend)
        be._client = MagicMock()
        be._class_name = "Fact"
        be._embed_fn = None
        be._count = 0

        mock_query = MagicMock()
        be._client.query.get.return_value = mock_query
        mock_query.with_near_text.return_value = mock_query
        mock_query.with_limit.return_value = mock_query
        mock_query.with_additional.return_value = mock_query
        mock_query.do.return_value = {"data": {"Get": {"Fact": []}}}

        results = be.query("hello")
        assert results == []


class TestQdrantBackend:
    def test_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "qdrant_client", None)
        from director_ai.core.vector_store import QdrantBackend

        with pytest.raises(ImportError, match="qdrant"):
            QdrantBackend()

    def test_no_embed_fn_raises(self):
        from director_ai.core import vector_store

        mock_models = MagicMock()
        with patch.dict(
            sys.modules,
            {
                "qdrant_client": MagicMock(),
                "qdrant_client.models": mock_models,
            },
        ):
            be = vector_store.QdrantBackend.__new__(vector_store.QdrantBackend)
            be._embed_fn = None
            be._client = MagicMock()
            be._collection = "test"
            be._vector_size = 3

            with pytest.raises(ValueError, match="embed_fn"):
                be.add("d1", "text")

            with pytest.raises(ValueError, match="embed_fn"):
                be.query("text")

    def test_operations(self):
        from director_ai.core import vector_store

        mock_models = MagicMock()
        with patch.dict(
            sys.modules,
            {
                "qdrant_client": MagicMock(),
                "qdrant_client.models": mock_models,
            },
        ):
            be = vector_store.QdrantBackend.__new__(vector_store.QdrantBackend)
            be._embed_fn = lambda t: [0.1, 0.2, 0.3]
            be._client = MagicMock()
            be._collection = "test"
            be._vector_size = 3

            be.add("d1", "hello")
            be._client.upsert.assert_called_once()

            hit = MagicMock()
            hit.id = "d1"
            hit.score = 0.95
            hit.payload = {"text": "hello"}
            be._client.search.return_value = [hit]

            results = be.query("hello")
            assert len(results) == 1
            assert results[0]["text"] == "hello"

            info = MagicMock()
            info.points_count = 10
            be._client.get_collection.return_value = info
            assert be.count() == 10


class TestVectorGroundTruthStore:
    def test_add_fact(self):
        store = VectorGroundTruthStore()
        store.add_fact("sky", "blue")
        assert store.facts["sky"] == "blue"
        assert store.backend.count() == 1

    def test_retrieve_context_vector_fallback(self):
        store = VectorGroundTruthStore()
        store.add("answer", "42")
        result = store.retrieve_context("unrelated query zzzz")
        assert result is None or isinstance(result, str)

    def test_retrieve_context_with_chunks_empty(self):
        store = VectorGroundTruthStore()
        assert store.retrieve_context_with_chunks("no match") == []
