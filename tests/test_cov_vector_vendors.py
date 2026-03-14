"""Coverage for vector_store.py — vendor backend import branches."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestPineconeBackend:
    def test_pinecone_import_error(self):
        from director_ai.core.vector_store import PineconeBackend

        with (
            patch.dict(sys.modules, {"pinecone": None}),
            pytest.raises(ImportError, match="pinecone"),
        ):
            PineconeBackend(api_key="fake", index_name="test")

    def test_pinecone_success(self):
        mock_pinecone = MagicMock()
        mock_index = MagicMock()
        mock_index.describe_index_stats.return_value = {
            "namespaces": {"": {"vector_count": 0}},
        }
        mock_pinecone.Pinecone.return_value.Index.return_value = mock_index

        with patch.dict(sys.modules, {"pinecone": mock_pinecone}):
            from director_ai.core.vector_store import PineconeBackend

            backend = PineconeBackend(api_key="fake", index_name="test")
            assert backend.count() == 0


class TestWeaviateBackend:
    def test_weaviate_import_error(self):
        with patch.dict(sys.modules, {"weaviate": None}):
            from director_ai.core.vector_store import WeaviateBackend

            with pytest.raises(ImportError, match="weaviate"):
                WeaviateBackend()

    def test_weaviate_success(self):
        mock_weaviate = MagicMock()
        mock_client = MagicMock()
        mock_weaviate.Client.return_value = mock_client

        with patch.dict(sys.modules, {"weaviate": mock_weaviate}):
            from director_ai.core.vector_store import WeaviateBackend

            backend = WeaviateBackend()
            assert backend.count() == 0

    def test_weaviate_add_with_embed_fn(self):
        mock_weaviate = MagicMock()
        mock_client = MagicMock()
        mock_weaviate.Client.return_value = mock_client

        with patch.dict(sys.modules, {"weaviate": mock_weaviate}):
            from director_ai.core.vector_store import WeaviateBackend

            def embed_fn(t):
                return [0.1, 0.2, 0.3]

            backend = WeaviateBackend(embed_fn=embed_fn)
            backend.add("id1", "text1")
            assert backend.count() == 1

    def test_weaviate_query_with_embed_fn(self):
        mock_weaviate = MagicMock()
        mock_client = MagicMock()
        mock_weaviate.Client.return_value = mock_client

        mock_query = MagicMock()
        mock_client.query.get.return_value = mock_query
        mock_query.with_near_vector.return_value = mock_query
        mock_query.with_limit.return_value = mock_query
        mock_query.with_additional.return_value = mock_query
        mock_query.do.return_value = {
            "data": {
                "Get": {
                    "DirectorFact": [
                        {
                            "text": "fact",
                            "doc_id": "id1",
                            "_additional": {"distance": 0.1, "id": "id1"},
                        },
                    ],
                },
            },
        }

        with patch.dict(sys.modules, {"weaviate": mock_weaviate}):
            from director_ai.core.vector_store import WeaviateBackend

            def embed_fn(t):
                return [0.1, 0.2, 0.3]

            backend = WeaviateBackend(embed_fn=embed_fn)
            results = backend.query("test")
            assert len(results) == 1

    def test_weaviate_query_without_embed_fn(self):
        mock_weaviate = MagicMock()
        mock_client = MagicMock()
        mock_weaviate.Client.return_value = mock_client

        mock_query = MagicMock()
        mock_client.query.get.return_value = mock_query
        mock_query.with_near_text.return_value = mock_query
        mock_query.with_limit.return_value = mock_query
        mock_query.with_additional.return_value = mock_query
        mock_query.do.return_value = {"data": {"Get": {"DirectorFact": []}}}

        with patch.dict(sys.modules, {"weaviate": mock_weaviate}):
            from director_ai.core.vector_store import WeaviateBackend

            backend = WeaviateBackend()
            results = backend.query("test")
            assert results == []


class TestQdrantBackend:
    def test_qdrant_import_error(self):
        with patch.dict(sys.modules, {"qdrant_client": None}):
            from director_ai.core.vector_store import QdrantBackend

            with pytest.raises(ImportError, match="qdrant"):
                QdrantBackend()

    def test_qdrant_success(self):
        mock_qdrant = MagicMock()
        mock_models = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "qdrant_client": mock_qdrant,
                "qdrant_client.models": mock_models,
            },
        ):
            from director_ai.core.vector_store import QdrantBackend

            backend = QdrantBackend()
            assert backend is not None


class TestChromaEmbeddingModel:
    def test_chroma_embedding_model_import_error(self):
        mock_chromadb = MagicMock()
        mock_chromadb_utils = MagicMock()
        mock_ef = MagicMock()
        mock_ef.SentenceTransformerEmbeddingFunction = MagicMock(
            side_effect=ImportError("no st"),
        )

        with patch.dict(
            sys.modules,
            {
                "chromadb": mock_chromadb,
                "chromadb.utils": mock_chromadb_utils,
                "chromadb.utils.embedding_functions": mock_ef,
            },
        ):
            from director_ai.core.vector_store import ChromaBackend

            backend = ChromaBackend(embedding_model="all-MiniLM-L6-v2")
            assert backend is not None


class TestSentenceTransformerBackend:
    def test_st_backend_import_error(self):
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            from director_ai.core.vector_store import SentenceTransformerBackend

            with pytest.raises(ImportError, match="sentence-transformers"):
                SentenceTransformerBackend()
