# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vector Store Backend Tests (STRONG)
"""Multi-angle tests for vector store backend pipeline (STRONG)."""

from unittest.mock import MagicMock, patch

import pytest


class TestPineconeBackend:
    def test_pinecone_import_error(self):
        with (
            patch.dict("sys.modules", {"pinecone": None}),
            pytest.raises(ImportError, match="PineconeBackend requires pinecone"),
        ):
            from director_ai.core.vector_store import PineconeBackend

            PineconeBackend(api_key="key", index_name="idx")

    def test_pinecone_add_query(self):
        mock_pinecone = MagicMock()
        mock_index = MagicMock()
        mock_pinecone.Pinecone.return_value.Index.return_value = mock_index
        mock_index.query.return_value = {
            "matches": [
                {"id": "d1", "score": 0.9, "metadata": {"text": "hello world"}}
            ],
        }

        with patch.dict("sys.modules", {"pinecone": mock_pinecone}):
            from director_ai.core.vector_store import PineconeBackend

            def embed_fn(text):
                return [0.1, 0.2, 0.3]

            backend = PineconeBackend(api_key="k", index_name="i", embed_fn=embed_fn)
            backend.add("d1", "hello world")
            mock_index.upsert.assert_called_once()

            results = backend.query("hello", n_results=1)
            assert len(results) == 1
            assert results[0]["id"] == "d1"
            assert results[0]["distance"] == pytest.approx(0.1)


class TestWeaviateBackend:
    def test_weaviate_import_error(self):
        with (
            patch.dict("sys.modules", {"weaviate": None}),
            pytest.raises(
                ImportError,
                match="WeaviateBackend requires weaviate-client",
            ),
        ):
            from director_ai.core.vector_store import WeaviateBackend

            WeaviateBackend(url="http://localhost:8080")


class TestQdrantBackend:
    def test_qdrant_import_error(self):
        with (
            patch.dict("sys.modules", {"qdrant_client": None}),
            pytest.raises(ImportError, match="QdrantBackend requires qdrant-client"),
        ):
            from director_ai.core.vector_store import QdrantBackend

            QdrantBackend(url="localhost")


class TestFAISSBackend:
    def test_faiss_import_error(self):
        with (
            patch.dict("sys.modules", {"faiss": None}),
            pytest.raises(ImportError, match="FAISSBackend requires faiss"),
        ):
            from director_ai.core.vector_store import FAISSBackend

            FAISSBackend(embed_fn=lambda t: [0.1] * 4)

    def test_faiss_add_query(self):
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.normalize_L2 = MagicMock()
        mock_index.search.return_value = (
            [[0.95]],
            [[0]],
        )

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from director_ai.core.vector_store import FAISSBackend

            backend = FAISSBackend(
                embed_fn=lambda t: [0.1, 0.2, 0.3, 0.4],
                vector_size=4,
            )
            backend.add("d1", "hello world")
            assert backend.count() == 1
            mock_index.add.assert_called_once()

            results = backend.query("hello", n_results=1)
            assert len(results) == 1
            assert results[0]["id"] == "d1"
            assert results[0]["distance"] == pytest.approx(0.05)

    def test_faiss_empty_returns_empty(self):
        mock_faiss = MagicMock()
        mock_faiss.IndexFlatIP.return_value = MagicMock()
        mock_faiss.normalize_L2 = MagicMock()

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from director_ai.core.vector_store import FAISSBackend

            backend = FAISSBackend(
                embed_fn=lambda t: [0.1, 0.2],
                vector_size=2,
            )
            assert backend.query("q") == []

    def test_faiss_no_embed_fn_raises(self):
        mock_faiss = MagicMock()
        mock_faiss.IndexFlatIP.return_value = MagicMock()

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from director_ai.core.vector_store import FAISSBackend

            backend = FAISSBackend(embed_fn=None, vector_size=4)
            with pytest.raises(ValueError, match="requires embed_fn"):
                backend.add("d1", "text")

    def test_faiss_tenant_filter(self):
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.normalize_L2 = MagicMock()
        mock_index.search.return_value = (
            [[0.9, 0.8]],
            [[0, 1]],
        )

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from director_ai.core.vector_store import FAISSBackend

            backend = FAISSBackend(
                embed_fn=lambda t: [0.1, 0.2],
                vector_size=2,
            )
            backend.add("d1", "a", metadata={"tenant_id": "t1"})
            backend.add("d2", "b", metadata={"tenant_id": "t2"})
            results = backend.query("q", n_results=1, tenant_id="t2")
            assert len(results) == 1
            assert results[0]["id"] == "d2"


class TestElasticsearchBackend:
    def test_elasticsearch_import_error(self):
        with (
            patch.dict("sys.modules", {"elasticsearch": None}),
            pytest.raises(
                ImportError,
                match="ElasticsearchBackend requires elasticsearch",
            ),
        ):
            from director_ai.core.vector_store import ElasticsearchBackend

            ElasticsearchBackend(url="http://localhost:9200")

    def test_elasticsearch_add_query_bm25(self):
        mock_es_mod = MagicMock()
        mock_client = MagicMock()
        mock_es_mod.Elasticsearch.return_value = mock_client
        mock_client.indices.exists.return_value = True
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_id": "d1",
                        "_score": 5.0,
                        "_source": {"text": "hello world", "doc_id": "d1"},
                    },
                ],
            },
        }

        with patch.dict("sys.modules", {"elasticsearch": mock_es_mod}):
            from director_ai.core.vector_store import ElasticsearchBackend

            backend = ElasticsearchBackend(
                url="http://localhost:9200",
                embed_fn=None,
            )
            backend.add("d1", "hello world")
            assert backend.count() == 1
            mock_client.index.assert_called_once()

            results = backend.query("hello", n_results=1)
            assert len(results) == 1
            assert results[0]["id"] == "d1"

    def test_elasticsearch_hybrid_query(self):
        mock_es_mod = MagicMock()
        mock_client = MagicMock()
        mock_es_mod.Elasticsearch.return_value = mock_client
        mock_client.indices.exists.return_value = False

        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_id": "d1",
                        "_score": 3.0,
                        "_source": {"text": "dense match", "doc_id": "d1"},
                    },
                ],
            },
        }

        with patch.dict("sys.modules", {"elasticsearch": mock_es_mod}):
            from director_ai.core.vector_store import ElasticsearchBackend

            backend = ElasticsearchBackend(
                embed_fn=lambda t: [0.1, 0.2, 0.3],
                vector_size=3,
                hybrid_weight=0.5,
            )
            mock_client.indices.create.assert_called_once()

            results = backend.query("test", n_results=1)
            assert len(results) == 1
            call_kwargs = mock_client.search.call_args
            assert "knn" in call_kwargs.kwargs

    def test_elasticsearch_tenant_filter(self):
        mock_es_mod = MagicMock()
        mock_client = MagicMock()
        mock_es_mod.Elasticsearch.return_value = mock_client
        mock_client.indices.exists.return_value = True
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_id": "d1",
                        "_score": 2.0,
                        "_source": {
                            "text": "filtered",
                            "doc_id": "d1",
                            "tenant_id": "t1",
                        },
                    },
                ],
            },
        }

        with patch.dict("sys.modules", {"elasticsearch": mock_es_mod}):
            from director_ai.core.vector_store import ElasticsearchBackend

            backend = ElasticsearchBackend(embed_fn=None)
            results = backend.query("test", tenant_id="t1")
            call_kwargs = mock_client.search.call_args.kwargs
            query = call_kwargs.get("query", {})
            assert "bool" in query
            assert len(results) == 1


class TestBackendImportErrors:
    def test_all_backends_give_clear_import_messages(self):
        backends = [
            ("pinecone", "PineconeBackend", "pinecone"),
            ("weaviate", "WeaviateBackend", "weaviate-client"),
            ("qdrant_client", "QdrantBackend", "qdrant-client"),
            ("faiss", "FAISSBackend", "faiss"),
            ("elasticsearch", "ElasticsearchBackend", "elasticsearch"),
        ]
        for module, cls_name, pkg_name in backends:
            with (
                patch.dict("sys.modules", {module: None}),
                pytest.raises(ImportError, match=pkg_name),
            ):
                from director_ai.core import vector_store

                kwargs: dict = (
                    {"api_key": "k", "index_name": "i"}
                    if cls_name == "PineconeBackend"
                    else {"url": "http://localhost:8080"}
                    if cls_name == "WeaviateBackend"
                    else {"url": "localhost"}
                    if cls_name == "QdrantBackend"
                    else {"embed_fn": lambda t: [0.1]}
                    if cls_name == "FAISSBackend"
                    else {"url": "http://localhost:9200"}
                )
                getattr(vector_store, cls_name)(**kwargs)
