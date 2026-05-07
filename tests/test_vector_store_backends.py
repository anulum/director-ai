# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vector Store Backend Tests
"""Multi-angle tests for vector store backend pipeline."""

from types import SimpleNamespace
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

    def test_pinecone_query_applies_tenant_filter_and_missing_namespace_count(self):
        from director_ai.core.vector_store import PineconeBackend

        backend = PineconeBackend.__new__(PineconeBackend)
        backend._embed_fn = lambda text: [0.1, 0.2]
        backend._index = MagicMock()
        backend._namespace = "tenant-ns"
        backend._texts = {}
        backend._index.query.return_value = {"matches": []}
        backend._index.describe_index_stats.return_value = {"namespaces": {}}

        assert backend.query("q", tenant_id="tenant-a") == []
        assert backend._index.query.call_args.kwargs["filter"] == {
            "tenant_id": {"$eq": "tenant-a"}
        }
        assert backend.count() == 0


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

    def test_weaviate_init_uses_api_key_auth(self):
        mock_weaviate = MagicMock()

        with patch.dict("sys.modules", {"weaviate": mock_weaviate}):
            from director_ai.core.vector_store import WeaviateBackend

            WeaviateBackend(url="http://weaviate:8080", api_key="secret")

        mock_weaviate.auth.AuthApiKey.assert_called_once_with("secret")
        mock_weaviate.Client.assert_called_once_with(
            url="http://weaviate:8080",
            auth_client_secret=mock_weaviate.auth.AuthApiKey.return_value,
        )

    def test_weaviate_query_applies_tenant_filter_and_doc_id_fallback(self):
        from director_ai.core.vector_store import WeaviateBackend

        backend = WeaviateBackend.__new__(WeaviateBackend)
        backend._client = MagicMock()
        backend._class_name = "Fact"
        backend._embed_fn = None
        backend._count = 0

        query = MagicMock()
        backend._client.query.get.return_value = query
        query.with_near_text.return_value = query
        query.with_limit.return_value = query
        query.with_additional.return_value = query
        query.with_where.return_value = query
        query.do.return_value = {
            "data": {
                "Get": {
                    "Fact": [
                        {
                            "text": "tenant fact",
                            "doc_id": "doc-1",
                            "tenant_id": "tenant-a",
                            "_additional": {"distance": 0.25},
                        }
                    ]
                }
            }
        }

        result = backend.query("tenant", tenant_id="tenant-a")

        assert result == [
            {
                "id": "doc-1",
                "text": "tenant fact",
                "distance": 0.25,
                "metadata": {
                    "text": "tenant fact",
                    "doc_id": "doc-1",
                    "tenant_id": "tenant-a",
                },
            }
        ]
        query.with_where.assert_called_once_with(
            {"path": ["tenant_id"], "operator": "Equal", "valueText": "tenant-a"}
        )


class TestQdrantBackend:
    def test_qdrant_import_error(self):
        with (
            patch.dict("sys.modules", {"qdrant_client": None}),
            pytest.raises(ImportError, match="QdrantBackend requires qdrant-client"),
        ):
            from director_ai.core.vector_store import QdrantBackend

            QdrantBackend(url="localhost")

    def test_ensure_collection_creates_missing_collection_and_query_filters(self):
        from director_ai.core.vector_store import QdrantBackend

        mock_models = MagicMock()

        class FakeFilter:
            def __init__(self, *, must):
                self.must = must

        mock_models.Filter = FakeFilter
        mock_models.FieldCondition = lambda **kwargs: ("field", kwargs)
        mock_models.MatchValue = lambda **kwargs: ("match", kwargs)
        mock_models.VectorParams = lambda **kwargs: ("vector", kwargs)
        mock_models.Distance.COSINE = "cosine"

        with patch.dict("sys.modules", {"qdrant_client.models": mock_models}):
            backend = QdrantBackend.__new__(QdrantBackend)
            backend._client = MagicMock()
            backend._client.get_collection.side_effect = [RuntimeError("missing")]
            backend._collection = "facts"
            backend._vector_size = 7
            backend._embed_fn = lambda text: [0.1] * 7

            backend._ensure_collection()
            backend._client.create_collection.assert_called_once()

            hit = SimpleNamespace(
                id=123,
                score=0.4,
                payload={"tenant_id": "tenant-a", "text": "answer"},
            )
            backend._client.search.return_value = [hit]
            result = backend.query("question", tenant_id="tenant-a")

        assert result[0]["id"] == "123"
        assert result[0]["distance"] == pytest.approx(0.6)
        query_filter = backend._client.search.call_args.kwargs["query_filter"]
        assert isinstance(query_filter, FakeFilter)
        assert query_filter.must[0][1]["key"] == "tenant_id"


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

    def test_faiss_ivf_trains_once_and_skips_invalid_indices(self):
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = MagicMock()
        mock_faiss.IndexIVFFlat.return_value = mock_index
        mock_faiss.normalize_L2 = MagicMock()
        mock_index.search.return_value = (
            [[0.99, 0.7, 0.6]],
            [[-1, 5, 0]],
        )

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from director_ai.core.vector_store import FAISSBackend

            backend = FAISSBackend(
                embed_fn=lambda text: [0.2, 0.8],
                vector_size=2,
                index_type="ivf",
            )
            backend.add("doc-1", "alpha", metadata={"tenant_id": "tenant-a"})
            backend.add("doc-2", "beta", metadata={"tenant_id": "tenant-b"})
            result = backend.query("alpha", n_results=1)

        mock_index.train.assert_called_once()
        assert backend._trained is True
        assert result == [
            {
                "id": "doc-1",
                "text": "alpha",
                "metadata": {"tenant_id": "tenant-a"},
                "distance": pytest.approx(0.4),
            }
        ]

    def test_faiss_query_handles_no_usable_hits_and_collects_multiple_results(self):
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.normalize_L2 = MagicMock()

        with patch.dict("sys.modules", {"faiss": mock_faiss}):
            from director_ai.core.vector_store import FAISSBackend

            backend = FAISSBackend(embed_fn=lambda text: [0.1, 0.2], vector_size=2)
            backend.add("doc-1", "alpha")
            mock_index.search.return_value = ([[0.3]], [[99]])
            assert backend.query("alpha") == []

            backend.add("doc-2", "beta")
            mock_index.search.return_value = ([[0.9, 0.8]], [[0, 1]])
            result = backend.query("alpha", n_results=2)

        assert [doc["id"] for doc in result] == ["doc-1", "doc-2"]


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

    def test_elasticsearch_api_key_embedding_and_hybrid_filter_paths(self):
        mock_es_mod = MagicMock()
        mock_client = MagicMock()
        mock_es_mod.Elasticsearch.return_value = mock_client
        mock_client.indices.exists.return_value = False
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_id": "d1",
                        "_score": None,
                        "_source": {
                            "text": "dense filtered",
                            "doc_id": "d1",
                            "tenant_id": "tenant-a",
                            "embedding": [0.1, 0.2],
                        },
                    }
                ]
            }
        }

        with patch.dict("sys.modules", {"elasticsearch": mock_es_mod}):
            from director_ai.core.vector_store import ElasticsearchBackend

            backend = ElasticsearchBackend(
                url="http://es:9200",
                api_key="key",
                embed_fn=lambda text: [0.1, 0.2],
                vector_size=2,
                hybrid_weight=-0.5,
            )
            backend.add("d1", "dense filtered", {"tenant_id": "tenant-a"})
            result = backend.query("dense", n_results=2, tenant_id="tenant-a")

        mock_es_mod.Elasticsearch.assert_called_once_with(
            hosts=["http://es:9200"],
            api_key="key",
        )
        mappings = mock_client.indices.create.call_args.kwargs["mappings"]
        assert mappings["properties"]["embedding"]["dims"] == 2
        indexed = mock_client.index.call_args.kwargs["document"]
        assert indexed["embedding"] == [0.1, 0.2]
        search_kwargs = mock_client.search.call_args.kwargs
        assert search_kwargs["knn"]["filter"] == {
            "bool": {"filter": [{"term": {"tenant_id": "tenant-a"}}]}
        }
        assert result[0]["distance"] == 1.0
        assert "embedding" not in result[0]["metadata"]

    def test_elasticsearch_existing_index_and_clamped_bm25_query_with_vector(self):
        mock_es_mod = MagicMock()
        mock_client = MagicMock()
        mock_es_mod.Elasticsearch.return_value = mock_client
        mock_client.indices.exists.return_value = True
        mock_client.search.return_value = {"hits": {"hits": []}}

        with patch.dict("sys.modules", {"elasticsearch": mock_es_mod}):
            from director_ai.core.vector_store import ElasticsearchBackend

            backend = ElasticsearchBackend(
                embed_fn=lambda text: [0.3],
                hybrid_weight=2.0,
            )
            assert backend._hybrid_weight == 1.0
            assert backend.query("plain", tenant_id="tenant-a") == []

        mock_client.indices.create.assert_not_called()
        assert "knn" not in mock_client.search.call_args.kwargs
        assert mock_client.search.call_args.kwargs["query"]["bool"]["filter"] == [
            {"term": {"tenant_id": "tenant-a"}}
        ]

    def test_elasticsearch_creates_text_only_index_without_embedding_mapping(self):
        mock_es_mod = MagicMock()
        mock_client = MagicMock()
        mock_es_mod.Elasticsearch.return_value = mock_client
        mock_client.indices.exists.return_value = False

        with patch.dict("sys.modules", {"elasticsearch": mock_es_mod}):
            from director_ai.core.vector_store import ElasticsearchBackend

            ElasticsearchBackend(embed_fn=None)

        mappings = mock_client.indices.create.call_args.kwargs["mappings"]
        assert "embedding" not in mappings["properties"]


class TestColBERTBackend:
    def test_colbert_import_error(self):
        with (
            patch.dict("sys.modules", {"ragatouille": None}),
            pytest.raises(ImportError, match="ColBERTBackend requires ragatouille"),
        ):
            from director_ai.core.vector_store import ColBERTBackend

            ColBERTBackend()

    def test_colbert_indexes_once_searches_and_counts_with_metadata(self):
        fake_model = MagicMock()
        fake_model.search.return_value = [
            {"document_id": "doc-2", "content": "second text", "score": 0.85},
            {"document_id": "missing", "content": "unknown text", "score": 0.25},
        ]
        ragatouille = MagicMock()
        ragatouille.RAGPretrainedModel.from_pretrained.return_value = fake_model

        with patch.dict("sys.modules", {"ragatouille": ragatouille}):
            from director_ai.core.vector_store import ColBERTBackend

            backend = ColBERTBackend(
                model_name="colbert-test",
                index_name="director-test",
                persist_dir="/tmp/index",
            )
            assert backend.query("empty") == []
            backend.add("doc-1", "first text", {"tenant_id": "tenant-a"})
            backend.add("doc-2", "second text", {"tenant_id": "tenant-b"})
            assert backend.count() == 2
            results = backend.query("second", n_results=2)
            second = backend.query("second", n_results=1)

        ragatouille.RAGPretrainedModel.from_pretrained.assert_called_once_with(
            "colbert-test"
        )
        fake_model.index.assert_called_once_with(
            collection=["first text", "second text"],
            document_ids=["doc-1", "doc-2"],
            index_name="director-test",
            split_documents=False,
            use_faiss=True,
        )
        assert results[0] == {
            "text": "second text",
            "distance": pytest.approx(0.15),
            "metadata": {"tenant_id": "tenant-b", "doc_id": "doc-2"},
        }
        assert results[1]["metadata"] == {"doc_id": "missing"}
        assert second[0]["text"] == "second text"

    def test_colbert_index_without_persist_dir_uses_plain_index_kwargs(self):
        fake_model = MagicMock()
        fake_model.search.return_value = []
        ragatouille = MagicMock()
        ragatouille.RAGPretrainedModel.from_pretrained.return_value = fake_model

        with patch.dict("sys.modules", {"ragatouille": ragatouille}):
            from director_ai.core.vector_store import ColBERTBackend

            backend = ColBERTBackend(index_name="plain-index", persist_dir="")
            backend.add("doc-1", "plain text")
            assert backend.query("plain") == []

        fake_model.index.assert_called_once_with(
            collection=["plain text"],
            document_ids=["doc-1"],
            index_name="plain-index",
            split_documents=False,
        )


class TestRemanentiaVectorBackend:
    class _Response:
        def __init__(self, status=200, body=b'{"results": []}'):
            self.status = status
            self._body = body

        def read(self):
            return self._body

    class _Connection:
        instances = []
        response = None
        request_error = None

        def __init__(self, host, *, port=None, timeout=None):
            self.host = host
            self.port = port
            self.timeout = timeout
            self.requests = []
            self.closed = False
            type(self).instances.append(self)

        def request(self, method, path, *, body=b"", headers=None):
            if type(self).request_error is not None:
                raise type(self).request_error
            self.requests.append(
                {
                    "method": method,
                    "path": path,
                    "body": body,
                    "headers": dict(headers or {}),
                }
            )

        def getresponse(self):
            assert type(self).response is not None
            return type(self).response

        def close(self):
            self.closed = True

    def setup_method(self):
        self._Connection.instances = []
        self._Connection.response = self._Response()
        self._Connection.request_error = None

    def test_rejects_invalid_configuration(self):
        from director_ai.core.vector_store import RemanentiaVectorBackend

        with pytest.raises(ValueError, match="timeout_s must be > 0"):
            RemanentiaVectorBackend(timeout_s=0)
        with pytest.raises(ValueError, match="scheme must be http or https"):
            RemanentiaVectorBackend("ftp://127.0.0.1:8001")
        with pytest.raises(ValueError, match="include a host"):
            RemanentiaVectorBackend("http:///missing-host")
        with pytest.raises(ValueError, match="must not include params"):
            RemanentiaVectorBackend("http://127.0.0.1:8001/search?debug=1")

    def test_read_only_add_raises_clear_error(self):
        from director_ai.core.vector_store import (
            RemanentiaBackendError,
            RemanentiaVectorBackend,
        )

        backend = RemanentiaVectorBackend()

        with pytest.raises(RemanentiaBackendError, match="read-only"):
            backend.add("doc-1", "text")

    def test_query_posts_public_search_payload_and_normalizes_results(self):
        from director_ai.core.vector_store import RemanentiaVectorBackend

        self._Connection.response = self._Response(
            body=(
                b'{"results": ['
                b'{"chunk_id": "chunk-1", "text": "grounding", "score": 0.82, '
                b'"metadata": {"source": "paper"}},'
                b'{"chunk_id": 2, "text": 123, "score": true, "metadata": []}'
                b"]}"
            )
        )

        with patch("http.client.HTTPConnection", self._Connection):
            backend = RemanentiaVectorBackend("http://127.0.0.1:8001/api")
            results = backend.query("claim", n_results=2, tenant_id="tenant-a")

        request = self._Connection.instances[0].requests[0]
        assert request["method"] == "POST"
        assert request["path"] == "/api/vector/search/public"
        assert request["headers"] == {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        assert request["body"] == (
            b'{"query": "claim", "top_k": 2, "source": "tenant-a"}'
        )
        assert self._Connection.instances[0].closed is True
        assert results == [
            {
                "id": "chunk-1",
                "text": "grounding",
                "distance": pytest.approx(0.18),
                "metadata": {"source": "paper"},
            },
            {"id": "2", "text": "123", "distance": 1.0, "metadata": {}},
        ]

    def test_query_uses_configured_source_before_tenant_and_allows_zero_limit(self):
        from director_ai.core.vector_store import RemanentiaVectorBackend

        with patch("http.client.HTTPConnection", self._Connection):
            backend = RemanentiaVectorBackend(source="curated-public")
            assert backend.query("claim", n_results=0, tenant_id="tenant-a") == []
            backend.query("claim", n_results=1, tenant_id="tenant-a")

        request = self._Connection.instances[0].requests[0]
        assert request["body"] == (
            b'{"query": "claim", "top_k": 1, "source": "curated-public"}'
        )

    def test_count_uses_status_endpoint_and_https_connection(self):
        from director_ai.core.vector_store import RemanentiaVectorBackend

        self._Connection.response = self._Response(
            body=b'{"semantic_memories": "4", "episodic_traces": 6}'
        )

        with patch("http.client.HTTPSConnection", self._Connection):
            backend = RemanentiaVectorBackend(
                "https://remanentia.local:8443",
                timeout_s=7.5,
            )
            assert backend.count() == 10

        conn = self._Connection.instances[0]
        assert conn.host == "remanentia.local"
        assert conn.port == 8443
        assert conn.timeout == 7.5
        assert conn.requests[0] == {
            "method": "GET",
            "path": "/status",
            "body": b"",
            "headers": {"Accept": "application/json"},
        }

    @pytest.mark.parametrize(
        ("body", "match"),
        [
            (b'{"items": []}', "missing results"),
            (b'{"results": {}}', "missing results"),
            (b'{"results": ["bad"]}', "result must be an object"),
        ],
    )
    def test_query_rejects_malformed_success_payloads(self, body, match):
        from director_ai.core.vector_store import (
            RemanentiaBackendError,
            RemanentiaVectorBackend,
        )

        self._Connection.response = self._Response(body=body)

        with patch("http.client.HTTPConnection", self._Connection):
            backend = RemanentiaVectorBackend()
            with pytest.raises(RemanentiaBackendError, match=match):
                backend.query("claim")

    @pytest.mark.parametrize(
        ("response", "match"),
        [
            (_Response(status=503, body=b"offline"), "HTTP 503"),
            (_Response(body=b"{not-json"), "invalid JSON"),
            (_Response(body=b"[]"), "response must be an object"),
        ],
    )
    def test_request_json_rejects_bad_api_responses(self, response, match):
        from director_ai.core.vector_store import (
            RemanentiaBackendError,
            RemanentiaVectorBackend,
        )

        self._Connection.response = response

        with patch("http.client.HTTPConnection", self._Connection):
            backend = RemanentiaVectorBackend()
            with pytest.raises(RemanentiaBackendError, match=match):
                backend.count()

    def test_request_json_wraps_transport_errors_and_closes_connection(self):
        from director_ai.core.vector_store import (
            RemanentiaBackendError,
            RemanentiaVectorBackend,
        )

        self._Connection.request_error = TimeoutError("slow")

        with patch("http.client.HTTPConnection", self._Connection):
            backend = RemanentiaVectorBackend()
            with pytest.raises(RemanentiaBackendError, match="request failed"):
                backend.count()

        assert self._Connection.instances[0].closed is True


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
