# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vector Store Backend Tests
"""Multi-angle tests for vector store backend pipeline."""

import sys
import types
import uuid
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

    def test_pinecone_requires_embedding_callback_before_add_or_query(self):
        from director_ai.core.vector_store import PineconeBackend

        backend = PineconeBackend.__new__(PineconeBackend)
        backend._embed_fn = None
        backend._index = MagicMock()
        backend._namespace = ""
        backend._texts = {}

        with pytest.raises(ValueError, match="requires embed_fn"):
            backend.add("doc-1", "tenant fact")
        with pytest.raises(ValueError, match="requires embed_fn"):
            backend.query("tenant")

    def test_pinecone_count_reads_configured_namespace(self):
        from director_ai.core.vector_store import PineconeBackend

        backend = PineconeBackend.__new__(PineconeBackend)
        backend._index = MagicMock()
        backend._namespace = "acme"
        backend._index.describe_index_stats.return_value = {
            "namespaces": {"acme": {"vector_count": "7"}}
        }

        assert backend.count() == 7


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

        mock_weaviate.classes.init.Auth.api_key.assert_called_once_with("secret")
        mock_weaviate.connect_to_custom.assert_called_once_with(
            http_host="weaviate",
            http_port=8080,
            http_secure=False,
            grpc_host="weaviate",
            grpc_port=50051,
            grpc_secure=False,
            auth_credentials=mock_weaviate.classes.init.Auth.api_key.return_value,
        )

    def test_weaviate_query_applies_tenant_filter_and_returns_doc_id(self):
        from director_ai.core.vector_store import WeaviateBackend

        backend = WeaviateBackend.__new__(WeaviateBackend)
        backend._client = MagicMock()
        backend._weaviate = MagicMock()
        backend._class_name = "Fact"
        backend._embed_fn = None

        obj = SimpleNamespace(
            properties={
                "text": "tenant fact",
                "doc_id": "doc-1",
                "tenant_id": "tenant-a",
            },
            uuid="obj-uuid",
            metadata=SimpleNamespace(distance=0.25),
        )
        collection = backend._client.collections.get.return_value
        collection.query.near_text.return_value = SimpleNamespace(objects=[obj])

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
        query_classes = backend._weaviate.classes.query
        query_classes.Filter.by_property.assert_called_once_with("tenant_id")
        query_classes.Filter.by_property.return_value.equal.assert_called_once_with(
            "tenant-a"
        )
        collection.query.near_text.assert_called_once_with(
            query="tenant",
            limit=3,
            filters=query_classes.Filter.by_property.return_value.equal.return_value,
            return_metadata=query_classes.MetadataQuery.return_value,
        )

    def test_weaviate_add_uses_embedding_vector_when_configured(self):
        from director_ai.core.vector_store import WeaviateBackend

        backend = WeaviateBackend.__new__(WeaviateBackend)
        backend._client = MagicMock()
        backend._class_name = "Fact"
        backend._embed_fn = lambda text: [0.4, 0.6]

        backend._weaviate = MagicMock()

        backend.add("doc-1", "embedded fact", {"tenant_id": "tenant-a"})

        collection = backend._client.collections.get.return_value
        collection.data.insert.assert_called_once_with(
            properties={
                "text": "embedded fact",
                "doc_id": "doc-1",
                "tenant_id": "tenant-a",
            },
            uuid=backend._weaviate.util.generate_uuid5.return_value,
            vector=[0.4, 0.6],
        )
        backend._weaviate.util.generate_uuid5.assert_called_once_with("doc-1")

    def test_weaviate_query_falls_back_to_object_uuid_without_doc_id(self):
        from director_ai.core.vector_store import WeaviateBackend

        backend = WeaviateBackend.__new__(WeaviateBackend)
        backend._client = MagicMock()
        backend._weaviate = MagicMock()
        backend._class_name = "Fact"
        backend._embed_fn = lambda text: [0.2, 0.8]

        obj = SimpleNamespace(
            properties={"text": "vector fact"},
            uuid="uuid-1",
            metadata=SimpleNamespace(distance=0.5),
        )
        collection = backend._client.collections.get.return_value
        collection.query.near_vector.return_value = SimpleNamespace(objects=[obj])

        result = backend.query("vector", n_results=1)

        collection.query.near_vector.assert_called_once_with(
            near_vector=[0.2, 0.8],
            limit=1,
            filters=None,
            return_metadata=backend._weaviate.classes.query.MetadataQuery.return_value,
        )
        assert result[0]["id"] == "uuid-1"
        assert result[0]["distance"] == 0.5

    def test_weaviate_init_without_key_parses_https_and_explicit_grpc(self):
        mock_weaviate = MagicMock()

        with patch.dict("sys.modules", {"weaviate": mock_weaviate}):
            from director_ai.core.vector_store import WeaviateBackend

            WeaviateBackend(url="https://secure.example", grpc_host="grpc.example")

        mock_weaviate.classes.init.Auth.api_key.assert_not_called()
        mock_weaviate.connect_to_custom.assert_called_once_with(
            http_host="secure.example",
            http_port=443,
            http_secure=True,
            grpc_host="grpc.example",
            grpc_port=50051,
            grpc_secure=True,
            auth_credentials=None,
        )

    def test_weaviate_add_without_embedding_inserts_without_vector(self):
        from director_ai.core.vector_store import WeaviateBackend

        backend = WeaviateBackend.__new__(WeaviateBackend)
        backend._client = MagicMock()
        backend._weaviate = MagicMock()
        backend._class_name = "Fact"
        backend._embed_fn = None

        backend.add("doc-1", "plain fact")

        collection = backend._client.collections.get.return_value
        collection.data.insert.assert_called_once_with(
            properties={"text": "plain fact", "doc_id": "doc-1"},
            uuid=backend._weaviate.util.generate_uuid5.return_value,
            vector=None,
        )

    def test_weaviate_query_defaults_distance_when_metadata_missing(self):
        from director_ai.core.vector_store import WeaviateBackend

        backend = WeaviateBackend.__new__(WeaviateBackend)
        backend._client = MagicMock()
        backend._weaviate = MagicMock()
        backend._class_name = "Fact"
        backend._embed_fn = None

        obj = SimpleNamespace(
            properties={"text": "no distance", "doc_id": "doc-9"},
            uuid="obj-uuid",
            metadata=SimpleNamespace(distance=None),
        )
        collection = backend._client.collections.get.return_value
        collection.query.near_text.return_value = SimpleNamespace(objects=[obj])

        result = backend.query("anything")

        assert result[0]["distance"] == 0.0
        assert result[0]["id"] == "doc-9"

    def test_weaviate_count_queries_server_aggregate(self):
        from director_ai.core.vector_store import WeaviateBackend

        backend = WeaviateBackend.__new__(WeaviateBackend)
        backend._client = MagicMock()
        backend._class_name = "Fact"
        collection = backend._client.collections.get.return_value
        collection.aggregate.over_all.return_value = SimpleNamespace(total_count=42)

        # count() reports the live server total, not an in-process counter.
        assert backend.count() == 42
        backend._client.collections.get.assert_called_once_with("Fact")
        collection.aggregate.over_all.assert_called_once_with(total_count=True)

    def test_weaviate_count_defaults_to_zero_when_total_count_none(self):
        from director_ai.core.vector_store import WeaviateBackend

        backend = WeaviateBackend.__new__(WeaviateBackend)
        backend._client = MagicMock()
        backend._class_name = "Fact"
        backend._client.collections.get.return_value.aggregate.over_all.return_value = (
            SimpleNamespace(total_count=None)
        )

        assert backend.count() == 0


class TestQdrantBackend:
    def test_qdrant_import_error(self):
        with (
            patch.dict("sys.modules", {"qdrant_client": None}),
            pytest.raises(ImportError, match="QdrantBackend requires qdrant-client"),
        ):
            from director_ai.core.vector_store import QdrantBackend

            QdrantBackend(url="localhost")

    def test_qdrant_init_constructs_client_and_ensures_collection(self):
        mock_qc = MagicMock()
        mock_models = MagicMock()
        mock_models.VectorParams = lambda **kwargs: ("vector", kwargs)
        mock_models.Distance.COSINE = "cosine"

        with patch.dict(
            "sys.modules",
            {"qdrant_client": mock_qc, "qdrant_client.models": mock_models},
        ):
            from director_ai.core.vector_store import QdrantBackend

            backend = QdrantBackend(
                url="qhost",
                port=1234,
                collection_name="facts",
                vector_size=7,
                embed_fn=lambda text: [0.0] * 7,
            )

        mock_qc.QdrantClient.assert_called_once_with(host="qhost", port=1234)
        backend._client.get_collection.assert_called_once_with("facts")
        backend._client.create_collection.assert_not_called()

    def test_qdrant_add_requires_embed_fn(self):
        from director_ai.core.vector_store import QdrantBackend

        backend = QdrantBackend.__new__(QdrantBackend)
        backend._client = MagicMock()
        backend._collection = "facts"
        backend._embed_fn = None

        # add() imports qdrant_client.models before the embed_fn guard, so the
        # module must resolve even when the qdrant extra is not installed.
        with (
            patch.dict("sys.modules", {"qdrant_client.models": MagicMock()}),
            pytest.raises(ValueError, match="requires embed_fn"),
        ):
            backend.add("d1", "text")

    def test_qdrant_add_upserts_embedded_point(self):
        from director_ai.core.vector_store import QdrantBackend

        mock_models = MagicMock()
        mock_models.PointStruct = lambda **kwargs: ("point", kwargs)

        backend = QdrantBackend.__new__(QdrantBackend)
        backend._client = MagicMock()
        backend._collection = "facts"
        backend._embed_fn = lambda text: [0.1, 0.2]

        with patch.dict("sys.modules", {"qdrant_client.models": mock_models}):
            backend.add("d1", "hello", {"k": "v"})

        backend._client.upsert.assert_called_once()
        kwargs = backend._client.upsert.call_args.kwargs
        assert kwargs["collection_name"] == "facts"
        # An arbitrary doc id is mapped to a deterministic UUID (Qdrant rejects
        # non-integer, non-UUID ids) while the original id is kept in the payload.
        _, point = kwargs["points"][0]
        assert point["id"] == str(uuid.uuid5(uuid.NAMESPACE_URL, "d1"))
        assert point["payload"] == {"text": "hello", "doc_id": "d1", "k": "v"}

    def test_qdrant_query_returns_original_doc_id_from_payload(self):
        from director_ai.core.vector_store import QdrantBackend

        backend = QdrantBackend.__new__(QdrantBackend)
        backend._client = MagicMock()
        backend._collection = "facts"
        backend._embed_fn = lambda text: [0.1, 0.2]

        hit = SimpleNamespace(
            id="9f1c0000-0000-5000-8000-000000000000",
            score=0.25,
            payload={"doc_id": "doc-2", "text": "answer"},
        )
        backend._client.query_points.return_value = SimpleNamespace(points=[hit])

        with patch.dict("sys.modules", {"qdrant_client.models": MagicMock()}):
            result = backend.query("question")

        # The caller-facing id is the original doc id from the payload, not the
        # internal UUID point id.
        assert result[0]["id"] == "doc-2"
        assert result[0]["distance"] == pytest.approx(0.75)

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
            backend._client.query_points.return_value = SimpleNamespace(points=[hit])
            result = backend.query("question", tenant_id="tenant-a")

        assert result[0]["id"] == "123"
        assert result[0]["distance"] == pytest.approx(0.6)
        query_filter = backend._client.query_points.call_args.kwargs["query_filter"]
        assert isinstance(query_filter, FakeFilter)
        assert query_filter.must[0][1]["key"] == "tenant_id"

    def test_qdrant_existing_collection_noops_and_count_reads_points(self):
        from director_ai.core.vector_store import QdrantBackend

        mock_models = MagicMock()
        mock_models.VectorParams = lambda **kwargs: ("vector", kwargs)
        mock_models.Distance.COSINE = "cosine"

        backend = QdrantBackend.__new__(QdrantBackend)
        backend._client = MagicMock()
        backend._collection = "facts"
        backend._client.get_collection.return_value = SimpleNamespace(points_count="11")

        with patch.dict("sys.modules", {"qdrant_client.models": mock_models}):
            backend._ensure_collection()

        backend._client.create_collection.assert_not_called()
        assert backend.count() == 11

    def test_qdrant_requires_embedding_callback_before_add_or_query(self):
        from director_ai.core.vector_store import QdrantBackend

        mock_models = MagicMock()
        mock_models.PointStruct = lambda **kwargs: ("point", kwargs)
        mock_models.Filter = lambda **kwargs: ("filter", kwargs)
        mock_models.FieldCondition = lambda **kwargs: ("field", kwargs)
        mock_models.MatchValue = lambda **kwargs: ("match", kwargs)

        with patch.dict("sys.modules", {"qdrant_client.models": mock_models}):
            backend = QdrantBackend.__new__(QdrantBackend)
            backend._client = MagicMock()
            backend._collection = "facts"
            backend._embed_fn = None

            with pytest.raises(ValueError, match="requires embed_fn"):
                backend.add("doc-1", "tenant fact")
            with pytest.raises(ValueError, match="requires embed_fn"):
                backend.query("tenant")

    def test_qdrant_query_uses_query_points_api(self):
        from director_ai.core.vector_store import QdrantBackend

        mock_models = MagicMock()
        mock_models.Filter = lambda **kwargs: ("filter", kwargs)
        mock_models.FieldCondition = lambda **kwargs: ("field", kwargs)
        mock_models.MatchValue = lambda **kwargs: ("match", kwargs)

        backend = QdrantBackend.__new__(QdrantBackend)
        backend._client = MagicMock()
        backend._collection = "facts"
        backend._embed_fn = lambda _t: [0.1, 0.2, 0.3]
        hit = SimpleNamespace(
            id=7, score=0.75, payload={"text": "grounded", "tenant_id": "t1"}
        )
        backend._client.query_points.return_value = SimpleNamespace(points=[hit])

        with patch.dict("sys.modules", {"qdrant_client.models": mock_models}):
            results = backend.query("question", n_results=5, tenant_id="t1")

        # qdrant-client 1.x exposes query_points, not the removed search().
        backend._client.search.assert_not_called()
        kwargs = backend._client.query_points.call_args.kwargs
        assert kwargs["query"] == [0.1, 0.2, 0.3]
        assert kwargs["limit"] == 5
        assert kwargs["query_filter"] is not None
        assert results == [
            {
                "id": "7",
                "text": "grounded",
                "distance": pytest.approx(0.25),
                "metadata": {"text": "grounded", "tenant_id": "t1"},
            }
        ]

    def test_qdrant_count_handles_none_points(self):
        from director_ai.core.vector_store import QdrantBackend

        backend = QdrantBackend.__new__(QdrantBackend)
        backend._client = MagicMock()
        backend._collection = "facts"
        backend._client.get_collection.return_value = SimpleNamespace(points_count=None)
        assert backend.count() == 0


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

    def test_elasticsearch_count_reads_server_count(self):
        from director_ai.core.vector_store import ElasticsearchBackend

        backend = ElasticsearchBackend.__new__(ElasticsearchBackend)
        backend._client = MagicMock()
        backend._index = "facts"
        backend._client.count.return_value = {"count": 5}

        # count() reflects the server's document total, not an in-process counter.
        assert backend.count() == 5
        backend._client.count.assert_called_once_with(index="facts")


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
            "id": "doc-2",
            "text": "second text",
            "distance": pytest.approx(0.15),
            "metadata": {"tenant_id": "tenant-b", "doc_id": "doc-2"},
        }
        assert results[1]["metadata"] == {"doc_id": "missing"}
        assert results[1]["id"] == "missing"
        assert second[0]["text"] == "second text"
        assert second[0]["id"] == "doc-2"

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

    def test_colbert_results_flow_through_store_with_source_id(self):
        """BUG-1 regression: ColBERT results omitted a top-level ``id`` while
        every other backend supplied one, so
        ``VectorGroundTruthStore.retrieve_context_with_chunks`` raised
        ``KeyError`` building ``source=f"vector:{r['id']}"``. The backend now
        normalises ``id`` and the real store surface yields a sourced chunk."""
        fake_model = MagicMock()
        fake_model.search.return_value = [
            {"document_id": "doc-9", "content": "grounded fact", "score": 0.9},
        ]
        ragatouille = MagicMock()
        ragatouille.RAGPretrainedModel.from_pretrained.return_value = fake_model

        with patch.dict("sys.modules", {"ragatouille": ragatouille}):
            from director_ai.core.vector_store import (
                ColBERTBackend,
                VectorGroundTruthStore,
            )

            backend = ColBERTBackend(
                model_name="colbert-test",
                index_name="director-flow",
                persist_dir="/tmp/index-flow",
            )
            backend.add("doc-9", "grounded fact")
            store = VectorGroundTruthStore(backend=backend)
            chunks = store.retrieve_context_with_chunks("grounded", top_k=1)

        assert len(chunks) == 1
        assert chunks[0].source == "vector:doc-9"
        assert chunks[0].text == "grounded fact"


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


def test_vector_store_package_auto_registers_available_vendor_backends(monkeypatch):
    import importlib
    from importlib.machinery import ModuleSpec

    import director_ai.core.retrieval.vector_store as vector_store_pkg
    import director_ai.core.retrieval.vector_store.base as base_mod

    original_registry = dict(base_mod._VECTOR_REGISTRY)
    original_find_spec = importlib.util.find_spec
    vendor_modules = {
        "chromadb",
        "pinecone",
        "weaviate",
        "qdrant_client",
        "faiss",
        "elasticsearch",
    }

    def fake_find_spec(name: str):
        if name in vendor_modules:
            return ModuleSpec(name, loader=None)
        return None

    monkeypatch.setattr("importlib.util.find_spec", fake_find_spec)

    try:
        base_mod._VECTOR_REGISTRY.clear()
        importlib.reload(vector_store_pkg)

        assert vector_store_pkg.get_vector_backend("chroma") is (
            vector_store_pkg.ChromaBackend
        )
        assert vector_store_pkg.get_vector_backend("pinecone") is (
            vector_store_pkg.PineconeBackend
        )
        assert vector_store_pkg.get_vector_backend("weaviate") is (
            vector_store_pkg.WeaviateBackend
        )
        assert vector_store_pkg.get_vector_backend("qdrant") is (
            vector_store_pkg.QdrantBackend
        )
        assert vector_store_pkg.get_vector_backend("faiss") is (
            vector_store_pkg.FAISSBackend
        )
        assert vector_store_pkg.get_vector_backend("elasticsearch") is (
            vector_store_pkg.ElasticsearchBackend
        )
    finally:
        vector_store_pkg.find_spec = original_find_spec
        base_mod._VECTOR_REGISTRY.clear()
        base_mod._VECTOR_REGISTRY.update(original_registry)


def test_vector_store_package_skips_unavailable_vendor_backends(monkeypatch):
    import importlib

    import director_ai.core.retrieval.vector_store as vector_store_pkg
    import director_ai.core.retrieval.vector_store.base as base_mod

    original_registry = dict(base_mod._VECTOR_REGISTRY)
    original_find_spec = importlib.util.find_spec

    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)

    try:
        base_mod._VECTOR_REGISTRY.clear()
        importlib.reload(vector_store_pkg)

        assert {"memory", "sentence-transformer", "hybrid", "remanentia"} <= set(
            vector_store_pkg.list_vector_backends()
        )
        assert "colbert" in vector_store_pkg.list_vector_backends()
        assert "chroma" not in vector_store_pkg.list_vector_backends()
        assert "pinecone" not in vector_store_pkg.list_vector_backends()
        assert "weaviate" not in vector_store_pkg.list_vector_backends()
        assert "qdrant" not in vector_store_pkg.list_vector_backends()
        assert "faiss" not in vector_store_pkg.list_vector_backends()
        assert "elasticsearch" not in vector_store_pkg.list_vector_backends()
    finally:
        vector_store_pkg.find_spec = original_find_spec
        base_mod._VECTOR_REGISTRY.clear()
        base_mod._VECTOR_REGISTRY.update(original_registry)


def test_sentence_transformer_delete_keeps_documents_and_embeddings_aligned():
    import threading

    import numpy as np

    from director_ai.core import vector_store

    backend = vector_store.SentenceTransformerBackend.__new__(
        vector_store.SentenceTransformerBackend,
    )
    backend._model = object()
    backend._docs = [
        {"id": "d1", "text": "alpha", "metadata": {}},
        {"id": "d2", "text": "beta", "metadata": {}},
    ]
    backend._embeddings = [
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
    ]
    backend._lock = threading.Lock()

    removed = backend.delete(["d1", "missing"])

    assert removed == 1
    assert [doc["id"] for doc in backend._docs] == ["d2"]
    assert len(backend._embeddings) == 1
    assert np.array_equal(backend._embeddings[0], np.asarray([0.0, 1.0]))


def test_sentence_transformer_backend_import_error_is_actionable():
    with (
        patch.dict("sys.modules", {"sentence_transformers": None}),
        pytest.raises(ImportError, match="requires sentence-transformers"),
    ):
        from director_ai.core import vector_store

        vector_store.SentenceTransformerBackend()


def test_sentence_transformer_backend_add_query_count_and_tenant_filter():
    import numpy as np

    from director_ai.core import vector_store

    class _SentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name

        def encode(self, text, normalize_embeddings=True):
            assert normalize_embeddings is True
            vectors = {
                "alpha text": [1.0, 0.0],
                "beta text": [0.0, 1.0],
                "alpha query": [1.0, 0.0],
                "negative query": [-1.0, 0.0],
            }
            return np.asarray(vectors[text], dtype=np.float32)

    fake_module = types.SimpleNamespace(SentenceTransformer=_SentenceTransformer)
    with patch.dict("sys.modules", {"sentence_transformers": fake_module}):
        backend = vector_store.SentenceTransformerBackend("local/model")

    assert backend.count() == 0
    assert backend.query("alpha query") == []

    backend.add("a", "alpha text", {"tenant_id": "tenant-a"})
    backend.add("b", "beta text", {"tenant_id": "tenant-b"})

    assert backend.count() == 2
    assert [row["id"] for row in backend.query("alpha query", n_results=2)] == ["a"]
    assert backend.query("alpha query", tenant_id="tenant-b") == []
    assert backend.query("alpha query", tenant_id="missing") == []
    assert backend.query("negative query") == []


@pytest.mark.parametrize("bad_doc_ids", ("d1", ["ok", ""], ["ok", 3]))
def test_sentence_transformer_delete_rejects_invalid_doc_ids(bad_doc_ids):
    from director_ai.core import vector_store

    backend = vector_store.SentenceTransformerBackend.__new__(
        vector_store.SentenceTransformerBackend,
    )

    with pytest.raises(ValueError, match="doc_ids"):
        backend.delete(bad_doc_ids)


def test_sentence_transformer_delete_empty_list_is_noop():
    from director_ai.core import vector_store

    backend = vector_store.SentenceTransformerBackend.__new__(
        vector_store.SentenceTransformerBackend,
    )

    assert backend.delete([]) == 0


def test_chroma_delete_delegates_ids_and_reports_removed_count():
    from unittest.mock import MagicMock

    from director_ai.core import vector_store

    collection = MagicMock()
    collection.count.side_effect = [3, 1]
    backend = vector_store.ChromaBackend.__new__(vector_store.ChromaBackend)
    backend._collection = collection

    removed = backend.delete(["d1", "d2"])

    assert removed == 2
    collection.delete.assert_called_once_with(ids=["d1", "d2"])


def test_chroma_backend_import_error_is_actionable():
    with (
        patch.dict("sys.modules", {"chromadb": None}),
        pytest.raises(ImportError, match="requires chromadb"),
    ):
        from director_ai.core import vector_store

        vector_store.ChromaBackend()


def test_chroma_backend_initialises_persistent_collection_with_embedding_function():
    from director_ai.core import vector_store

    collection = MagicMock()
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    chromadb = types.SimpleNamespace(
        PersistentClient=MagicMock(return_value=client),
        HttpClient=MagicMock(side_effect=AssertionError("HTTP Chroma is forbidden")),
    )

    class _EmbeddingFunction:
        def __init__(self, model_name):
            self.model_name = model_name

    embedding_functions = types.ModuleType("chromadb.utils.embedding_functions")
    embedding_functions.SentenceTransformerEmbeddingFunction = _EmbeddingFunction
    utils_module = types.ModuleType("chromadb.utils")
    utils_module.embedding_functions = embedding_functions

    with patch.dict(
        sys.modules,
        {
            "chromadb": chromadb,
            "chromadb.utils": utils_module,
            "chromadb.utils.embedding_functions": embedding_functions,
        },
    ):
        backend = vector_store.ChromaBackend(
            collection_name="facts",
            persist_directory="/tmp/director-ai-chroma",
            embedding_model="local/embedder",
        )

    chromadb.PersistentClient.assert_called_once_with(path="/tmp/director-ai-chroma")
    chromadb.HttpClient.assert_not_called()
    kwargs = client.get_or_create_collection.call_args.kwargs
    assert kwargs["name"] == "facts"
    assert kwargs["embedding_function"].model_name == "local/embedder"
    assert backend._collection is collection


def test_chroma_backend_uses_ephemeral_client_and_warns_without_embedding_extra(
    caplog,
):
    from director_ai.core import vector_store

    collection = MagicMock()
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    chromadb = types.SimpleNamespace(
        Client=MagicMock(return_value=client),
        HttpClient=MagicMock(side_effect=AssertionError("HTTP Chroma is forbidden")),
    )

    with patch.dict(
        sys.modules,
        {
            "chromadb": chromadb,
            "chromadb.utils.embedding_functions": None,
        },
    ):
        backend = vector_store.ChromaBackend(embedding_model="local/embedder")

    chromadb.Client.assert_called_once_with()
    chromadb.HttpClient.assert_not_called()
    client.get_or_create_collection.assert_called_once_with(name="director_ai_facts")
    assert "sentence-transformers not installed" in caplog.text
    assert backend._collection is collection


def test_chroma_backend_initialises_without_embedding_model():
    from director_ai.core import vector_store

    collection = MagicMock()
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    chromadb = types.SimpleNamespace(
        Client=MagicMock(return_value=client),
        HttpClient=MagicMock(side_effect=AssertionError("HTTP Chroma is forbidden")),
    )

    with patch.dict(sys.modules, {"chromadb": chromadb}):
        backend = vector_store.ChromaBackend(collection_name="facts")

    chromadb.Client.assert_called_once_with()
    chromadb.HttpClient.assert_not_called()
    client.get_or_create_collection.assert_called_once_with(name="facts")
    assert backend._collection is collection


def test_chroma_add_query_count_and_default_result_fields():
    from director_ai.core import vector_store

    collection = MagicMock()
    collection.count.return_value = 2
    collection.query.return_value = {
        "documents": [["alpha", "beta"]],
        "metadatas": [[{"tenant_id": "tenant-a"}, {"tenant_id": "tenant-b"}]],
    }
    backend = vector_store.ChromaBackend.__new__(vector_store.ChromaBackend)
    backend._collection = collection

    backend.add("a", "alpha", {"tenant_id": "tenant-a"})
    backend.add("b", "beta")
    results = backend.query("query", n_results=5, tenant_id="tenant-a")

    assert backend.count() == 2
    assert collection.add.call_args_list[0].kwargs == {
        "ids": ["a"],
        "documents": ["alpha"],
        "metadatas": [{"tenant_id": "tenant-a"}],
    }
    assert collection.add.call_args_list[1].kwargs == {
        "ids": ["b"],
        "documents": ["beta"],
        "metadatas": None,
    }
    collection.query.assert_called_once_with(
        query_texts=["query"],
        n_results=2,
        where={"tenant_id": "tenant-a"},
    )
    assert results == [
        {
            "id": "doc_0",
            "text": "alpha",
            "metadata": {"tenant_id": "tenant-a"},
            "distance": 0.0,
        },
        {
            "id": "doc_1",
            "text": "beta",
            "metadata": {"tenant_id": "tenant-b"},
            "distance": 0.0,
        },
    ]


def test_chroma_query_empty_collection_returns_empty_without_query():
    from director_ai.core import vector_store

    collection = MagicMock()
    collection.count.return_value = 0
    backend = vector_store.ChromaBackend.__new__(vector_store.ChromaBackend)
    backend._collection = collection

    assert backend.query("query") == []
    collection.query.assert_not_called()


@pytest.mark.parametrize("bad_doc_ids", ("d1", ["ok", ""], ["ok", 3]))
def test_chroma_delete_rejects_invalid_doc_ids(bad_doc_ids):
    from director_ai.core import vector_store

    backend = vector_store.ChromaBackend.__new__(vector_store.ChromaBackend)

    with pytest.raises(ValueError, match="doc_ids"):
        backend.delete(bad_doc_ids)


def test_chroma_delete_empty_list_is_noop():
    from director_ai.core import vector_store

    backend = vector_store.ChromaBackend.__new__(vector_store.ChromaBackend)

    assert backend.delete([]) == 0


def test_chroma_delete_never_reports_negative_removed_count():
    from director_ai.core import vector_store

    collection = MagicMock()
    collection.count.side_effect = [1, 3]
    backend = vector_store.ChromaBackend.__new__(vector_store.ChromaBackend)
    backend._collection = collection

    assert backend.delete(["d1"]) == 0
