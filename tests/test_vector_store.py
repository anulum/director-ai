# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vector Store Tests

import pytest
from director_ai.core.vector_store import (
    _VECTOR_REGISTRY,
    InMemoryBackend,
    VectorBackend,
    VectorGroundTruthStore,
    get_vector_backend,
    list_vector_backends,
    register_vector_backend,
)


@pytest.mark.consumer
class TestInMemoryBackend:
    def test_add_and_count(self):
        backend = InMemoryBackend()
        assert backend.count() == 0
        backend.add("doc1", "The sky is blue")
        assert backend.count() == 1

    def test_query_returns_relevant(self):
        backend = InMemoryBackend()
        backend.add("doc1", "The sky is blue")
        backend.add("doc2", "Water is wet")
        backend.add("doc3", "Fire is hot")
        results = backend.query("What color is the sky?", n_results=2)
        assert len(results) > 0
        assert any("sky" in r["text"].lower() for r in results)

    def test_query_empty_store(self):
        backend = InMemoryBackend()
        results = backend.query("anything")
        assert results == []


@pytest.mark.consumer
class TestVectorGroundTruthStore:
    def test_default_store_is_empty(self):
        store = VectorGroundTruthStore()
        assert store.backend.count() == 0
        assert store.facts == {}

    def test_ingest_and_retrieve(self):
        store = VectorGroundTruthStore()
        store.ingest(["The sky is blue", "SCPN has 16 layers"])
        context = store.retrieve_context("How many layers in SCPN?")
        assert context is not None
        assert "16" in context

    def test_retrieve_context_sky_color(self):
        store = VectorGroundTruthStore()
        store.ingest(["sky color is blue"])
        context = store.retrieve_context("What color is the sky?")
        assert context is not None
        assert "blue" in context.lower()

    def test_add_custom_fact(self):
        store = VectorGroundTruthStore()
        store.add_fact("gravity", "9.81 m/s²")
        assert store.backend.count() == 1
        assert "gravity" in store.facts

    def test_retrieve_custom_fact(self):
        store = VectorGroundTruthStore()
        store.add_fact("planck constant", "6.626e-34 J·s")
        context = store.retrieve_context("What is the planck constant?")
        assert context is not None

    def test_keyword_fallback(self):
        """If vector search fails, keyword matching should still work."""
        store = VectorGroundTruthStore(backend=InMemoryBackend())
        store.add("sky color", "blue")
        context = store.retrieve_context("sky color")
        assert context is not None

    def test_tenant_id_stored(self):
        store = VectorGroundTruthStore(tenant_id="acme")
        assert store.tenant_id == "acme"

    def test_tenant_id_default_empty(self):
        store = VectorGroundTruthStore()
        assert store.tenant_id == ""


@pytest.mark.consumer
class TestVectorRegistry:
    def test_register_and_get(self):
        class _TestBackend(VectorBackend):
            def add(self, doc_id, text, metadata=None):
                pass

            def query(self, text, n_results=3):
                return []

            def count(self):
                return 0

        register_vector_backend("_test_dummy", _TestBackend)
        assert get_vector_backend("_test_dummy") is _TestBackend
        _VECTOR_REGISTRY.pop("_test_dummy", None)

    def test_list_includes_memory(self):
        backends = list_vector_backends()
        assert "memory" in backends
        assert backends["memory"] is InMemoryBackend

    def test_get_unknown_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown vector backend"):
            get_vector_backend("__nonexistent__")

    def test_register_non_subclass_raises_type_error(self):
        with pytest.raises(TypeError, match="VectorBackend subclass"):
            register_vector_backend("bad", str)  # type: ignore[arg-type]


@pytest.mark.enterprise
class TestTenantVectorIsolation:
    def test_two_tenants_no_data_leak(self):
        from director_ai.core.tenant import TenantRouter

        router = TenantRouter()
        store_a = router.get_vector_store("tenant_a")
        store_b = router.get_vector_store("tenant_b")
        store_a.add_fact("secret", "Tenant A secret data")
        store_b.add_fact("secret", "Tenant B secret data")

        ctx_a = store_a.retrieve_context("secret")
        ctx_b = store_b.retrieve_context("secret")
        assert "Tenant A" in ctx_a
        assert "Tenant B" in ctx_b
        assert "Tenant B" not in ctx_a
        assert "Tenant A" not in ctx_b

    def test_tenant_id_propagated(self):
        from director_ai.core.tenant import TenantRouter

        router = TenantRouter()
        store = router.get_vector_store("t1")
        assert store.tenant_id == "t1"

    def test_unknown_backend_type_raises(self):
        from director_ai.core.tenant import TenantRouter

        router = TenantRouter()
        with pytest.raises(ValueError, match="Unknown vector backend_type"):
            router.get_vector_store("t1", backend_type="invalid")

    def test_vector_store_cache_hit(self):
        from director_ai.core.tenant import TenantRouter

        router = TenantRouter()
        store_1 = router.get_vector_store("t1")
        store_2 = router.get_vector_store("t1")
        assert store_1 is store_2

    def test_chroma_backend_dispatch(self):
        from unittest.mock import MagicMock, patch

        from director_ai.core.tenant import TenantRouter

        mock_chroma = MagicMock()
        with patch("director_ai.core.vector_store.ChromaBackend", mock_chroma):
            router = TenantRouter()
            router.get_vector_store("t1", backend_type="chroma")
            mock_chroma.assert_called_once()
            call_kwargs = mock_chroma.call_args[1]
            assert call_kwargs["collection_name"] == "director_ai_t1"

    def test_pinecone_backend_dispatch(self):
        from unittest.mock import MagicMock, patch

        from director_ai.core.tenant import TenantRouter

        mock_pinecone = MagicMock()
        with patch(
            "director_ai.core.vector_store.PineconeBackend",
            mock_pinecone,
        ):
            router = TenantRouter()
            router.get_vector_store("t1", backend_type="pinecone")
            mock_pinecone.assert_called_once()
            assert mock_pinecone.call_args[1]["namespace"] == "t1"

    def test_qdrant_backend_dispatch(self):
        from unittest.mock import MagicMock, patch

        from director_ai.core.tenant import TenantRouter

        mock_qdrant = MagicMock()
        with patch(
            "director_ai.core.vector_store.QdrantBackend",
            mock_qdrant,
        ):
            router = TenantRouter()
            router.get_vector_store("t1", backend_type="qdrant")
            mock_qdrant.assert_called_once()
            call_kwargs = mock_qdrant.call_args[1]
            assert call_kwargs["collection_name"] == "director_facts_t1"
