# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vector Store Tests
"""Multi-angle tests for vector store pipeline."""

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
        store.add_fact("gravity", "9.81 m/sÂ˛")
        assert store.backend.count() == 1
        assert "gravity" in store.facts

    def test_retrieve_custom_fact(self):
        store = VectorGroundTruthStore()
        store.add_fact("planck constant", "6.626e-34 JÂ·s")
        context = store.retrieve_context("What is the planck constant?")
        assert context is not None

    def test_keyword_fallback(self):
        """If vector search fails, keyword matching should still work."""
        store = VectorGroundTruthStore(backend=InMemoryBackend())
        store.add("sky color", "blue")
        context = store.retrieve_context("sky color")
        assert context is not None

    def test_keyword_fallback_uses_store_tenant(self):
        class EmptyBackend(InMemoryBackend):
            def query(self, text, n_results=3, tenant_id=""):
                return []

        store = VectorGroundTruthStore(backend=EmptyBackend(), tenant_id="acme")
        store.add_fact("secret", "Tenant-scoped fallback fact")

        context = store.retrieve_context("secret")
        chunks = store.retrieve_context_with_chunks("secret")

        assert context is not None
        assert "Tenant-scoped fallback fact" in context
        assert len(chunks) == 1
        assert chunks[0].source == "keyword"
        assert "Tenant-scoped fallback fact" in chunks[0].text

    def test_keyword_fallback_separates_explicit_tenants(self):
        class EmptyBackend(InMemoryBackend):
            def query(self, text, n_results=3, tenant_id=""):
                return []

        store = VectorGroundTruthStore(backend=EmptyBackend())
        store.add_fact("secret", "Tenant A fallback fact", tenant_id="tenant_a")
        store.add_fact("secret", "Tenant B fallback fact", tenant_id="tenant_b")

        ctx_a = store.retrieve_context("secret", tenant_id="tenant_a")
        ctx_b = store.retrieve_context("secret", tenant_id="tenant_b")
        chunks_a = store.retrieve_context_with_chunks("secret", tenant_id="tenant_a")

        assert ctx_a is not None
        assert ctx_b is not None
        assert "Tenant A" in ctx_a
        assert "Tenant B" not in ctx_a
        assert "Tenant B" in ctx_b
        assert "Tenant A" not in ctx_b
        assert len(chunks_a) == 1
        assert "Tenant A" in chunks_a[0].text

    def test_tenant_id_stored(self):
        store = VectorGroundTruthStore(tenant_id="acme")
        assert store.tenant_id == "acme"

    def test_tenant_id_default_empty(self):
        store = VectorGroundTruthStore()
        assert store.tenant_id == ""

    def test_fact_versions_start_at_semantic_one(self):
        store = VectorGroundTruthStore()
        store.add_fact("gravity", "9.81 m/s^2")

        record = store.fact_version_record("gravity")

        assert store.fact_version("gravity") == "1.0.0"
        assert record is not None
        assert record["version"] == "1.0.0"
        assert record["record_kind"] == "fact"
        assert record["previous_hash"] == ""

    def test_fact_replacement_bumps_patch_version(self):
        store = VectorGroundTruthStore()
        store.add_fact("policy", "refunds in 30 days")
        first = store.fact_version_record("policy")
        store.add_fact("policy", "refunds in 45 days")
        second = store.fact_version_record("policy")

        assert first is not None
        assert second is not None
        assert second["version"] == "1.0.1"
        assert second["previous_hash"] == first["content_hash"]

    def test_fact_replacement_can_bump_minor_version(self):
        store = VectorGroundTruthStore()
        store.add("policy", "refunds in 30 days")
        store.add(
            "policy",
            "refunds in 45 days",
            metadata={"kb_version_bump": "minor"},
        )

        assert store.fact_version("policy") == "1.1.0"

    def test_versions_are_tenant_scoped(self):
        store = VectorGroundTruthStore()
        store.add_fact("policy", "tenant a value", tenant_id="tenant_a")
        store.add_fact("policy", "tenant b value", tenant_id="tenant_b")
        store.add_fact("policy", "tenant a replacement", tenant_id="tenant_a")

        manifest_a = store.version_manifest("tenant_a")
        manifest_b = store.version_manifest("tenant_b")

        assert store.fact_version("policy", tenant_id="tenant_a") == "1.0.1"
        assert store.fact_version("policy", tenant_id="tenant_b") == "1.0.0"
        assert set(manifest_a) == {"tenant_a::policy"}
        assert set(manifest_b) == {"tenant_b::policy"}

    def test_ingest_stamps_derived_chunk_versions(self):
        store = VectorGroundTruthStore()
        store.ingest(["alpha chunk", "beta chunk"])

        manifest = store.version_manifest()
        results = store.backend.query("alpha", n_results=1)

        assert manifest["ingest_0_"]["record_kind"] == "derived_chunk"
        assert manifest["ingest_0_"]["version"] == "1.0.0"
        assert results[0]["metadata"]["kb_chunk_version"] == "1.0.0"
        assert results[0]["metadata"]["kb_record_kind"] == "derived_chunk"

    def test_invalid_version_bump_rejected(self):
        store = VectorGroundTruthStore()
        store.add("policy", "refunds in 30 days")

        with pytest.raises(ValueError, match="kb_version_bump"):
            store.add(
                "policy",
                "refunds in 45 days",
                metadata={"kb_version_bump": "calendar"},
            )


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
