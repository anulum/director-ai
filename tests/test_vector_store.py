# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vector Store Tests
"""Multi-angle tests for vector store pipeline."""

from unittest.mock import patch

import pytest

from director_ai.core.vector_store import (
    _VECTOR_REGISTRY,
    HybridBackend,
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

    @pytest.mark.parametrize(
        ("doc_id", "text", "metadata", "message"),
        [
            ("", "text", None, "doc_id"),
            ("   ", "text", None, "doc_id"),
            (123, "text", None, "doc_id"),
            ("doc1", 123, None, "text"),
            ("doc1", "text", "bad", "metadata"),
        ],
    )
    def test_add_rejects_invalid_document_fields(
        self,
        doc_id,
        text,
        metadata,
        message,
    ):
        backend = InMemoryBackend()
        with pytest.raises(ValueError, match=message):
            backend.add(doc_id, text, metadata)  # type: ignore[arg-type]

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

    @pytest.mark.parametrize("n_results", [-1, 1.5, True])
    def test_query_rejects_invalid_n_results(self, n_results):
        backend = InMemoryBackend()
        backend.add("doc1", "The sky is blue")
        with pytest.raises(ValueError, match="n_results"):
            backend.query("sky", n_results=n_results)


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

    def test_add_writes_backend_once(self):
        class CountingBackend(InMemoryBackend):
            def __init__(self):
                super().__init__()
                self.add_calls = 0

            def add(self, doc_id, text, metadata=None):
                self.add_calls += 1
                return super().add(doc_id, text, metadata=metadata)

        backend = CountingBackend()
        store = VectorGroundTruthStore(backend=backend)
        store.add_fact("gravity", "9.81 m/s^2")
        assert backend.add_calls == 1

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            ("", "value", "key"),
            ("   ", "value", "key"),
            ("key", "", "value"),
            ("key", "   ", "value"),
        ],
    )
    def test_add_fact_rejects_empty_fields(self, key, value, message):
        store = VectorGroundTruthStore()
        with pytest.raises(ValueError, match=message):
            store.add_fact(key, value)

    @pytest.mark.parametrize("query", ["", "   "])
    def test_retrieve_context_rejects_empty_query(self, query):
        store = VectorGroundTruthStore()
        store.add_fact("sky", "blue")
        with pytest.raises(ValueError, match="query"):
            store.retrieve_context(query)

    def test_retrieve_context_rejects_negative_top_k(self):
        store = VectorGroundTruthStore()
        store.add_fact("sky", "blue")
        with pytest.raises(ValueError, match="top_k"):
            store.retrieve_context("sky", top_k=-1)

    def test_ingest_rejects_blank_documents(self):
        store = VectorGroundTruthStore()
        with pytest.raises(ValueError, match="texts"):
            store.ingest(["valid", " "])

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

    def test_ingest_propagates_tenant_metadata_and_empty_batch_is_noop(self):
        class CapturingBackend:
            def __init__(self):
                self.added = []

            def add(self, doc_id, text, metadata=None):
                self.added.append((doc_id, text, dict(metadata or {})))

            def query(self, text, n_results=3, tenant_id=""):
                return []

            def count(self):
                return len(self.added)

        backend = CapturingBackend()
        store = VectorGroundTruthStore(backend=backend, tenant_id="tenant-a")
        revision_before = store.revision

        assert store.ingest([]) == 0
        assert store.revision == revision_before
        assert store.ingest(["tenant-specific SOP"]) == 1

        doc_id, text, metadata = backend.added[0]
        assert doc_id == "ingest_0_tenant-a"
        assert text == "tenant-specific SOP"
        assert metadata["tenant_id"] == "tenant-a"
        assert metadata["kb_record_kind"] == "derived_chunk"
        assert set(store.version_manifest("tenant-a")) == {
            "tenant-a::ingest_0_tenant-a"
        }

    def test_add_wraps_backend_failure_and_preserves_revision(self):
        class FailingBackend:
            def add(self, doc_id, text, metadata=None):
                raise RuntimeError("vector service unavailable")

            def query(self, text, n_results=3, tenant_id=""):
                return []

            def count(self):
                return 0

        store = VectorGroundTruthStore(backend=FailingBackend())
        revision_before = store.revision

        with pytest.raises(ValueError, match="Failed to add to vector store"):
            store.add("policy", "refunds in 30 days")

        assert store.revision == revision_before
        assert store.facts == {}
        assert store.version_manifest() == {}

    def test_add_fact_backend_failure_does_not_mutate_keyword_fallback(self):
        class FailingBackend:
            def add(self, doc_id, text, metadata=None):
                raise RuntimeError("vector service unavailable")

            def query(self, text, n_results=3, tenant_id=""):
                return []

            def count(self):
                return 0

        store = VectorGroundTruthStore(backend=FailingBackend())

        with pytest.raises(ValueError, match="Failed to add to vector store"):
            store.add_fact("policy", "refunds in 30 days")

        assert store.facts == {}
        assert store.version_manifest() == {}

    def test_readding_identical_fact_preserves_version_and_previous_hash(self):
        store = VectorGroundTruthStore()
        store.add("policy", "refunds in 30 days")
        first = store.fact_version_record("policy")

        store.add("policy", "refunds in 30 days")
        second = store.fact_version_record("policy")

        assert first is not None
        assert second is not None
        assert second["version"] == first["version"]
        assert second["previous_hash"] == first["previous_hash"]

    def test_invalid_version_bump_rejected(self):
        store = VectorGroundTruthStore()
        store.add("policy", "refunds in 30 days")

        with pytest.raises(ValueError, match="kb_version_bump"):
            store.add(
                "policy",
                "refunds in 45 days",
                metadata={"kb_version_bump": "calendar"},
            )

    def test_retract_fact_records_event_and_blocks_retrieval(self):
        store = VectorGroundTruthStore()
        store.add_fact("policy", "refunds in 30 days")

        event = store.retract_fact("policy", reason="source withdrawn")
        context = store.retrieve_context("policy")
        chunks = store.retrieve_context_with_chunks("policy")
        records = store.retraction_records()
        version = store.fact_version_record("policy")

        assert event["event"] == "retracted"
        assert records == [event]
        assert version is not None
        assert version["status"] == "retracted"
        assert context is None
        assert chunks == []

    def test_retraction_is_tenant_scoped(self):
        store = VectorGroundTruthStore()
        store.add_fact("policy", "tenant a value", tenant_id="tenant_a")
        store.add_fact("policy", "tenant b value", tenant_id="tenant_b")

        store.retract_fact("policy", tenant_id="tenant_a", reason="old source")

        ctx_a = store.retrieve_context("policy", tenant_id="tenant_a")
        ctx_b = store.retrieve_context("policy", tenant_id="tenant_b")
        records_b = store.retraction_records("tenant_b")

        assert ctx_a is None
        assert ctx_b is not None
        assert "tenant b value" in ctx_b
        assert records_b == []

    def test_replace_fact_records_superseded_hash(self):
        store = VectorGroundTruthStore()
        store.add_fact("policy", "refunds in 30 days")
        before = store.fact_version_record("policy")

        event = store.replace_fact(
            "policy",
            "refunds in 45 days",
            reason="policy update",
        )
        after = store.fact_version_record("policy")
        context = store.retrieve_context("policy")

        assert before is not None
        assert after is not None
        assert event["event"] == "replaced"
        assert event["from_hash"] == before["content_hash"]
        assert event["to_hash"] == after["content_hash"]
        assert event["to_version"] == "1.0.1"
        assert context is not None
        assert "45 days" in context
        assert "30 days" not in context

    def test_retract_derived_chunk_blocks_vector_result(self):
        store = VectorGroundTruthStore()
        store.ingest(["alpha chunk"])

        store.retract_fact("ingest_0_", reason="source withdrawn")
        context = store.retrieve_context("alpha")
        chunks = store.retrieve_context_with_chunks("alpha")

        assert context is None
        assert chunks == []

    def test_in_memory_backend_delete_removes_requested_documents(self):
        backend = InMemoryBackend()
        backend.add("doc-a", "alpha beta")
        backend.add("doc-b", "gamma delta")

        removed = backend.delete(["doc-a", "missing"])

        assert removed == 1
        assert backend.count() == 1
        assert backend.query("alpha") == []
        assert backend.query("gamma")[0]["id"] == "doc-b"

    def test_retract_unknown_fact_raises_key_error(self):
        store = VectorGroundTruthStore()

        with pytest.raises(KeyError, match="cannot retract unknown fact"):
            store.retract_fact("missing")

    def test_replace_unknown_fact_raises_key_error(self):
        store = VectorGroundTruthStore()

        with pytest.raises(KeyError, match="cannot replace unknown fact"):
            store.replace_fact("missing", "new value")

    def test_empty_kb_snapshot_root_is_deterministic(self):
        store = VectorGroundTruthStore()

        root_a = store.kb_snapshot_root()
        root_b = VectorGroundTruthStore().kb_snapshot_root()

        assert root_a == root_b
        assert len(root_a) == 64

    def test_kb_snapshot_root_is_independent_of_insert_order(self):
        left = VectorGroundTruthStore()
        right = VectorGroundTruthStore()

        left.add_fact("alpha", "first value")
        left.add_fact("beta", "second value")
        right.add_fact("beta", "second value")
        right.add_fact("alpha", "first value")

        assert left.kb_snapshot_records() == right.kb_snapshot_records()
        assert left.kb_snapshot_root() == right.kb_snapshot_root()

    def test_kb_snapshot_root_changes_for_retraction_and_replacement(self):
        store = VectorGroundTruthStore()
        store.add_fact("policy", "refunds in 30 days")
        initial = store.kb_snapshot_root()

        store.replace_fact("policy", "refunds in 45 days", reason="policy update")
        replaced = store.kb_snapshot_root()
        store.retract_fact("policy", reason="source withdrawn")
        retracted = store.kb_snapshot_root()

        assert replaced != initial
        assert retracted != replaced

    def test_kb_snapshot_root_is_tenant_scoped(self):
        store = VectorGroundTruthStore()
        store.add_fact("policy", "tenant a value", tenant_id="tenant_a")
        store.add_fact("policy", "tenant b value", tenant_id="tenant_b")

        root_a = store.kb_snapshot_root("tenant_a")
        root_b = store.kb_snapshot_root("tenant_b")
        records_a = store.kb_snapshot_records("tenant_a")

        assert root_a != root_b
        assert {record["tenant_id"] for record in records_a} == {"tenant_a"}

    def test_kb_snapshot_audit_record_reports_root_and_counts(self):
        store = VectorGroundTruthStore()
        store.add_fact("policy", "refunds in 30 days")
        store.replace_fact("policy", "refunds in 45 days", reason="policy update")

        record = store.kb_snapshot_audit_record()

        assert record["event"] == "kb_snapshot"
        assert record["record_count"] == 1
        assert record["replacement_count"] == 1
        assert record["retraction_count"] == 0
        assert record["conflict_count"] == 0
        assert record["revision"] == store.revision
        assert record["merkle_root"] == store.kb_snapshot_root()

    def test_freshness_status_signals_from_metadata(self):
        store = VectorGroundTruthStore()
        store.add_fact(
            "paper-a",
            "trial result changed",
            metadata={
                "external_id": "doi:10.example/paper-a",
                "source_timestamp": "1700000000",
                "updated_timestamp": "1710000000",
                "citation_status": "superseded",
                "status_source": "publisher-feed",
                "status_observed_at": "1720000000",
            },
            tenant_id="lab",
        )

        signals = store.freshness_status_signals("lab")

        assert signals == [
            {
                "source_id": "doi:10.example/paper-a",
                "status": "superseded",
                "status_source": "publisher-feed",
                "published_at": 1700000000.0,
                "updated_at": 1710000000.0,
                "observed_at": 1720000000.0,
            }
        ]

    def test_freshness_status_signal_can_filter_key(self):
        store = VectorGroundTruthStore()
        store.add_fact(
            "paper-a",
            "old result",
            metadata={"citation_status": "superseded"},
            tenant_id="lab",
        )
        store.add_fact("paper-b", "stable result", tenant_id="lab")

        signals = store.freshness_status_signals("lab", key="paper-a")

        assert len(signals) == 1
        assert signals[0]["source_id"] == "paper-a"

    def test_freshness_status_signals_ignore_records_without_temporal_metadata(self):
        store = VectorGroundTruthStore()
        store.add_fact("paper-a", "stable result without citation feed metadata")

        assert store.freshness_status_signals() == []

    def test_conflict_report_for_retracted_fact_key(self):
        store = VectorGroundTruthStore()
        store.add_fact("paper-a", "withdrawn result")
        store.retract_fact("paper-a", reason="withdrawn source")

        store.add_fact("paper-a", "new result")
        reports = store.conflict_reports()

        assert len(reports) == 1
        assert reports[0]["conflict_type"] == "retraction_record"
        assert reports[0]["key"] == "paper-a"
        assert reports[0]["existing_key"] == "paper-a"

    def test_conflict_report_for_signed_fact_claim(self):
        store = VectorGroundTruthStore()
        store.add_fact(
            "signed-dose",
            "Dose is 5 mg.",
            metadata={
                "claim_id": "dose-claim",
                "signed_fact_id": "signed-1",
                "claim_source": "signed_fact",
            },
        )

        store.add_fact(
            "incoming-dose",
            "Dose is 10 mg.",
            metadata={"claim_id": "dose-claim"},
        )
        report = store.conflict_reports(key="incoming-dose")[0]

        assert report["conflict_type"] == "signed_fact"
        assert report["signed_fact_id"] == "signed-1"
        assert report["claim_id"] == "dose-claim"

    def test_conflict_report_for_passport_claim(self):
        store = VectorGroundTruthStore()
        store.add_fact(
            "passport-scope",
            "Agent may read corpus A.",
            metadata={
                "claim_id": "scope-claim",
                "passport_claim_id": "passport-1",
                "claim_source": "passport_claim",
            },
        )

        store.add_fact(
            "incoming-scope",
            "Agent may write corpus A.",
            metadata={"claim_id": "scope-claim"},
        )
        report = store.conflict_reports(key="incoming-scope")[0]

        assert report["conflict_type"] == "passport_claim"
        assert report["passport_claim_id"] == "passport-1"
        assert report["claim_id"] == "scope-claim"

    def test_conflict_report_for_explicit_relation(self):
        store = VectorGroundTruthStore()
        store.add_fact("basis", "The permit is active.")

        store.add_fact(
            "incoming",
            "The permit is inactive.",
            metadata={"contradicts": "basis"},
        )
        report = store.conflict_reports(key="incoming")[0]

        assert report["conflict_type"] == "explicit_relation"
        assert report["existing_key"] == "basis"
        assert report["reason"] == "new fact declares a contradiction target"

    def test_conflict_reports_are_tenant_scoped(self):
        store = VectorGroundTruthStore()
        store.add_fact("paper", "old value", tenant_id="tenant_a")
        store.retract_fact("paper", tenant_id="tenant_a")

        store.add_fact("paper", "new value", tenant_id="tenant_a")
        store.add_fact("paper", "other value", tenant_id="tenant_b")

        assert len(store.conflict_reports("tenant_a")) == 1
        assert store.conflict_reports("tenant_b") == []
        assert store.conflict_reports("tenant_a", key="other-paper") == []

    def test_conflict_helpers_ignore_unrelated_refs_and_dedupe_duplicate_reports(self):
        store = VectorGroundTruthStore()
        store.add_fact("withdrawn-a", "withdrawn result")
        store.retract_fact("withdrawn-a", reason="source withdrawn")
        store.add_fact(
            "signed-dose",
            "Dose is 5 mg.",
            metadata={"claim_id": "dose", "signed_fact_id": "signed-1"},
        )

        store.add_fact("unrelated", "No overlapping protected references.")
        assert store.conflict_reports(key="unrelated") == []

        store.add_fact(
            "unrelated-explicit",
            "Declares contradiction against a source not in this KB.",
            metadata={"contradicts": "external-only"},
        )
        assert store.conflict_reports(key="unrelated-explicit") == []

        first = store._conflict_record(
            key="incoming",
            tenant_id="",
            conflict_type="signed_fact",
            existing={"key": "signed-dose", "claim_id": "dose"},
            incoming={"kb_content_hash": "incoming-hash", "claim_id": "dose"},
            reason="duplicate synthetic input",
        )
        duplicate = dict(first)
        unique = dict(first, reference="signed-1")

        assert store._dedupe_conflicts([first, duplicate, unique]) == [first, unique]

    def test_conflict_type_accepts_claim_source_without_explicit_ids(self):
        assert (
            VectorGroundTruthStore._protected_conflict_type(
                {"claim_source": "signed_fact"},
                {},
            )
            == "signed_fact"
        )
        assert (
            VectorGroundTruthStore._protected_conflict_type(
                {},
                {"claim_source": "passport_claim"},
            )
            == "passport_claim"
        )
        assert VectorGroundTruthStore._protected_conflict_type({}, {}) == ""

    def test_metadata_list_accepts_blank_scalar_and_collection_values(self):
        assert VectorGroundTruthStore._metadata_list(None) == set()
        assert VectorGroundTruthStore._metadata_list("") == set()
        assert VectorGroundTruthStore._metadata_list("a, b,,") == {"a", "b"}
        assert VectorGroundTruthStore._metadata_list(["a", " ", 7]) == {"a", "7"}
        assert VectorGroundTruthStore._metadata_list(42) == {"42"}

    def test_merkle_root_duplicates_odd_leaf_and_semver_major_validation(self):
        leaves = [b"a" * 32, b"b" * 32, b"c" * 32]
        root = VectorGroundTruthStore._merkle_root_hex(leaves)

        assert len(root) == 64
        assert VectorGroundTruthStore._next_semver("1.2.3", "major") == "2.0.0"
        with pytest.raises(ValueError, match="invalid semantic version"):
            VectorGroundTruthStore._next_semver("1.2", "patch")

    def test_legacy_backend_query_signature_is_supported_for_context_and_chunks(self):
        class LegacyBackend:
            def add(self, doc_id, text, metadata=None):
                pass

            def query(self, text, n_results=3):
                assert text == "legacy query"
                return [
                    {
                        "id": "legacy-doc",
                        "text": "Legacy backend context",
                        "distance": 0.25,
                        "metadata": {},
                    }
                ]

            def count(self):
                return 1

        store = VectorGroundTruthStore(backend=LegacyBackend(), tenant_id="tenant-a")

        context = store.retrieve_context("legacy query")
        chunks = store.retrieve_context_with_chunks("legacy query")

        assert context == "Legacy backend context"
        assert len(chunks) == 1
        assert chunks[0].text == "Legacy backend context"
        assert chunks[0].distance == pytest.approx(0.25)
        assert chunks[0].source == "vector:legacy-doc"

    def test_query_errors_are_wrapped_for_context_and_chunks(self):
        class BrokenQueryBackend:
            def add(self, doc_id, text, metadata=None):
                pass

            def query(self, text, n_results=3, tenant_id=""):
                raise RuntimeError("index offline")

            def count(self):
                return 0

        store = VectorGroundTruthStore(backend=BrokenQueryBackend())

        with pytest.raises(ValueError, match="Failed to query vector store"):
            store.retrieve_context("policy")
        with pytest.raises(ValueError, match="Failed to query vector store"):
            store.retrieve_context_with_chunks("policy")

    def test_grounded_factory_falls_back_when_dense_backend_is_unavailable(self):
        with patch(
            "director_ai.core.retrieval.vector_store.store.SentenceTransformerBackend",
            side_effect=RuntimeError("missing sentence-transformers"),
        ):
            dense_store = VectorGroundTruthStore.grounded(use_hybrid=False)
            hybrid_store = VectorGroundTruthStore.grounded(use_hybrid=True, rrf_k=12)

        assert isinstance(dense_store.backend, InMemoryBackend)
        assert isinstance(hybrid_store.backend, HybridBackend)
        assert hybrid_store.backend._rrf_k == 12

    def test_active_results_keep_backend_rows_without_metadata(self):
        store = VectorGroundTruthStore()
        rows = [{"id": "raw", "text": "raw backend row", "metadata": "not a dict"}]

        assert store._active_results(rows, tenant_id="tenant-a") == rows


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

    @pytest.mark.parametrize("name", ["", "   ", 123])
    def test_register_rejects_invalid_names(self, name):
        with pytest.raises(ValueError, match="backend name"):
            register_vector_backend(name, InMemoryBackend)  # type: ignore[arg-type]

    @pytest.mark.parametrize("name", ["", "   ", 123])
    def test_get_rejects_invalid_names(self, name):
        with pytest.raises(ValueError, match="backend name"):
            get_vector_backend(name)  # type: ignore[arg-type]


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
