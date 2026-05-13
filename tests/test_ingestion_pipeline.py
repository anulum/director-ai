# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Document Ingestion Pipeline Tests
"""Tests for the reusable document ingestion pipeline facade."""

from __future__ import annotations

from director_ai.core.ingestion import DocumentIngestionPipeline, IngestionConfig
from director_ai.core.retrieval.vector_store import VectorGroundTruthStore


class TestDocumentIngestionPipeline:
    def test_ingest_text_chunks_registers_and_retrieves_by_tenant(self):
        store = VectorGroundTruthStore()
        pipeline = DocumentIngestionPipeline(
            store=store,
            config=IngestionConfig(chunk_size=36, overlap=0),
        )

        result = pipeline.ingest_text(
            "Alpha support policy. Beta refund policy. Gamma invoice policy.",
            doc_id="policy",
            source="policy.md",
            tenant_id="tenant-a",
        )

        assert result.doc_id == "policy"
        assert result.source == "policy.md"
        assert result.tenant_id == "tenant-a"
        assert result.chunk_count >= 2
        assert result.chunk_ids == [
            f"policy:chunk:{i}" for i in range(result.chunk_count)
        ]
        assert pipeline.registry.get("policy", "tenant-a") is not None
        assert store.backend.query("refund", tenant_id="tenant-a")
        assert store.backend.query("refund", tenant_id="tenant-b") == []

    def test_update_text_returns_unchanged_without_reingesting_same_content(self):
        store = VectorGroundTruthStore()
        pipeline = DocumentIngestionPipeline(store=store)

        first = pipeline.ingest_text(
            "The refund window is 30 days.",
            doc_id="refunds",
            source="refunds.md",
            tenant_id="",
        )
        before_count = store.backend.count()

        second = pipeline.update_text(
            "The refund window is 30 days.",
            doc_id="refunds",
            source="refunds.md",
            tenant_id="",
        )

        assert second.unchanged
        assert second.chunk_ids == first.chunk_ids
        assert store.backend.count() == before_count

    def test_update_text_replaces_old_chunks_transactionally(self):
        store = VectorGroundTruthStore()
        pipeline = DocumentIngestionPipeline(store=store)

        pipeline.ingest_text(
            "The refund window is 30 days.",
            doc_id="refunds",
            source="refunds.md",
            tenant_id="",
        )

        result = pipeline.update_text(
            "The refund window is 45 days.",
            doc_id="refunds",
            source="refunds-v2.md",
            tenant_id="",
        )

        assert not result.unchanged
        assert result.source == "refunds-v2.md"
        assert all(":rev:" in chunk_id for chunk_id in result.chunk_ids)
        assert store.backend.query("45 days")
        assert store.backend.query("30") == []

    def test_delete_document_removes_registry_and_chunks(self):
        store = VectorGroundTruthStore()
        pipeline = DocumentIngestionPipeline(store=store)
        pipeline.ingest_text(
            "Enterprise support includes phone support.",
            doc_id="support",
            source="support.md",
            tenant_id="enterprise",
        )

        deleted = pipeline.delete("support", tenant_id="enterprise")

        assert deleted.doc_id == "support"
        assert deleted.chunks_removed == 1
        assert pipeline.registry.get("support", "enterprise") is None
        assert store.backend.query("phone", tenant_id="enterprise") == []

    def test_ingest_bytes_uses_parser_and_rejects_empty_parse(self):
        pipeline = DocumentIngestionPipeline(store=VectorGroundTruthStore())

        result = pipeline.ingest_bytes(
            b"title,body\nRefunds,Refund window is 30 days",
            filename="policy.csv",
            doc_id="csv-policy",
            source="policy.csv",
            tenant_id="tenant-a",
        )

        assert result.chunk_count == 1
        assert pipeline.registry.get("csv-policy", "tenant-a") is not None

        try:
            pipeline.ingest_bytes(
                b"",
                filename="empty.txt",
                doc_id="empty",
                source="empty.txt",
            )
        except ValueError as exc:
            assert "no text" in str(exc)
        else:
            raise AssertionError("empty parsed documents must be rejected")

    def test_rejects_duplicate_doc_id_and_invalid_config(self):
        pipeline = DocumentIngestionPipeline(store=VectorGroundTruthStore())
        pipeline.ingest_text("First document.", doc_id="doc", source="doc.txt")

        try:
            pipeline.ingest_text("Second document.", doc_id="doc", source="doc.txt")
        except ValueError as exc:
            assert "already exists" in str(exc)
        else:
            raise AssertionError("duplicate doc_id must fail")

        try:
            IngestionConfig(chunk_size=32, overlap=32)
        except ValueError as exc:
            assert "overlap" in str(exc)
        else:
            raise AssertionError("invalid chunk configuration must fail")
