# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Document Ingestion Pipeline Tests
"""Tests for the reusable document ingestion pipeline facade."""

from __future__ import annotations

import pytest

from director_ai.core.ingestion import DocumentIngestionPipeline, IngestionConfig
from director_ai.core.ingestion import pipeline as pipeline_module
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

    def test_update_and_delete_reject_missing_document(self):
        pipeline = DocumentIngestionPipeline(store=VectorGroundTruthStore())

        with pytest.raises(KeyError, match="missing"):
            pipeline.update_text("Replacement text.", doc_id="missing")

        with pytest.raises(KeyError, match="missing"):
            pipeline.delete("missing")

    def test_update_rolls_back_staged_chunks_when_old_chunk_delete_fails(self):
        store = _StoreWithBackend(_FailingDeleteBackend())
        pipeline = DocumentIngestionPipeline(store=store)
        original = pipeline.ingest_text(
            "Original retention policy.",
            doc_id="policy",
            source="policy.md",
        )

        with pytest.raises(RuntimeError, match="delete failed"):
            pipeline.update_text(
                "Replacement retention policy.",
                doc_id="policy",
                source="policy-v2.md",
            )

        assert pipeline.registry.get("policy", "default").chunk_ids == original.chunk_ids
        assert sorted(store.facts) == original.chunk_ids
        assert any(":rev:" in chunk_id for chunk_id in store.backend.deleted)

    def test_stage_chunks_cleans_previous_adds_when_backend_add_fails(self):
        store = _StoreWithBackend(_FailingSecondAddBackend())
        pipeline = DocumentIngestionPipeline(
            store=store,
            config=IngestionConfig(chunk_size=12, overlap=0),
        )

        with pytest.raises(RuntimeError, match="add failed"):
            pipeline.ingest_text(
                "Alpha text. Beta text. Gamma text.",
                doc_id="policy",
                source="policy.md",
            )

        assert store.facts == {}
        assert store.backend.deleted == ["policy:chunk:0"]

    def test_delete_accepts_backend_without_integer_delete_count(self):
        store = _StoreWithBackend(_NonIntegerDeleteBackend())
        pipeline = DocumentIngestionPipeline(store=store)
        pipeline.ingest_text(
            "Enterprise support policy.",
            doc_id="support",
            source="support.md",
        )

        deleted = pipeline.delete("support")

        assert deleted.chunks_removed == 1
        assert store.facts == {}

    def test_delete_rejects_backend_count_mismatch(self):
        store = _StoreWithBackend(_WrongCountDeleteBackend())
        pipeline = DocumentIngestionPipeline(store=store)
        pipeline.ingest_text(
            "Enterprise support policy.",
            doc_id="support",
            source="support.md",
        )

        with pytest.raises(RuntimeError, match="reported 0 deletions"):
            pipeline.delete("support")

    def test_rejects_empty_text_no_chunks_and_invalid_identifiers(self, monkeypatch):
        pipeline = DocumentIngestionPipeline(store=VectorGroundTruthStore())

        with pytest.raises(ValueError, match="text must be a non-empty string"):
            pipeline.ingest_text(" ", doc_id="empty")

        monkeypatch.setattr(pipeline_module, "split", lambda text, config: [])
        with pytest.raises(ValueError, match="produced no chunks"):
            pipeline.ingest_text("Valid text.", doc_id="no-chunks")

        with pytest.raises(ValueError, match="path separators"):
            pipeline.ingest_text("Valid text.", doc_id="../escape")

        with pytest.raises(ValueError, match="tenant_id must be a string"):
            pipeline.ingest_text("Valid text.", doc_id="tenant", tenant_id=42)

        with pytest.raises(ValueError, match="source must be a non-empty string"):
            pipeline.ingest_text("Valid text.", doc_id="source", source=" ")

        with pytest.raises(ValueError, match="source contains control characters"):
            pipeline.ingest_text("Valid text.", doc_id="control", source="bad\nsource")


class _StoreWithBackend:
    def __init__(self, backend) -> None:
        self.backend = backend
        self.facts: dict[str, str] = {}


class _RecordingBackend:
    def __init__(self) -> None:
        self.docs: dict[str, tuple[str, dict]] = {}
        self.deleted: list[str] = []

    def add(self, *, doc_id: str, text: str, metadata: dict) -> None:
        self.docs[doc_id] = (text, metadata)

    def delete(self, doc_ids: list[str]):
        self.deleted.extend(doc_ids)
        for doc_id in doc_ids:
            self.docs.pop(doc_id, None)
        return len(doc_ids)


class _FailingDeleteBackend(_RecordingBackend):
    def delete(self, doc_ids: list[str]):
        self.deleted.extend(doc_ids)
        if any(":rev:" not in doc_id for doc_id in doc_ids):
            raise RuntimeError("delete failed")
        for doc_id in doc_ids:
            self.docs.pop(doc_id, None)
        return len(doc_ids)


class _FailingSecondAddBackend(_RecordingBackend):
    def add(self, *, doc_id: str, text: str, metadata: dict) -> None:
        if self.docs:
            raise RuntimeError("add failed")
        super().add(doc_id=doc_id, text=text, metadata=metadata)


class _NonIntegerDeleteBackend(_RecordingBackend):
    def delete(self, doc_ids: list[str]):
        super().delete(doc_ids)
        return None


class _WrongCountDeleteBackend(_RecordingBackend):
    def delete(self, doc_ids: list[str]):
        self.deleted.extend(doc_ids)
        return 0
