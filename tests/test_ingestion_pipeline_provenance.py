# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ingestion pipeline + provenance ledger wiring tests

"""Tests for the contract between DocumentIngestionPipeline and the
KnowledgeProvenanceLedger: ingest/update/delete each append the matching
signed event, the unchanged-update path appends nothing, chunk content
digests bind to verifiable inclusion proofs, and a pipeline without a
ledger keeps its original behaviour."""

from __future__ import annotations

import hashlib

import pytest

from director_ai.core.ingestion import DocumentIngestionPipeline, IngestionConfig
from director_ai.core.provenance import (
    KnowledgeProvenanceLedger,
    KnowledgeSupersessionPolicy,
)
from director_ai.core.retrieval.vector_store import VectorGroundTruthStore

_SECRET = b"director-ai-pipeline-provenance-secret-key"


def _pipeline_with_ledger(tmp_path):
    store = VectorGroundTruthStore()
    ledger = KnowledgeProvenanceLedger(secret=_SECRET, path=tmp_path / "kb.jsonl")
    pipeline = DocumentIngestionPipeline(
        store=store,
        config=IngestionConfig(chunk_size=36, overlap=0),
        ledger=ledger,
    )
    return pipeline, ledger


class TestIngestWiring:
    def test_ingest_appends_event_with_verifiable_proofs(self, tmp_path):
        pipeline, ledger = _pipeline_with_ledger(tmp_path)
        result = pipeline.ingest_text(
            "Alpha support policy. Beta refund policy. Gamma invoice policy.",
            doc_id="policy",
            source="policy.md",
            tenant_id="tenant-a",
        )
        assert len(ledger) == 1
        history = ledger.history_for("policy")
        assert [event.event_type for event in history] == ["ingest"]
        for chunk_id in result.chunk_ids:
            prov = ledger.provenance_of(chunk_id)
            assert prov is not None
            assert prov.doc_id == "policy"
            assert prov.source == "policy.md"
            assert prov.verified
        assert ledger.verify() == (True, None)

    def test_leaf_binds_chunk_text(self, tmp_path):
        pipeline, ledger = _pipeline_with_ledger(tmp_path)
        pipeline.ingest_text(
            "Single short chunk.",
            doc_id="d",
            source="s",
            tenant_id="",
        )
        chunk_id = "d:chunk:0"
        stored_text = pipeline.store.facts[chunk_id]
        prov = ledger.provenance_of(chunk_id)
        assert prov is not None
        assert prov.proof.leaf == hashlib.sha256(stored_text.encode()).digest()


class TestUpdateWiring:
    def test_changed_update_appends_event_and_retires_old(self, tmp_path):
        pipeline, ledger = _pipeline_with_ledger(tmp_path)
        first = pipeline.ingest_text(
            "The refund window is 30 days.",
            doc_id="refunds",
            source="refunds.md",
            tenant_id="",
        )
        pipeline.update_text(
            "The refund window is now 14 days for all customers.",
            doc_id="refunds",
            source="refunds.md",
            tenant_id="",
        )
        assert [event.event_type for event in ledger.history_for("refunds")] == [
            "ingest",
            "update",
        ]
        for old_chunk_id in first.chunk_ids:
            assert ledger.provenance_of(old_chunk_id) is None
        # An empty tenant id normalises to "default" inside the pipeline.
        updated_record = pipeline.registry.get("refunds", "default")
        for chunk_id in updated_record.chunk_ids:
            assert ledger.provenance_of(chunk_id).verified
        assert ledger.verify() == (True, None)

    def test_unchanged_update_appends_no_event(self, tmp_path):
        pipeline, ledger = _pipeline_with_ledger(tmp_path)
        pipeline.ingest_text(
            "The refund window is 30 days.",
            doc_id="refunds",
            source="refunds.md",
            tenant_id="",
        )
        pipeline.update_text(
            "The refund window is 30 days.",
            doc_id="refunds",
            source="refunds.md",
            tenant_id="",
        )
        assert len(ledger) == 1


class TestDeleteWiring:
    def test_delete_appends_event_and_retires_chunks(self, tmp_path):
        pipeline, ledger = _pipeline_with_ledger(tmp_path)
        result = pipeline.ingest_text(
            "Alpha support policy. Beta refund policy.",
            doc_id="policy",
            source="policy.md",
            tenant_id="",
        )
        pipeline.delete("policy", tenant_id="")
        assert [event.event_type for event in ledger.history_for("policy")] == [
            "ingest",
            "delete",
        ]
        for chunk_id in result.chunk_ids:
            assert ledger.provenance_of(chunk_id) is None
        assert ledger.verify() == (True, None)


class TestSupersessionWiring:
    def _ingest_two_versions(self, tmp_path):
        pipeline, ledger = _pipeline_with_ledger(tmp_path)
        pipeline.ingest_text(
            "The refund window is 30 days.",
            doc_id="refunds_v1",
            source="refunds.md",
            tenant_id="acme",
        )
        new = pipeline.ingest_text(
            "The refund window is 14 days for all customers.",
            doc_id="refunds_v2",
            source="refunds.md",
            tenant_id="acme",
        )
        return pipeline, ledger, new

    def test_approved_supersession_retires_old_and_records_event(self, tmp_path):
        pipeline, ledger, new = self._ingest_two_versions(tmp_path)
        old_record = pipeline.registry.get("refunds_v1", "acme")
        decision = KnowledgeSupersessionPolicy().evaluate(
            incoming_doc_id="refunds_v2",
            incoming_source="refunds.md",
            tenant_id="acme",
            existing=pipeline.registry.list_for_tenant("acme"),
        )
        assert decision.action == "recommend"
        result = pipeline.apply_supersession(decision, approved=True)
        assert result.superseded_doc_ids == ("refunds_v1",)
        assert result.superseded_count == 1
        assert result.chunks_removed == len(old_record.chunk_ids)
        assert pipeline.registry.get("refunds_v1", "acme") is None
        assert ledger.provenance_of(old_record.chunk_ids[0]) is None
        for chunk_id in new.chunk_ids:
            assert ledger.provenance_of(chunk_id).verified
        history = [event.event_type for event in ledger.history_for("refunds_v2")]
        assert "supersede" in history
        assert ledger.verify() == (True, None)

    def test_unapproved_supersession_refused(self, tmp_path):
        pipeline, _ledger, _new = self._ingest_two_versions(tmp_path)
        decision = KnowledgeSupersessionPolicy().evaluate(
            incoming_doc_id="refunds_v2",
            incoming_source="refunds.md",
            tenant_id="acme",
            existing=pipeline.registry.list_for_tenant("acme"),
        )
        with pytest.raises(PermissionError, match="human approval"):
            pipeline.apply_supersession(decision)
        # The refused decision must not have mutated the store.
        assert pipeline.registry.get("refunds_v1", "acme") is not None

    def test_auto_promoted_supersession_applies_without_approval(self, tmp_path):
        pipeline, ledger, _new = self._ingest_two_versions(tmp_path)
        decision = KnowledgeSupersessionPolicy(
            auto_promote=True, auto_promote_threshold=0.95
        ).evaluate(
            incoming_doc_id="refunds_v2",
            incoming_source="refunds.md",
            tenant_id="acme",
            existing=pipeline.registry.list_for_tenant("acme"),
            explicit_supersedes=["refunds_v1"],
        )
        assert decision.action == "promote"
        result = pipeline.apply_supersession(decision)
        assert result.superseded_doc_ids == ("refunds_v1",)
        assert ledger.verify() == (True, None)

    def test_already_removed_target_is_skipped(self, tmp_path):
        pipeline, _ledger, _new = self._ingest_two_versions(tmp_path)
        # Build the decision while refunds_v1 is still a candidate, then
        # remove it before applying — apply must skip the vanished target.
        decision = KnowledgeSupersessionPolicy(
            auto_promote=True, auto_promote_threshold=0.95
        ).evaluate(
            incoming_doc_id="refunds_v2",
            incoming_source="refunds.md",
            tenant_id="acme",
            existing=pipeline.registry.list_for_tenant("acme"),
            explicit_supersedes=["refunds_v1"],
        )
        assert decision.superseded_doc_ids == ("refunds_v1",)
        pipeline.delete("refunds_v1", tenant_id="acme")
        result = pipeline.apply_supersession(decision)
        assert result.superseded_doc_ids == ()
        assert result.chunks_removed == 0

    def test_supersession_without_ledger_still_retires(self):
        store = VectorGroundTruthStore()
        pipeline = DocumentIngestionPipeline(store=store)
        pipeline.ingest_text("old", doc_id="d_v1", source="d.md", tenant_id="acme")
        pipeline.ingest_text(
            "new text here", doc_id="d_v2", source="d.md", tenant_id="acme"
        )
        decision = KnowledgeSupersessionPolicy(
            auto_promote=True, auto_promote_threshold=0.95
        ).evaluate(
            incoming_doc_id="d_v2",
            incoming_source="d.md",
            tenant_id="acme",
            existing=pipeline.registry.list_for_tenant("acme"),
            explicit_supersedes=["d_v1"],
        )
        result = pipeline.apply_supersession(decision)
        assert result.superseded_doc_ids == ("d_v1",)
        assert pipeline.registry.get("d_v1", "acme") is None

    def test_none_decision_is_noop(self, tmp_path):
        pipeline, ledger, _new = self._ingest_two_versions(tmp_path)
        decision = KnowledgeSupersessionPolicy().evaluate(
            incoming_doc_id="refunds_v2",
            incoming_source="unrelated.md",
            tenant_id="acme",
            existing=[],
        )
        result = pipeline.apply_supersession(decision)
        assert result.superseded_doc_ids == ()
        assert result.chunks_removed == 0
        assert pipeline.registry.get("refunds_v1", "acme") is not None


class TestNoLedgerBackwardCompatible:
    def test_ingest_update_delete_without_ledger(self):
        store = VectorGroundTruthStore()
        pipeline = DocumentIngestionPipeline(store=store)
        assert pipeline.ledger is None
        pipeline.ingest_text(
            "The refund window is 30 days.",
            doc_id="refunds",
            source="refunds.md",
            tenant_id="",
        )
        pipeline.update_text(
            "The refund window is 14 days.",
            doc_id="refunds",
            source="refunds.md",
            tenant_id="",
        )
        deleted = pipeline.delete("refunds", tenant_id="")
        assert deleted.doc_id == "refunds"
