# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — KnowledgeProvenanceLedger tests

"""Tests for the operational knowledge-provenance ledger.

Covers event recording and HMAC chaining, content-commitment inclusion
proofs, the chunk admit/retire lifecycle through update and delete,
document and tenant history queries, JSONL persistence with
reload-and-verify, and tamper rejection for edited fields, a wrong secret,
reordered events, and a forged in-memory tag."""

from __future__ import annotations

import hashlib

import pytest

from director_ai.core.provenance import (
    KnowledgeProvenanceLedger,
    LedgerEvent,
    LedgerTamperError,
)

_SECRET = b"director-ai-provenance-ledger-secret-key"


def _leaf(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


def _clock(*ticks: float):
    iterator = iter(ticks)
    return lambda: next(iterator)


@pytest.fixture
def ledger_path(tmp_path):
    return tmp_path / "ledger.jsonl"


def _ingest_doc1(ledger: KnowledgeProvenanceLedger) -> LedgerEvent:
    return ledger.record_ingest(
        doc_id="doc1",
        tenant_id="t1",
        source="manual.pdf",
        content_hash="h1",
        chunk_leaves=[
            ("doc1:c0", _leaf("alpha")),
            ("doc1:c1", _leaf("beta")),
            ("doc1:c2", _leaf("gamma")),
        ],
    )


# --- recording + integrity -----------------------------------------


class TestRecording:
    def test_ingest_appends_verifiable_event(self, ledger_path):
        ledger = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        event = _ingest_doc1(ledger)
        assert event.index == 0
        assert event.event_type == "ingest"
        assert len(ledger) == 1
        assert ledger.verify() == (True, None)

    def test_chain_links_successive_events(self, ledger_path):
        ledger = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        first = _ingest_doc1(ledger)
        second = ledger.record_ingest(
            doc_id="doc2",
            tenant_id="t1",
            source="policy.txt",
            content_hash="h2",
            chunk_leaves=[("doc2:c0", _leaf("delta"))],
        )
        assert second.parent_hash != first.parent_hash
        assert ledger.verify() == (True, None)

    def test_short_secret_rejected(self, ledger_path):
        with pytest.raises(ValueError, match="at least 32 bytes"):
            KnowledgeProvenanceLedger(secret=b"tooshort", path=ledger_path)

    def test_empty_chunk_set_rejected(self, ledger_path):
        ledger = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        with pytest.raises(ValueError, match="at least one chunk"):
            ledger.record_ingest(
                doc_id="d",
                tenant_id="t",
                source="s",
                content_hash="h",
                chunk_leaves=[],
            )

    def test_non_bytes_digest_rejected(self, ledger_path):
        ledger = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        with pytest.raises(ValueError, match="non-empty bytes"):
            ledger.record_ingest(
                doc_id="d",
                tenant_id="t",
                source="s",
                content_hash="h",
                chunk_leaves=[("d:c0", "not-bytes")],  # type: ignore[list-item]
            )

    def test_empty_chunk_id_rejected(self, ledger_path):
        ledger = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        with pytest.raises(ValueError, match="chunk_id must be non-empty"):
            ledger.record_ingest(
                doc_id="d",
                tenant_id="t",
                source="s",
                content_hash="h",
                chunk_leaves=[("", _leaf("a"))],
            )

    def test_negative_event_index_rejected(self):
        with pytest.raises(ValueError, match="index must be non-negative"):
            LedgerEvent(
                index=-1,
                event_type="ingest",
                doc_id="d",
                tenant_id="t",
                source="s",
                content_hash="h",
                content_root="r",
                chunk_ids=("c0",),
                leaf_hashes=("a",),
                removed_chunk_ids=(),
                supersedes=(),
                timestamp=0.0,
                parent_hash="0" * 64,
                tag="0" * 64,
            )


# --- chunk lifecycle + proofs --------------------------------------


class TestChunkLifecycle:
    def test_provenance_of_admitted_chunk(self, ledger_path):
        ledger = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        _ingest_doc1(ledger)
        prov = ledger.provenance_of("doc1:c1")
        assert prov is not None
        assert prov.doc_id == "doc1"
        assert prov.source == "manual.pdf"
        assert prov.event_index == 0
        assert prov.verified

    def test_unknown_chunk_returns_none(self, ledger_path):
        ledger = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        _ingest_doc1(ledger)
        assert ledger.provenance_of("missing") is None

    def test_update_retires_old_admits_new(self, ledger_path):
        ledger = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        _ingest_doc1(ledger)
        ledger.record_update(
            doc_id="doc1",
            tenant_id="t1",
            source="manual_v2.pdf",
            content_hash="h2",
            chunk_leaves=[("doc1:rev:c0", _leaf("alpha2"))],
            removed_chunk_ids=["doc1:c0", "doc1:c1", "doc1:c2"],
        )
        assert ledger.provenance_of("doc1:c1") is None
        new_prov = ledger.provenance_of("doc1:rev:c0")
        assert new_prov is not None
        assert new_prov.event_index == 1
        assert new_prov.verified

    def test_delete_retires_chunks(self, ledger_path):
        ledger = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        _ingest_doc1(ledger)
        event = ledger.record_delete(
            doc_id="doc1",
            tenant_id="t1",
            removed_chunk_ids=["doc1:c0", "doc1:c1", "doc1:c2"],
        )
        assert event.event_type == "delete"
        assert ledger.provenance_of("doc1:c0") is None
        assert ledger.verify() == (True, None)

    def test_delete_requires_chunks(self, ledger_path):
        ledger = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        with pytest.raises(ValueError, match="at least one chunk"):
            ledger.record_delete(doc_id="d", tenant_id="t", removed_chunk_ids=[])

    def test_supersede_records_lineage_and_retires(self, ledger_path):
        ledger = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        _ingest_doc1(ledger)
        ledger.record_ingest(
            doc_id="doc1_v2",
            tenant_id="t1",
            source="manual_v2.pdf",
            content_hash="h2",
            chunk_leaves=[("doc1_v2:c0", _leaf("alpha2"))],
        )
        event = ledger.record_supersede(
            doc_id="doc1_v2",
            tenant_id="t1",
            source="manual_v2.pdf",
            supersedes=["doc1"],
            removed_chunk_ids=["doc1:c0", "doc1:c1", "doc1:c2"],
        )
        assert event.event_type == "supersede"
        assert event.supersedes == ("doc1",)
        assert ledger.provenance_of("doc1:c0") is None
        assert ledger.provenance_of("doc1_v2:c0").verified
        assert ledger.verify() == (True, None)

    def test_supersede_survives_reload(self, ledger_path):
        writer = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        _ingest_doc1(writer)
        writer.record_supersede(
            doc_id="doc1_v2",
            tenant_id="t1",
            source="manual_v2.pdf",
            supersedes=["doc1"],
            removed_chunk_ids=["doc1:c0", "doc1:c1", "doc1:c2"],
        )
        reader = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        assert reader.verify() == (True, None)
        assert reader.provenance_of("doc1:c0") is None

    def test_supersede_requires_targets(self, ledger_path):
        ledger = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        with pytest.raises(ValueError, match="at least one superseded document"):
            ledger.record_supersede(
                doc_id="d",
                tenant_id="t",
                source="s",
                supersedes=[],
                removed_chunk_ids=["d:c0"],
            )

    def test_supersede_requires_chunks(self, ledger_path):
        ledger = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        with pytest.raises(ValueError, match="at least one chunk"):
            ledger.record_supersede(
                doc_id="d",
                tenant_id="t",
                source="s",
                supersedes=["old"],
                removed_chunk_ids=[],
            )


# --- history queries -----------------------------------------------


class TestHistory:
    def test_history_for_document(self, ledger_path):
        ledger = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        _ingest_doc1(ledger)
        ledger.record_ingest(
            doc_id="doc2",
            tenant_id="t1",
            source="s",
            content_hash="h",
            chunk_leaves=[("doc2:c0", _leaf("x"))],
        )
        ledger.record_delete(
            doc_id="doc1", tenant_id="t1", removed_chunk_ids=["doc1:c0"]
        )
        history = ledger.history_for("doc1")
        assert [event.event_type for event in history] == ["ingest", "delete"]

    def test_history_filtered_by_tenant(self, ledger_path):
        ledger = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        ledger.record_ingest(
            doc_id="shared",
            tenant_id="t1",
            source="s",
            content_hash="h",
            chunk_leaves=[("shared:t1:c0", _leaf("a"))],
        )
        ledger.record_ingest(
            doc_id="shared",
            tenant_id="t2",
            source="s",
            content_hash="h",
            chunk_leaves=[("shared:t2:c0", _leaf("b"))],
        )
        assert len(ledger.history_for("shared")) == 2
        assert len(ledger.history_for("shared", tenant_id="t2")) == 1


# --- persistence ---------------------------------------------------


class TestPersistence:
    def test_reload_preserves_and_verifies(self, ledger_path):
        writer = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        _ingest_doc1(writer)
        writer.record_update(
            doc_id="doc1",
            tenant_id="t1",
            source="manual_v2.pdf",
            content_hash="h2",
            chunk_leaves=[("doc1:rev:c0", _leaf("alpha2"))],
            removed_chunk_ids=["doc1:c0", "doc1:c1", "doc1:c2"],
        )
        reader = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        assert len(reader) == 2
        assert reader.verify() == (True, None)
        assert reader.provenance_of("doc1:rev:c0").verified
        assert reader.provenance_of("doc1:c0") is None

    def test_appends_continue_after_reload(self, ledger_path):
        writer = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        _ingest_doc1(writer)
        reader = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        reader.record_ingest(
            doc_id="doc2",
            tenant_id="t1",
            source="s",
            content_hash="h",
            chunk_leaves=[("doc2:c0", _leaf("y"))],
        )
        assert reader.verify() == (True, None)
        assert len(reader) == 2

    def test_in_memory_ledger_writes_no_file(self, tmp_path):
        ledger = KnowledgeProvenanceLedger(secret=_SECRET, path=None)
        _ingest_doc1(ledger)
        assert len(ledger) == 1
        assert not any(tmp_path.iterdir())

    def test_reload_skips_blank_lines(self, ledger_path):
        writer = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        _ingest_doc1(writer)
        # A stray blank line must not break the reload.
        with open(ledger_path, "a", encoding="utf-8") as handle:
            handle.write("\n")
        reader = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        assert len(reader) == 1
        assert reader.verify() == (True, None)


# --- tamper detection ----------------------------------------------


class TestTamperDetection:
    def _tamper(self, path, replace_from: str, replace_to: str) -> None:
        path.write_text(path.read_text().replace(replace_from, replace_to))

    def test_field_edit_detected(self, ledger_path):
        writer = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        _ingest_doc1(writer)
        self._tamper(ledger_path, "manual.pdf", "evil.pdf")
        with pytest.raises(LedgerTamperError, match="integrity check"):
            KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)

    def test_wrong_secret_detected(self, ledger_path):
        writer = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        _ingest_doc1(writer)
        with pytest.raises(LedgerTamperError):
            KnowledgeProvenanceLedger(
                secret=b"another-secret-key-at-least-32bytes!", path=ledger_path
            )

    def test_reordered_events_detected(self, ledger_path):
        writer = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        _ingest_doc1(writer)
        writer.record_ingest(
            doc_id="doc2",
            tenant_id="t1",
            source="s",
            content_hash="h",
            chunk_leaves=[("doc2:c0", _leaf("z"))],
        )
        lines = ledger_path.read_text().splitlines()
        ledger_path.write_text("\n".join(reversed(lines)) + "\n")
        with pytest.raises(LedgerTamperError):
            KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)

    def test_verify_reports_first_bad_index(self):
        ledger = KnowledgeProvenanceLedger(secret=_SECRET, path=None)
        _ingest_doc1(ledger)
        ledger.record_ingest(
            doc_id="doc2",
            tenant_id="t1",
            source="s",
            content_hash="h",
            chunk_leaves=[("doc2:c0", _leaf("z"))],
        )
        # Forge the second event's tag in place.
        events = ledger.snapshot()
        forged = LedgerEvent(
            **{**vars(events[1]), "tag": "0" * 64},
        )
        ledger._events[1] = forged  # noqa: SLF001 - white-box tamper test
        ok, bad_index = ledger.verify()
        assert ok is False
        assert bad_index == 1


# --- serialisation -------------------------------------------------


class TestSerialisation:
    def test_event_json_round_trip(self, ledger_path):
        ledger = KnowledgeProvenanceLedger(secret=_SECRET, path=ledger_path)
        event = _ingest_doc1(ledger)
        restored = LedgerEvent.from_json(event.to_json())
        assert restored == event

    def test_event_type_validation(self):
        with pytest.raises(ValueError, match="event_type"):
            LedgerEvent(
                index=0,
                event_type="bogus",
                doc_id="d",
                tenant_id="t",
                source="s",
                content_hash="h",
                content_root="r",
                chunk_ids=("c0",),
                leaf_hashes=("a",),
                removed_chunk_ids=(),
                supersedes=(),
                timestamp=0.0,
                parent_hash="0" * 64,
                tag="0" * 64,
            )

    def test_chunk_leaf_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            LedgerEvent(
                index=0,
                event_type="ingest",
                doc_id="d",
                tenant_id="t",
                source="s",
                content_hash="h",
                content_root="r",
                chunk_ids=("c0", "c1"),
                leaf_hashes=("a",),
                removed_chunk_ids=(),
                supersedes=(),
                timestamp=0.0,
                parent_hash="0" * 64,
                tag="0" * 64,
            )


# --- clock injection -----------------------------------------------


class TestClock:
    def test_injected_clock_timestamps(self, ledger_path):
        ledger = KnowledgeProvenanceLedger(
            secret=_SECRET, path=ledger_path, clock=_clock(111.0, 222.0)
        )
        first = _ingest_doc1(ledger)
        second = ledger.record_delete(
            doc_id="doc1", tenant_id="t1", removed_chunk_ids=["doc1:c0"]
        )
        assert first.timestamp == 111.0
        assert second.timestamp == 222.0
