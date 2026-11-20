# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — document ingestion pipeline properties
"""Bounded KB-write properties over the real registry, index and ledger."""

from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from director_ai.core.ingestion import DocumentIngestionPipeline, IngestionConfig
from director_ai.core.provenance import KnowledgeProvenanceLedger
from director_ai.core.retrieval.vector_store import (
    InMemoryBackend,
    VectorGroundTruthStore,
)

_LEDGER_KEY = b"test-fixture-document-ingest-provenance-key"
_UNICODE_WORD = st.text(alphabet="aé中žΩ", min_size=1, max_size=6)


class _InterruptOnce(InMemoryBackend):
    """Raise a storage error once, before the selected backend write."""

    def __init__(self, fail_at: int) -> None:
        super().__init__()
        self._writes = 0
        self._fail_at = fail_at

    def add(
        self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Delegate real writes on either side of one simulated I/O failure."""
        self._writes += 1
        if self._writes == self._fail_at:
            raise OSError("simulated storage interruption")
        super().add(doc_id, text, metadata)


@settings(max_examples=20, deadline=2000, database=None)
@given(word=_UNICODE_WORD, fail_at=st.integers(min_value=2, max_value=4))
def test_interrupted_chunk_write_leaves_no_partial_document(
    word: str, fail_at: int
) -> None:
    """Roll back a partially written document, then retry."""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "provenance.jsonl"
        ledger = KnowledgeProvenanceLedger(secret=_LEDGER_KEY, path=path)
        backend = _InterruptOnce(fail_at)
        store = VectorGroundTruthStore(backend=backend)
        pipeline = DocumentIngestionPipeline(
            store=store, ledger=ledger, config=IngestionConfig(chunk_size=21, overlap=0)
        )
        text = "\n\n".join(f"policy {word} item {i}" for i in range(4))

        with pytest.raises(OSError, match="simulated storage interruption"):
            pipeline.ingest_text(
                text, doc_id="policy", source="notes", tenant_id="alpha"
            )

        assert len(ledger) == 0
        assert backend.count() == 0
        assert not store.facts
        assert pipeline.registry.get("policy", "alpha") is None

        result = pipeline.ingest_text(
            text, doc_id="policy", source="notes", tenant_id="alpha"
        )
        assert result.chunk_count == backend.count() == 4
        assert set(result.chunk_ids) == set(store.facts)
        assert len(ledger) == 1
        assert ledger.verify() == (True, None)


@settings(max_examples=32, deadline=2000, database=None)
@given(word=_UNICODE_WORD, replays=st.integers(min_value=1, max_value=4))
def test_repeated_updates_preserve_tenant_scope_and_durable_provenance(
    word: str, replays: int
) -> None:
    """Keep unchanged writes idempotent and preserve each tenant's latest proof."""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "provenance.jsonl"
        ledger = KnowledgeProvenanceLedger(secret=_LEDGER_KEY, path=path)
        backend = InMemoryBackend()
        store = VectorGroundTruthStore(backend=backend)
        pipeline = DocumentIngestionPipeline(store=store, ledger=ledger)
        peer = pipeline.ingest_text(
            f"policy older {word}", doc_id="peer", source="peer.txt", tenant_id="beta"
        )
        first = pipeline.ingest_text(
            f"policy initial {word}",
            doc_id="policy",
            source="old.txt",
            tenant_id="alpha",
        )
        latest_text = f"policy revised {word}"
        with pytest.raises(KeyError, match="not found"):
            pipeline.update_text(latest_text, doc_id="policy", tenant_id="beta")

        latest = pipeline.update_text(
            latest_text, doc_id="policy", source="new.txt", tenant_id="alpha"
        )
        for _ in range(replays):
            repeated = pipeline.update_text(
                latest_text, doc_id="policy", source="new.txt", tenant_id="alpha"
            )
            assert repeated.unchanged
            assert repeated.chunk_ids == latest.chunk_ids

        assert backend.count() == 2
        assert first.chunk_ids[0] not in store.facts
        own_rows = backend.query("policy", tenant_id="alpha")
        peer_rows = backend.query("policy", tenant_id="beta")
        assert [row["id"] for row in own_rows] == latest.chunk_ids
        assert own_rows[0]["text"] == latest_text
        assert own_rows[0]["metadata"]["source"] == "new.txt"
        assert [row["id"] for row in peer_rows] == peer.chunk_ids

        restored = KnowledgeProvenanceLedger(secret=_LEDGER_KEY, path=path)
        assert len(restored) == 3
        assert restored.verify() == (True, None)
        assert restored.provenance_of(first.chunk_ids[0]) is None
        proof = restored.provenance_of(latest.chunk_ids[0])
        assert proof is not None
        assert (proof.doc_id, proof.source, proof.tenant_id) == (
            "policy",
            "new.txt",
            "alpha",
        )
        assert proof.verified
        assert proof.proof.leaf == hashlib.sha256(latest_text.encode()).digest()
        peer_proof = restored.provenance_of(peer.chunk_ids[0])
        assert peer_proof is not None
        assert (peer_proof.source, peer_proof.tenant_id) == ("peer.txt", "beta")
        assert peer_proof.verified
