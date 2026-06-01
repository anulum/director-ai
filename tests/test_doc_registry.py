# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Document registry contracts
"""Behavioural tests for document registry tenant and lifecycle rules."""

import pytest

from director_ai.core.retrieval.doc_registry import DocRegistry


def test_update_unknown_document_fails_explicitly() -> None:
    registry = DocRegistry()

    with pytest.raises(KeyError):
        registry.update("missing", ["chunk-0"])


def test_wrong_tenant_cannot_read_registered_document() -> None:
    registry = DocRegistry()
    registry.register("doc-1", "source.pdf", "tenant-a", ["chunk-0"])

    assert registry.get("doc-1", "tenant-b") is None


def test_registered_record_preserves_source_tenant_and_chunk_count() -> None:
    registry = DocRegistry()

    record = registry.register("doc-1", "source.pdf", "tenant-a", ["c0", "c1"])

    assert record.doc_id == "doc-1"
    assert record.source == "source.pdf"
    assert record.tenant_id == "tenant-a"
    assert record.chunk_count == 2
    assert record.chunk_ids == ["c0", "c1"]
    assert record.updated_at >= record.created_at


@pytest.mark.parametrize("chunk_ids", ["chunk-0", b"chunk-0", ["chunk-0", " "]])
def test_chunk_id_validation_rejects_string_bytes_and_blank_members(
    chunk_ids,
) -> None:
    registry = DocRegistry()

    with pytest.raises(ValueError, match="chunk_ids"):
        registry.register("doc-1", "source.pdf", "tenant-a", chunk_ids)


def test_delete_missing_document_returns_none() -> None:
    registry = DocRegistry()

    assert registry.delete("missing-doc") is None


def test_update_without_optional_source_or_hash_keeps_existing_metadata() -> None:
    registry = DocRegistry()
    registry.register(
        "doc-1",
        "source.pdf",
        "tenant-a",
        ["chunk-0"],
        content_hash="hash-a",
    )

    updated = registry.update("doc-1", ["chunk-1"])

    assert updated.source == "source.pdf"
    assert updated.content_hash == "hash-a"
    assert updated.chunk_ids == ["chunk-1"]


def test_update_can_replace_content_hash_without_source_change() -> None:
    registry = DocRegistry()
    registry.register("doc-1", "source.pdf", "tenant-a", ["chunk-0"])

    updated = registry.update("doc-1", ["chunk-1"], content_hash=" hash-b ")

    assert updated.source == "source.pdf"
    assert updated.content_hash == "hash-b"
