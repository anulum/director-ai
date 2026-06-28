# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real-surface coverage for document parser, chunker, and registry wiring."""

from __future__ import annotations

import hashlib
import importlib
from typing import cast

import pytest

from director_ai.core.retrieval.doc_chunker import ChunkConfig, split
from director_ai.core.retrieval.doc_parser import parse
from director_ai.core.retrieval.doc_registry import DocRegistry


def _minimal_text_pdf(text: str) -> bytes:
    """Build a small valid PDF containing extractable Helvetica text."""
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length "
        + str(len(content)).encode("ascii")
        + b" >>\nstream\n"
        + content
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{object_number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"

    xref_offset = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii")
    for offset in offsets:
        pdf += f"{offset:010d} 00000 n \n".encode("ascii")
    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")
    return bytes(pdf)


def _minimal_blank_pdf() -> bytes:
    """Build a small valid PDF page without extractable text."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{object_number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"

    xref_offset = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii")
    for offset in offsets:
        pdf += f"{offset:010d} 00000 n \n".encode("ascii")
    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")
    return bytes(pdf)


def test_public_compatibility_paths_delegate_to_retrieval_modules() -> None:
    """Compatibility import paths expose the same runtime retrieval functions."""
    public_chunker = importlib.import_module("director_ai.core.doc_chunker")
    public_parser = importlib.import_module("director_ai.core.doc_parser")
    public_registry = importlib.import_module("director_ai.core.doc_registry")

    assert public_chunker.ChunkConfig is ChunkConfig
    assert public_chunker.split is split
    assert public_parser.parse is parse
    assert public_registry.DocRegistry is DocRegistry


def test_document_pipeline_registers_real_ingestion_record() -> None:
    """A parsed document can be chunked and registered through real APIs."""
    html = b"""
    <html>
      <body>
        <header>Navigation</header>
        <main>
          <p>Tenant safety policy requires a documented rollback plan.</p>
          <p>Every deployment must retain operator approval evidence.</p>
        </main>
        <script>window.evil = true;</script>
      </body>
    </html>
    """

    parsed = parse(html, "policy.html")
    chunks = split(parsed, ChunkConfig(chunk_size=72, overlap=0))
    content_hash = hashlib.sha256(parsed.encode("utf-8")).hexdigest()
    registry = DocRegistry()

    record = registry.register(
        "policy-html",
        "policy.html",
        "tenant-a",
        [f"policy-html:chunk:{index}" for index, _chunk in enumerate(chunks)],
        content_hash=content_hash,
    )

    assert "Tenant safety policy" in parsed
    assert "operator approval evidence" in parsed
    assert "Navigation" not in parsed
    assert "window.evil" not in parsed
    assert len(chunks) >= 2
    assert record.doc_id == "policy-html"
    assert record.source == "policy.html"
    assert record.tenant_id == "tenant-a"
    assert record.chunk_count == len(chunks)
    assert record.content_hash == content_hash
    assert registry.get("policy-html", "tenant-b") is None


def test_registry_snapshots_are_detached_from_real_state() -> None:
    """Returned records cannot mutate stored registry state."""
    registry = DocRegistry()
    registered = registry.register(
        "ops-runbook",
        "runbook.md",
        "tenant-a",
        ["ops-runbook:chunk:0"],
    )
    registered.source = "mutated.md"
    registered.chunk_ids.append("ops-runbook:chunk:evil")

    stored = registry.get("ops-runbook", "tenant-a")
    listed = registry.list_for_tenant("tenant-a")

    assert stored is not None
    assert stored.source == "runbook.md"
    assert stored.chunk_ids == ["ops-runbook:chunk:0"]
    assert listed == [stored]
    assert listed[0] is not stored


def test_parser_chunker_registry_round_trip_updates_and_deletes() -> None:
    """CSV and text payloads round-trip through chunk metadata lifecycle calls."""
    csv_text = parse(
        b"requirement,owner\nrollback approval,operations\nhash evidence,security\n",
        "controls.csv",
    )
    note_text = parse(
        b"Document updates must preserve tenant isolation. "
        b"Deletion must return the final stored snapshot.",
        "notes.txt",
    )
    csv_chunks = split(csv_text, ChunkConfig(chunk_size=48, overlap=0))
    note_chunks = split(note_text, ChunkConfig(chunk_size=58, overlap=0))
    registry = DocRegistry()

    registry.register(
        "controls",
        "controls.csv",
        "tenant-a",
        [f"controls:chunk:{index}" for index, _chunk in enumerate(csv_chunks)],
    )
    updated = registry.update(
        "controls",
        [f"controls:chunk:{index}" for index, _chunk in enumerate(note_chunks)],
        source="notes.txt",
        content_hash=hashlib.sha256(note_text.encode("utf-8")).hexdigest(),
    )
    deleted = registry.delete("controls")

    assert "rollback approval | operations" in csv_text
    assert "hash evidence | security" in csv_text
    assert updated.source == "notes.txt"
    assert updated.chunk_count == len(note_chunks)
    assert deleted is not None
    assert deleted.source == "notes.txt"
    assert registry.count == 0
    assert registry.exists("controls") is False


def test_pdf_parser_extracts_text_from_valid_document() -> None:
    """PDF parsing reads text from a real single-page PDF payload."""
    parsed = parse(_minimal_text_pdf("PDF approval evidence"), "evidence.pdf")

    assert parsed == "PDF approval evidence"


def test_pdf_parser_ignores_blank_pages() -> None:
    """PDF parsing skips pages that do not expose text."""
    parsed = parse(_minimal_blank_pdf(), "blank.pdf")

    assert parsed == ""


def test_registry_rejects_invalid_chunk_identifier_sequences() -> None:
    """Registry registration validates chunk-id containers and values."""
    registry = DocRegistry()

    with pytest.raises(ValueError, match="chunk_ids"):
        registry.register(
            "bad-string",
            "doc.txt",
            "tenant-a",
            cast("list[str]", "not-a-list"),
        )
    with pytest.raises(ValueError, match="chunk_ids"):
        registry.register("bad-value", "doc.txt", "tenant-a", [""])


def test_registry_update_without_source_preserves_existing_source() -> None:
    """Updating only chunks and hash leaves the registered source unchanged."""
    registry = DocRegistry()
    registry.register("source-stable", "original.txt", "tenant-a", ["chunk:0"])

    updated = registry.update(
        "source-stable",
        ["chunk:0", "chunk:1"],
        content_hash="  refreshed-hash  ",
    )

    assert updated.source == "original.txt"
    assert updated.content_hash == "refreshed-hash"
    assert updated.chunk_count == 2


def test_registry_missing_update_and_delete_are_explicit() -> None:
    """Missing updates raise while missing deletes return ``None``."""
    registry = DocRegistry()

    with pytest.raises(KeyError, match="missing"):
        registry.update("missing", ["chunk:0"])

    assert registry.delete("missing") is None
