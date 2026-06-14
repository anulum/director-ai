# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — document parser tests

from __future__ import annotations

import pytest

from director_ai.core.retrieval.doc_parser import parse


def test_parse_plain_text_decodes_utf8():
    assert parse("héllo wörld".encode(), "notes.txt") == "héllo wörld"


def test_unknown_extension_falls_back_to_text():
    assert parse(b"raw payload", "data.unknownext") == "raw payload"


def test_no_extension_falls_back_to_text():
    assert parse(b"plain", "README") == "plain"


def test_parse_csv_joins_cells():
    assert "a | b" in parse(b"a,b\n1,2\n", "table.csv")


def test_rejects_non_bytes_content():
    with pytest.raises(ValueError, match="content must be bytes"):
        parse("a string, not bytes", "doc.txt")  # type: ignore[arg-type]


def test_rejects_empty_filename():
    with pytest.raises(ValueError, match="filename must be a non-empty string"):
        parse(b"x", "   ")


def test_invalid_pdf_raises_clean_value_error():
    # Garbage bytes with a .pdf extension reach _parse_pdf; pypdf rejects them
    # and the parser surfaces a clean ValueError rather than a pypdf-internal
    # exception.
    with pytest.raises(ValueError, match="invalid PDF document"):
        parse(b"this is definitely not a PDF file", "broken.pdf")


def test_non_pypdf_exception_is_reraised(monkeypatch):
    # A failure that is not a pypdf parse error (e.g. an I/O fault) must
    # propagate unchanged rather than being masked as "invalid PDF document".
    import pypdf

    def _raise_runtime(_stream):
        raise RuntimeError("backing store unavailable")

    monkeypatch.setattr(pypdf, "PdfReader", _raise_runtime)
    with pytest.raises(RuntimeError, match="backing store unavailable"):
        parse(b"whatever", "doc.pdf")
