# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — document parser tests

"""Check format dispatch, input refusals and PDF reader error translation."""

from __future__ import annotations

from io import BytesIO
from typing import NoReturn

import pytest

from director_ai.core.retrieval.doc_parser import parse


def test_parse_plain_text_decodes_utf8() -> None:
    """Decode the supplied UTF-8 accents without altering the plain text."""
    assert parse("héllo wörld".encode(), "notes.txt") == "héllo wörld"


def test_unknown_extension_falls_back_to_text() -> None:
    """Decode the supplied bytes as text when the extension has no registered parser."""
    assert parse(b"raw payload", "data.unknownext") == "raw payload"


def test_no_extension_falls_back_to_text() -> None:
    """Decode the supplied bytes as text when the filename has no extension."""
    assert parse(b"plain", "README") == "plain"


def test_parse_csv_joins_cells() -> None:
    """Join the supplied CSV header cells with the documented column separator."""
    assert "a | b" in parse(b"a,b\n1,2\n", "table.csv")


def test_rejects_non_bytes_content() -> None:
    """Reject the original text payload through the public parser and match its error."""
    with pytest.raises(ValueError, match="content must be bytes"):
        parse("a string, not bytes", "doc.txt")


def test_rejects_empty_filename() -> None:
    """Reject the supplied whitespace-only filename before format selection."""
    with pytest.raises(ValueError, match="filename must be a non-empty string"):
        parse(b"x", "   ")


def test_invalid_pdf_raises_clean_value_error() -> None:
    # Garbage bytes with a .pdf extension reach _parse_pdf; pypdf rejects them
    # and the parser surfaces a clean ValueError rather than a pypdf-internal
    # exception.
    """Translate the PDF reader refusal of garbage bytes into the public value error."""
    with pytest.raises(ValueError, match="invalid PDF document"):
        parse(b"this is definitely not a PDF file", "broken.pdf")


def test_non_pypdf_exception_is_reraised(monkeypatch: pytest.MonkeyPatch) -> None:
    # A failure that is not a pypdf parse error (e.g. an I/O fault) must
    # propagate unchanged rather than being masked as "invalid PDF document".
    """Propagate the injected reader storage failure through the real PDF parser."""
    import pypdf

    def _raise_runtime(_stream: BytesIO) -> NoReturn:
        """Raise the injected storage failure when the real PDF parser constructs a reader."""
        raise RuntimeError("backing store unavailable")

    monkeypatch.setattr(pypdf, "PdfReader", _raise_runtime)
    with pytest.raises(RuntimeError, match="backing store unavailable"):
        parse(b"whatever", "doc.pdf")
