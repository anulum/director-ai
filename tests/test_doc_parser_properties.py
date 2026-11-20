# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — document parser properties
"""Bounded format properties for the public document parser."""

from __future__ import annotations

import csv
from html import escape
from importlib import import_module
from io import BytesIO, StringIO

import pytest
from docx import Document
from hypothesis import example, given, settings
from hypothesis import strategies as st

from director_ai.core.exceptions import DomainRefusalError
from director_ai.core.retrieval.doc_parser import parse

_UNICODE_WORD = st.text(alphabet="Aé中žΩ", min_size=1, max_size=16)
_CSV_CELL = st.text(alphabet='Aé中žΩ," \n', max_size=24)


@pytest.fixture(scope="module", autouse=True)
def load_required_format_readers() -> None:
    """Load the locked readers once outside the per-example deadline."""
    import_module("bs4")
    import_module("pypdf")


@settings(max_examples=32, deadline=1000, database=None)
@given(text=st.text(max_size=128), invalid=st.booleans())
@example(text="Žltý kôň 🐎", invalid=True)
def test_text_preserves_unicode_with_replacement(text: str, invalid: bool) -> None:
    """Decode valid text and replace a terminal invalid byte in place."""
    encoded = text.encode()
    if invalid:
        encoded += b"\xff"
    assert parse(encoded, "notes.MD") == text + ("\ufffd" if invalid else "")


@settings(max_examples=32, deadline=1000, database=None)
@given(
    rows=st.lists(st.lists(_CSV_CELL, min_size=1, max_size=5), min_size=1, max_size=5)
)
@example(rows=[["first,second", 'quo"te', "多\n行", "  "], ["café", "d"]])
def test_csv_preserves_quoted_cells_and_rows(rows: list[list[str]]) -> None:
    """Recover nonblank cell contents independently of CSV quoting."""
    stream = StringIO(newline="")
    csv.writer(stream).writerows(rows)
    expected = "\n".join(
        " | ".join(cell.strip() for cell in row if cell.strip()) for row in rows
    )
    assert parse(stream.getvalue().encode(), "table.CSV") == expected


@settings(max_examples=24, deadline=1000, database=None)
@given(visible=_UNICODE_WORD, omitted=_UNICODE_WORD)
def test_html_extracts_only_visible_body_text(visible: str, omitted: str) -> None:
    """Keep visible Unicode text and omit header, footer and script content."""
    visible_text = f"Article {visible} & facts"
    omitted_text = escape(f"Boilerplate {omitted}")
    html = (
        '<meta charset="utf-8">'
        f"<header>{omitted_text}</header><script>{omitted_text}</script>"
        f"<main>{escape(visible_text)}</main><footer>{omitted_text}</footer>"
    )
    assert parse(html.encode(), "page.htm") == visible_text


@settings(max_examples=12, deadline=2000, database=None)
@given(paragraphs=st.lists(_UNICODE_WORD, min_size=1, max_size=4))
def test_docx_preserves_nonblank_paragraph_order(paragraphs: list[str]) -> None:
    """Read Unicode through the actual word-processing file format."""
    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
        document.add_paragraph("")
    encoded = BytesIO()
    document.save(encoded)
    assert parse(encoded.getvalue(), "notes.docx") == "\n\n".join(paragraphs)


@settings(max_examples=20, deadline=1000, database=None)
@given(text=_UNICODE_WORD, format_name=st.sampled_from(("PDF", "DOCX")))
@example(text="ž", format_name="PDF")
@example(text="Ω", format_name="DOCX")
def test_binary_formats_refuse_incorrect_document_type(
    text: str, format_name: str
) -> None:
    """Reject ordinary text saved with a binary document extension."""
    with pytest.raises(DomainRefusalError, match=f"invalid {format_name} document"):
        parse(f"Plain text {text}".encode(), f"notes.{format_name.lower()}")
