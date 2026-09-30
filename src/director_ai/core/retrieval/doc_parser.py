# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Document Parser

"""File format → plain text. Heavy deps imported lazily.

Supports: PDF, DOCX, HTML, CSV, TXT, Markdown.
Install optional deps: ``pip install director-ai[ingestion]``
"""

from __future__ import annotations

import csv
import io
import logging
from collections.abc import Callable as _Callable

from ..exceptions import DomainRefusalError

logger = logging.getLogger("DirectorAI.DocParser")


def parse(content: bytes, filename: str) -> str:
    """Extract plain text using the filename's format parser.

    Parameters
    ----------
    content : bytes
        Complete uploaded document bytes.
    filename : str
        Filename whose extension selects the parser.

    Returns
    -------
    str
        Extracted plain text; unknown extensions use the text decoder.

    Raises
    ------
    DomainRefusalError
        Content or filename is invalid, or PDF/DOCX parsing refuses the document.
    ImportError
        The selected optional format parser is unavailable.
    """
    if not isinstance(content, bytes):
        raise DomainRefusalError("content must be bytes")
    if not isinstance(filename, str) or not filename.strip():
        raise DomainRefusalError("filename must be a non-empty string")

    normalized_filename = filename.strip()
    ext = (
        normalized_filename.rsplit(".", 1)[-1].lower()
        if "." in normalized_filename
        else ""
    )
    parser = _PARSERS.get(ext, _parse_text)
    return parser(content)


def _parse_pdf(content: bytes) -> str:
    """Extract text from PDF pages with the optional pypdf reader.

    Parameters
    ----------
    content : bytes
        Encoded PDF document.

    Returns
    -------
    str
        Non-empty page text separated by blank lines.

    Raises
    ------
    DomainRefusalError
        The PDF reader refuses the document while loading or extracting pages.
    ImportError
        pypdf is unavailable.
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(
            "pypdf required for PDF parsing. Install: pip install director-ai[ingestion]"
        ) from e

    try:
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except Exception as e:
        exc_type = type(e)
        if exc_type.__module__.startswith("pypdf") or exc_type.__name__.startswith(
            "Pdf"
        ):
            raise DomainRefusalError("invalid PDF document") from e
        raise  # pragma: no cover - preserves unexpected non-pypdf failures.


def _parse_docx(content: bytes) -> str:
    """Extract non-empty DOCX paragraphs with python-docx.

    Parameters
    ----------
    content : bytes
        Encoded DOCX archive.

    Returns
    -------
    str
        Non-empty paragraph text separated by blank lines.

    Raises
    ------
    DomainRefusalError
        The archive or document fails the parser's input validation.
    ImportError
        python-docx is unavailable.
    """
    import zipfile

    try:
        from docx import Document
        from lxml.etree import XMLSyntaxError
    except ImportError as e:
        raise ImportError(
            "python-docx required for DOCX parsing. Install: pip install director-ai[ingestion]"
        ) from e

    try:
        doc = Document(io.BytesIO(content))
    except (ValueError, zipfile.BadZipFile, XMLSyntaxError) as e:
        raise DomainRefusalError("invalid DOCX document") from e
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _parse_html(content: bytes) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise ImportError(
            "beautifulsoup4 required for HTML parsing. Install: pip install director-ai[ingestion]"
        ) from e

    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return str(soup.get_text(separator="\n", strip=True))


def _parse_csv(content: bytes) -> str:
    text = content.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows = []
    for row in reader:
        rows.append(" | ".join(cell.strip() for cell in row if cell.strip()))
    return "\n".join(rows)


def _parse_text(content: bytes) -> str:
    return content.decode("utf-8", errors="replace")


_PARSERS: dict[str, _Callable[[bytes], str]] = {
    "pdf": _parse_pdf,
    "docx": _parse_docx,
    "html": _parse_html,
    "htm": _parse_html,
    "csv": _parse_csv,
    "txt": _parse_text,
    "md": _parse_text,
    "markdown": _parse_text,
    "json": _parse_text,
    "xml": _parse_text,
}
