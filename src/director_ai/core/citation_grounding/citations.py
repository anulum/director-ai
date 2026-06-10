# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — citation extraction

"""Extract and resolve the citations in a generated answer.

A grounded answer cites real sources for its factual assertions. To check that
grounding, the citing references first have to be located and resolved to
concrete source identifiers. Four citation styles are recognised:

* **numeric** — inline ``[1]``, ``[2, 3]``, ``[4-6]`` markers that index a
  trailing reference list;
* **DOI** — ``10.xxxx/...`` identifiers, with or without a ``doi:`` /
  ``https://doi.org/`` prefix;
* **arXiv** — ``arXiv:2411.04368`` or ``arxiv.org/abs/2411.04368`` (new ids and
  the legacy ``cond-mat/0211034`` form);
* **URL** — bare ``http(s)://`` links;
* **author-year** — ``(Smith et al., 2024)`` / ``(Doe 2023)`` parentheticals.

:func:`extract_inline_citations` finds every citing marker in the answer body.
:func:`parse_reference_section` reads a trailing *References* / *Bibliography*
block and maps each numeric label to the concrete identifier it points at.
:func:`resolve_citations` combines the two: numeric markers are resolved through
the reference map to their DOI/arXiv/URL, while inline DOI/arXiv/URL citations
are already concrete and pass through unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "Citation",
    "CitationKind",
    "extract_inline_citations",
    "parse_reference_section",
    "resolve_citations",
]


class CitationKind(Enum):
    """The form a citation takes in the text."""

    NUMERIC = "numeric"
    DOI = "doi"
    ARXIV = "arxiv"
    URL = "url"
    AUTHOR_YEAR = "author_year"


# A DOI is ``10.<registrant>/<suffix>``; the suffix runs to whitespace or a
# closing bracket/paren and never swallows trailing sentence punctuation (a DOI
# may contain an internal ``.`` but does not end in ``.``/``,``/``;``).
_DOI_CORE = r"10\.\d{4,9}/[^\s\]\)<>\"']*[^\s\]\)<>\"'.,;]"
_DOI_RE = re.compile(rf"(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?({_DOI_CORE})", re.I)
# New-style arXiv id (YYMM.NNNNN, optional version) or the legacy archive form.
_ARXIV_RE = re.compile(
    r"arxiv[:\s]\s*(\d{4}\.\d{4,5}(?:v\d+)?)"
    r"|arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)"
    r"|arxiv[:\s]\s*([a-z][a-z.-]+/\d{7})",
    re.I,
)
# Stop the URL before trailing sentence punctuation (``review.`` → ``review``).
_URL_RE = re.compile(r"https?://[^\s\]\)<>\"']*[^\s\]\)<>\"'.,;:]")
_NUMERIC_RE = re.compile(r"\[(\d+(?:\s*[-,]\s*\d+)*)\]")
# ``(Name, YEAR)`` / ``(Name et al., YEAR)`` / ``(Name and Other, YEAR)``.
_AUTHOR_YEAR_RE = re.compile(
    r"\(([A-Z][A-Za-z.'’-]+(?:\s+(?:et al\.?|and|&)\s*[A-Za-z.'’-]*)?),?\s+"
    r"(\d{4}[a-z]?)\)"
)
# A reference-list line: a leading ``[n]`` / ``n.`` / ``n)`` label.
_REF_LABEL_RE = re.compile(r"^\s*(?:\[(\d+)\]|(\d+)[.)])\s+(.*)$")
_SECTION_RE = re.compile(r"(?im)^\s*(references|bibliography|works cited)\s*:?\s*$")


@dataclass(frozen=True)
class Citation:
    """One citation occurrence resolved to a concrete identifier where possible.

    ``raw`` is the exact source text; ``identifier`` is the resolved DOI/arXiv
    id/URL (or, for an unresolved numeric marker, the label itself); ``start``
    and ``end`` are character offsets into the text the citation came from.
    """

    raw: str
    kind: CitationKind
    identifier: str
    start: int
    end: int


def _expand_numeric_label(label: str) -> list[str]:
    """Expand ``"1"`` / ``"1, 3"`` / ``"4-6"`` into individual label strings."""
    out: list[str] = []
    for part in label.split(","):
        part = part.strip()
        if "-" in part:
            lo_s, hi_s = (p.strip() for p in part.split("-", 1))
            if lo_s.isdigit() and hi_s.isdigit():
                lo, hi = int(lo_s), int(hi_s)
                if lo <= hi:
                    out.extend(str(n) for n in range(lo, hi + 1))
                    continue
        if part.isdigit():
            out.append(part)
    return out


def _first_identifier(text: str) -> tuple[CitationKind, str] | None:
    """Return the most specific concrete identifier in a reference-list entry.

    DOI and arXiv ids are preferred over a bare URL, since they name the work
    rather than a (possibly redirecting) landing page.
    """
    doi = _DOI_RE.search(text)
    if doi:
        return CitationKind.DOI, doi.group(1)
    arxiv = _ARXIV_RE.search(text)
    if arxiv:
        return CitationKind.ARXIV, next(g for g in arxiv.groups() if g)
    url = _URL_RE.search(text)
    if url:
        return CitationKind.URL, url.group(0)
    return None


def extract_inline_citations(text: str) -> list[Citation]:
    """Find every citing marker in ``text``, ordered by position.

    A DOI/arXiv id that appears inside a URL is reported once, as the more
    specific DOI/arXiv citation, not also as a URL.
    """
    found: list[Citation] = []
    claimed: list[tuple[int, int]] = []  # spans already taken by a citation

    def _overlaps(start: int, end: int) -> bool:
        return any(start < e and end > s for s, e in claimed)

    for kind, pattern in (
        (CitationKind.DOI, _DOI_RE),
        (CitationKind.ARXIV, _ARXIV_RE),
        (CitationKind.URL, _URL_RE),
    ):
        for m in pattern.finditer(text):
            if _overlaps(m.start(), m.end()):
                continue
            if kind is CitationKind.ARXIV:
                ident = next(g for g in m.groups() if g)
            elif kind is CitationKind.DOI:
                ident = m.group(1)
            else:
                ident = m.group(0)
            claimed.append((m.start(), m.end()))
            found.append(Citation(m.group(0), kind, ident, m.start(), m.end()))

    for m in _NUMERIC_RE.finditer(text):
        for label in _expand_numeric_label(m.group(1)):
            found.append(
                Citation(m.group(0), CitationKind.NUMERIC, label, m.start(), m.end())
            )

    for m in _AUTHOR_YEAR_RE.finditer(text):
        ident = f"{m.group(1)} {m.group(2)}"
        found.append(
            Citation(m.group(0), CitationKind.AUTHOR_YEAR, ident, m.start(), m.end())
        )

    found.sort(key=lambda c: (c.start, c.kind.value))
    return found


def parse_reference_section(text: str) -> dict[str, str]:
    """Map each numeric reference label to its concrete identifier.

    Reads from the last *References* / *Bibliography* heading to the end of the
    text. Labels with no resolvable DOI/arXiv/URL are omitted.
    """
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return {}
    body = text[matches[-1].end() :]
    refs: dict[str, str] = {}
    for line in body.splitlines():
        label_match = _REF_LABEL_RE.match(line)
        if not label_match:
            continue
        label = label_match.group(1) or label_match.group(2)
        ident = _first_identifier(label_match.group(3))
        if ident is not None:
            refs[label] = ident[1]
    return refs


def resolve_citations(text: str) -> list[Citation]:
    """Extract citations with numeric markers resolved through the reference list.

    Each numeric marker is replaced by a citation carrying the DOI/arXiv/URL its
    label points at (kind re-typed accordingly); a marker with no matching
    reference entry is dropped. Inline DOI/arXiv/URL/author-year citations that
    fall inside the reference section itself are excluded so a work is not
    counted both as a citation and as its own bibliography entry.
    """
    refs = parse_reference_section(text)
    section = _SECTION_RE.search(text)
    body_end = section.start() if section is not None else len(text)

    resolved: list[Citation] = []
    for cite in extract_inline_citations(text):
        if cite.start >= body_end:
            continue  # inside the reference list, not a citing marker
        if cite.kind is CitationKind.NUMERIC:
            target = refs.get(cite.identifier)
            if target is None:
                continue
            kind = _kind_of_identifier(target)
            resolved.append(Citation(cite.raw, kind, target, cite.start, cite.end))
        else:
            resolved.append(cite)
    return resolved


def _kind_of_identifier(identifier: str) -> CitationKind:
    if _DOI_RE.fullmatch(identifier) or re.fullmatch(_DOI_CORE, identifier):
        return CitationKind.DOI
    if identifier.lower().startswith(("http://", "https://")):
        return CitationKind.URL
    return CitationKind.ARXIV
