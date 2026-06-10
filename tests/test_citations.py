# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — citation extraction tests
"""Multi-angle tests for citation extraction and resolution.

Covers the four citation styles (numeric markers incl. lists and ranges, DOIs
with and without prefixes, new and legacy arXiv ids, bare URLs trimmed of
trailing punctuation, author-year parentheticals incl. ``et al.`` / ``and`` /
``&``), the DOI-inside-URL de-duplication, positional ordering, the reference-
section parser (label styles, DOI>arXiv>URL preference, unresolvable labels
dropped, last-heading wins), and ``resolve_citations`` mapping numeric markers
through the reference list while excluding the bibliography itself.
"""

from __future__ import annotations

from director_ai.core.citation_grounding import (
    Citation,
    CitationKind,
    extract_inline_citations,
    parse_reference_section,
    resolve_citations,
)


def _idents(text):
    return [(c.kind, c.identifier) for c in extract_inline_citations(text)]


class TestNumeric:
    def test_single_marker(self):
        assert (CitationKind.NUMERIC, "1") in _idents("A claim [1].")

    def test_list_marker_expands(self):
        idents = [
            i for k, i in _idents("Both [2, 3] agree.") if k is CitationKind.NUMERIC
        ]
        assert idents == ["2", "3"]

    def test_range_marker_expands(self):
        idents = [i for k, i in _idents("See [4-6].") if k is CitationKind.NUMERIC]
        assert idents == ["4", "5", "6"]

    def test_reversed_range_ignored(self):
        idents = [i for k, i in _idents("Bad [6-4].") if k is CitationKind.NUMERIC]
        assert idents == []


class TestDOI:
    def test_bare_doi(self):
        assert (CitationKind.DOI, "10.3847/2041-8213/ab50c5") in _idents(
            "Result 10.3847/2041-8213/ab50c5 holds."
        )

    def test_doi_prefix(self):
        assert (CitationKind.DOI, "10.1000/xyz123") in _idents("doi:10.1000/xyz123")

    def test_doi_url(self):
        # A doi.org URL is reported once, as a DOI, not also as a URL.
        idents = _idents("See https://doi.org/10.1000/xyz123 now.")
        assert (CitationKind.DOI, "10.1000/xyz123") in idents
        assert not any(k is CitationKind.URL for k, _ in idents)

    def test_doi_drops_trailing_period(self):
        ((_, ident),) = [
            i for i in _idents("End 10.1000/abc.") if i[0] is CitationKind.DOI
        ]
        assert ident == "10.1000/abc"


class TestArxiv:
    def test_new_style(self):
        assert (CitationKind.ARXIV, "2411.04368") in _idents("arXiv:2411.04368")

    def test_versioned(self):
        assert (CitationKind.ARXIV, "1912.05705v1") in _idents("arXiv:1912.05705v1")

    def test_abs_url(self):
        idents = _idents("https://arxiv.org/abs/2411.04368")
        assert (CitationKind.ARXIV, "2411.04368") in idents

    def test_legacy_id(self):
        assert (CitationKind.ARXIV, "cond-mat/0211034") in _idents(
            "arXiv:cond-mat/0211034"
        )


class TestUrl:
    def test_plain_url(self):
        assert (CitationKind.URL, "https://example.org/x") in _idents(
            "Visit https://example.org/x for more."
        )

    def test_trailing_punctuation_trimmed(self):
        ((_, ident),) = [
            i
            for i in _idents("At https://example.org/review.")
            if i[0] is CitationKind.URL
        ]
        assert ident == "https://example.org/review"


class TestAuthorYear:
    def test_single_author(self):
        assert (CitationKind.AUTHOR_YEAR, "Doe 2023") in _idents("As shown (Doe 2023).")

    def test_et_al(self):
        assert (CitationKind.AUTHOR_YEAR, "Riess et al. 2022") in _idents(
            "Tension persists (Riess et al., 2022)."
        )

    def test_two_authors(self):
        assert (CitationKind.AUTHOR_YEAR, "Smith and Jones 2021") in _idents(
            "Per (Smith and Jones, 2021)."
        )


class TestOrderingAndDedup:
    def test_ordered_by_position(self):
        cites = extract_inline_citations("First [1] then arXiv:2411.04368 last.")
        positions = [c.start for c in cites]
        assert positions == sorted(positions)


class TestReferenceSection:
    _REFS = (
        "Body [1] and [2].\n\n"
        "References:\n"
        "[1] Bogdanov et al. 10.3847/2041-8213/ab50c5\n"
        "[2] NICER, arXiv:2411.04368\n"
    )

    def test_maps_labels_to_identifiers(self):
        refs = parse_reference_section(self._REFS)
        assert refs == {"1": "10.3847/2041-8213/ab50c5", "2": "2411.04368"}

    def test_dotted_label_style(self):
        refs = parse_reference_section("References\n1. Paper https://example.org/p\n")
        assert refs == {"1": "https://example.org/p"}

    def test_prefers_doi_over_url(self):
        refs = parse_reference_section(
            "References\n[1] Title 10.1000/abc https://example.org/landing\n"
        )
        assert refs["1"] == "10.1000/abc"

    def test_unresolvable_label_omitted(self):
        refs = parse_reference_section("References\n[1] A book with no identifier\n")
        assert refs == {}

    def test_no_section_returns_empty(self):
        assert parse_reference_section("Just prose with [1] marker.") == {}

    def test_last_section_wins(self):
        text = (
            "References\n[1] old https://old.example\n\n"
            "Bibliography\n[1] new https://new.example\n"
        )
        assert parse_reference_section(text)["1"] == "https://new.example"


class TestResolveCitations:
    _DOC = (
        "Radii constrain the EOS [1]. NICER measured this [2].\n\n"
        "References:\n"
        "[1] Bogdanov 10.3847/2041-8213/ab50c5\n"
        "[2] NICER arXiv:2411.04368\n"
    )

    def test_numeric_resolved_to_concrete(self):
        resolved = resolve_citations(self._DOC)
        idents = {(c.kind, c.identifier) for c in resolved}
        assert (CitationKind.DOI, "10.3847/2041-8213/ab50c5") in idents
        assert (CitationKind.ARXIV, "2411.04368") in idents

    def test_reference_section_excluded(self):
        # Only the two citing markers resolve; the bibliography lines are not
        # re-counted as their own citations.
        resolved = resolve_citations(self._DOC)
        assert len(resolved) == 2
        assert all(c.kind is not CitationKind.NUMERIC for c in resolved)

    def test_unresolved_numeric_dropped(self):
        # [9] has no reference entry → dropped, [1] resolves.
        doc = "Claim [1] and [9].\n\nReferences:\n[1] X https://example.org/x\n"
        resolved = resolve_citations(doc)
        assert [c.identifier for c in resolved] == ["https://example.org/x"]

    def test_inline_concrete_citation_passthrough(self):
        resolved = resolve_citations("Direct arXiv:2411.04368 reference, no list.")
        assert [(c.kind, c.identifier) for c in resolved] == [
            (CitationKind.ARXIV, "2411.04368")
        ]

    def test_empty_text(self):
        assert resolve_citations("") == []
        assert extract_inline_citations("") == []


def test_citation_is_frozen_dataclass():
    c = Citation("[1]", CitationKind.NUMERIC, "1", 0, 3)
    assert (c.raw, c.kind, c.identifier, c.start, c.end) == (
        "[1]",
        CitationKind.NUMERIC,
        "1",
        0,
        3,
    )
