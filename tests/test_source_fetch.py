# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — citation source fetcher tests
"""Offline tests for the citation source fetcher.

A stub HTTP getter returns the verified Crossref/arXiv response shapes, so every
branch — DOI→Crossref, arXiv→Atom, URL→doc_parser, the unfetchable author-year /
numeric kinds, non-200 and network-exception failures, empty-abstract and
parse-failure failures, the JATS/HTML markup stripping, the content-type
filename dispatch, the polite-pool User-Agent, and the fetch_all dedup/filter — is
exercised without touching the network.
"""

from __future__ import annotations

import ipaddress
import json
import socket

import pytest

from director_ai.core.citation_grounding import (
    Citation,
    CitationKind,
    FetchedSource,
    SourceFetcher,
)
from director_ai.core.citation_grounding.fetch import (
    _content_filename,
    _is_public_http_url,
    _parse_arxiv,
    _parse_crossref,
    _strip_markup,
)


@pytest.fixture(autouse=True)
def _hermetic_dns(monkeypatch):
    """Resolve hostnames to a fixed public IP and numeric hosts to themselves.

    Keeps the SSRF guard's getaddrinfo call off the real network: literal IPs
    (used by the SSRF tests) resolve to themselves, hostnames to a public IP.
    """

    def _fake(host, port, *args, **kwargs):
        try:
            ip = str(ipaddress.ip_address(host))
        except ValueError:
            ip = "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))]

    monkeypatch.setattr(socket, "getaddrinfo", _fake)


_CROSSREF = json.dumps(
    {
        "status": "ok",
        "message": {
            "title": ["PSR J0030 Mass and Radius"],
            "abstract": "<jats:title>Abstract</jats:title>"
            "<jats:p>Neutron stars constrain the equation of state.</jats:p>",
        },
    }
).encode()
_ARXIV = (
    b"<?xml version='1.0'?><feed><title>arXiv Query: id_list=2411.04368</title>"
    b"<entry><title>Measuring short-form factuality</title>"
    b"<summary>We present SimpleQA, a factuality benchmark.</summary></entry></feed>"
)
_HTML = b"<html><body><p>A web source about cheese.</p></body></html>"


class _StubHttp:
    def __init__(self, routes):
        self.routes = routes  # substring -> (status, body, content_type)
        self.headers_seen: list[dict] = []

    def get(self, url, *, headers=None):
        self.headers_seen.append(dict(headers or {}))
        for needle, resp in self.routes.items():
            if needle in url:
                return resp
        return 404, b"", ""


def _fetcher(routes, **kw):
    return SourceFetcher(http=_StubHttp(routes), **kw)


_OK_ROUTES = {
    "crossref": (200, _CROSSREF, "application/json"),
    "arxiv": (200, _ARXIV, "application/atom+xml"),
    "example.org": (200, _HTML, "text/html"),
}

_DOI = Citation("[1]", CitationKind.DOI, "10.3847/2041-8213/ab50c5", 0, 3)
_ARXIV_CITE = Citation("[2]", CitationKind.ARXIV, "2411.04368", 0, 3)
_URL_CITE = Citation("x", CitationKind.URL, "https://example.org/x", 0, 1)


class TestParsers:
    def test_strip_jats_and_entities(self):
        out = _strip_markup("<jats:p>A &amp; B   tag</jats:p>")
        assert out == "A & B tag"

    def test_parse_crossref(self):
        title, abstract = _parse_crossref(_CROSSREF)
        assert title == "PSR J0030 Mass and Radius"
        assert "equation of state" in abstract
        assert "<jats" not in abstract

    def test_parse_crossref_missing_fields(self):
        body = json.dumps({"message": {}}).encode()
        assert _parse_crossref(body) == ("", "")

    def test_parse_arxiv_reads_entry_not_feed_title(self):
        title, abstract = _parse_arxiv(_ARXIV)
        assert title == "Measuring short-form factuality"  # entry, not the query feed
        assert "SimpleQA" in abstract

    def test_parse_arxiv_no_entry(self):
        assert _parse_arxiv(b"<feed><title>empty</title></feed>") == ("", "")

    def test_content_filename_dispatch(self):
        assert _content_filename("application/pdf") == "source.pdf"
        assert _content_filename("text/html; charset=utf-8") == "source.html"
        assert _content_filename("text/plain") == "source.txt"


class TestFetch:
    def test_doi_success(self):
        result = _fetcher(_OK_ROUTES).fetch(_DOI)
        assert result.ok
        assert result.title == "PSR J0030 Mass and Radius"
        assert "equation of state" in result.text

    def test_arxiv_success(self):
        result = _fetcher(_OK_ROUTES).fetch(_ARXIV_CITE)
        assert result.ok
        assert "SimpleQA" in result.text

    def test_arxiv_http_error_is_failure(self):
        result = _fetcher({"arxiv": (503, b"", "")}).fetch(_ARXIV_CITE)
        assert not result.ok
        assert "arxiv HTTP 503" in result.error

    def test_arxiv_empty_abstract_is_failure(self):
        body = b"<feed><entry><title>Title only</title></entry></feed>"
        result = _fetcher({"arxiv": (200, body, "application/atom+xml")}).fetch(
            _ARXIV_CITE
        )
        assert not result.ok
        assert "arxiv abstract not found" in result.error

    def test_url_parsed_via_doc_parser(self):
        result = _fetcher(_OK_ROUTES).fetch(_URL_CITE)
        assert result.ok
        assert result.text == "A web source about cheese."

    def test_url_http_error_is_failure(self):
        result = _fetcher({"example.org": (403, b"", "")}).fetch(_URL_CITE)
        assert not result.ok
        assert "url HTTP 403" in result.error

    def test_url_parse_failure_is_failure(self, monkeypatch):
        from director_ai.core.retrieval import doc_parser

        def fail_parse(body, filename):
            raise ValueError(f"bad {filename}")

        monkeypatch.setattr(doc_parser, "parse", fail_parse)

        result = _fetcher(_OK_ROUTES).fetch(_URL_CITE)
        assert not result.ok
        assert "parse failed: bad source.html" in result.error

    def test_author_year_unfetchable(self):
        result = _fetcher(_OK_ROUTES).fetch(
            Citation("(Doe 2023)", CitationKind.AUTHOR_YEAR, "Doe 2023", 0, 1)
        )
        assert not result.ok
        assert "no retrievable source" in result.error

    def test_numeric_unfetchable(self):
        result = _fetcher(_OK_ROUTES).fetch(
            Citation("[9]", CitationKind.NUMERIC, "9", 0, 3)
        )
        assert not result.ok

    def test_http_error_is_failure(self):
        result = _fetcher({"crossref": (500, b"", "")}).fetch(_DOI)
        assert not result.ok
        assert "500" in result.error

    def test_network_exception_is_failure(self):
        class _Boom:
            def get(self, url, *, headers=None):
                raise OSError("connection refused")

        result = SourceFetcher(http=_Boom()).fetch(_DOI)
        assert not result.ok
        assert "HTTP 0" in result.error

    def test_empty_abstract_is_failure(self):
        body = json.dumps({"message": {"title": ["T"]}}).encode()
        result = _fetcher({"crossref": (200, body, "application/json")}).fetch(_DOI)
        assert not result.ok
        assert "abstract not found" in result.error

    def test_empty_document_is_failure(self):
        result = _fetcher({"example.org": (200, b"<html></html>", "text/html")}).fetch(
            _URL_CITE
        )
        assert not result.ok


class TestUserAgentAndBatch:
    def test_default_getter_keeps_timeout_without_network(self):
        fetcher = SourceFetcher(timeout=2.5)

        assert fetcher._http._timeout == 2.5

    def test_mailto_in_user_agent(self):
        http = _StubHttp(_OK_ROUTES)
        SourceFetcher(http=http, mailto="x@anulum.li").fetch(_DOI)
        assert "mailto:x@anulum.li" in http.headers_seen[0]["User-Agent"]

    def test_no_mailto_plain_user_agent(self):
        http = _StubHttp(_OK_ROUTES)
        SourceFetcher(http=http).fetch(_DOI)
        ua = http.headers_seen[0]["User-Agent"]
        assert ua.startswith("director-ai-citation-grounding/1.0")
        assert "mailto" not in ua

    def test_fetch_all_returns_identifier_map(self):
        sources = _fetcher(_OK_ROUTES).fetch_all([_DOI, _ARXIV_CITE])
        assert set(sources) == {"10.3847/2041-8213/ab50c5", "2411.04368"}
        assert "equation of state" in sources["10.3847/2041-8213/ab50c5"]

    def test_fetch_all_dedups_and_filters_failures(self):
        http = _StubHttp(_OK_ROUTES)
        fetcher = SourceFetcher(http=http)
        # _DOI twice (dedup) plus an unfetchable author-year (filtered out).
        sources = fetcher.fetch_all(
            [_DOI, _DOI, Citation("(D 2023)", CitationKind.AUTHOR_YEAR, "D 2023", 0, 1)]
        )
        assert list(sources) == ["10.3847/2041-8213/ab50c5"]
        # the duplicate DOI was fetched once; the author-year never hits HTTP
        assert len(http.headers_seen) == 1


def test_fetched_source_dataclass():
    src = FetchedSource("10.1/x", CitationKind.DOI, True, "T", "body", "http://u")
    assert (src.identifier, src.ok, src.title, src.text) == (
        "10.1/x",
        True,
        "T",
        "body",
    )


class TestUrlSsrfGuard:
    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/",  # cloud metadata
            "http://127.0.0.1/admin",  # loopback
            "http://[::1]/",  # loopback v6
            "http://10.0.0.5/internal",  # private
            "http://192.168.1.1/",  # private
            "http://172.16.0.1/",  # private
            "ftp://example.org/x",  # non-http scheme
            "file:///etc/passwd",  # file scheme
            "http://0.0.0.0/",  # unspecified
            "not-a-url",  # no host
        ],
    )
    def test_blocks_non_public_urls(self, url):
        assert _is_public_http_url(url) is False

    def test_allows_public_url(self):
        assert _is_public_http_url("https://93.184.216.34/x") is True
        assert _is_public_http_url("https://example.org/x") is True

    def test_fetch_url_refuses_internal_target_without_calling_http(self):
        # An LLM-emitted citation pointing at internal infrastructure must be
        # refused before any request is made (SSRF guard).
        http = _StubHttp(
            {"169.254.169.254": (200, b"<html>secret</html>", "text/html")}
        )
        fetcher = SourceFetcher(http=http)
        cite = Citation("x", CitationKind.URL, "http://169.254.169.254/", 0, 1)
        result = fetcher.fetch(cite)
        assert result.ok is False
        assert "SSRF" in (result.error or "")
        assert http.headers_seen == []  # never dialled out
