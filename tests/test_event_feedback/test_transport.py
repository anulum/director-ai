# SPDX-License-Identifier: LicenseRef-BUSL-1.1-ANULUM-TDM
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Event feedback protocol tests
"""Exercise real loopback transport, destination validation and credential refusal."""

from __future__ import annotations

from pathlib import Path

import pytest

from director_ai.agentic.event_feedback.codec import FeedbackInvalidError
from director_ai.agentic.event_feedback.transport import (
    HttpEndpoint,
    HttpResponseInvalidError,
)

from .conftest import ProtocolControl, protocol_control


def test_body_timeout_retains_received_status_and_headers() -> None:
    """A real delayed body cannot erase the HTTP status already received."""
    control = ProtocolControl(body=b"later", body_delay=0.1)
    with protocol_control(control) as endpoint:
        endpoint.timeout = 0.05
        with pytest.raises(HttpResponseInvalidError) as refused:
            endpoint.exchange("POST", "/route", b"request")
    assert refused.value.response.status == 200
    assert refused.value.response.complete is False
    assert refused.value.response.raw == b""


def test_conflicting_transfer_framing_refuses_with_received_evidence() -> None:
    """A complete chunked body cannot clear contradictory framing headers."""
    control = ProtocolControl(
        body=b"5\r\nhello\r\n0\r\n\r\n", headers=[("Transfer-Encoding", "chunked")]
    )
    with (
        protocol_control(control) as endpoint,
        pytest.raises(HttpResponseInvalidError) as refused,
    ):
        endpoint.exchange("POST", "/route", b"request")
    assert refused.value.response.raw == b"hello"
    assert refused.value.response.complete is True


def test_failed_chunk_decoder_keeps_headers_without_invented_body() -> None:
    """A real invalid chunk parser yields explicit incomplete transport evidence."""
    control = ProtocolControl(
        body=b"invalid\r\n",
        headers=[("Transfer-Encoding", "chunked")],
        omit_content_length=True,
    )
    with (
        protocol_control(control) as endpoint,
        pytest.raises(HttpResponseInvalidError) as refused,
    ):
        endpoint.exchange("POST", "/route", b"request")
    assert refused.value.response.status == 200
    assert refused.value.response.complete is False


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "ftp://127.0.0.1",
        "http://",
        "http://user:key@localhost",
        "http://localhost/?query=x",
        "http://localhost/#fragment",
        "http://localhost:bad",
        "http://[broken",
    ],
)
def test_unsafe_destinations_refuse(url: str) -> None:
    """Refuse remote plaintext, userinfo and ambiguous endpoint URLs."""
    with pytest.raises(FeedbackInvalidError):
        HttpEndpoint(url)


def test_tls_and_nonpositive_timeout_validation() -> None:
    """Require a usable timeout while retaining verified HTTPS support."""
    endpoint = HttpEndpoint("https://example.com/prefix/")
    assert endpoint.secure and endpoint.prefix == "/prefix"
    with pytest.raises(FeedbackInvalidError):
        HttpEndpoint("http://localhost", timeout=0)


def test_private_credential_and_prefix_are_used_on_actual_socket(
    tmp_path: Path,
) -> None:
    """Transmit only the private operator credential to the selected destination."""
    token = tmp_path / "token"
    token.write_bytes(b"operator-token")
    token.chmod(0o600)
    control = ProtocolControl(
        body=b"original", status=302, headers=[("Location", "https://example.com")]
    )
    with protocol_control(control) as endpoint:
        endpoint.token_file = token
        endpoint.prefix = "/prefix"
        response = endpoint.exchange("POST", "/route", b"request")
    assert response.status == 302 and response.raw == b"original"
    assert control.requests == [
        ("POST", "/prefix/route", b"request", "Bearer operator-token")
    ]


@pytest.mark.parametrize(
    "route", ["relative", "//remote", "/bad\rheader", "/bad\nheader"]
)
def test_route_injection_refuses_without_connecting(route: str) -> None:
    """Refuse ambiguous or injectable routes before network IO."""
    with pytest.raises(FeedbackInvalidError):
        HttpEndpoint("http://localhost").exchange("GET", route)


@pytest.mark.parametrize("raw", [b"", b" token", b"token\n", b"token\rheader", b"\xff"])
def test_malformed_credentials_are_not_transmitted(tmp_path: Path, raw: bytes) -> None:
    """Reject newline, blank, Unicode or whitespace credential ambiguity."""
    token = tmp_path / "token"
    token.write_bytes(raw)
    token.chmod(0o600)
    endpoint = HttpEndpoint("http://localhost", token)
    with pytest.raises(FeedbackInvalidError):
        endpoint.exchange("GET", "/route")


def test_duplicate_header_response_is_ambiguous() -> None:
    """Reject two conflicting digest headers from an actual HTTP response."""
    control = ProtocolControl(
        headers=[("X-Content-SHA256", "a"), ("X-Content-SHA256", "b")]
    )
    with protocol_control(control) as endpoint, pytest.raises(FeedbackInvalidError):
        endpoint.exchange("GET", "/route")
