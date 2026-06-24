# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the REMANENTIA recall-correctness HTTP client.

Covers configuration guards, the success path (payload, bearer header, path,
returned event_id), the 404 no-prior-recall passthrough that yields None, the
hard-failure modes (non-2xx with and without a detail, 422, missing event_id,
empty body, invalid JSON, transport error), bearer-header presence/absence by
token, the HTTPS connection class, and the hot-path-safe try_record that swallows
both transport and protocol failures.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from director_ai.core.calibration.recall_correctness import RecallOutcome
from director_ai.core.calibration.recall_correctness_client import (
    RemanentiaCorrectnessClient,
    RemanentiaCorrectnessError,
)


class _Response:
    def __init__(self, status: int = 200, body: bytes = b'{"event_id": "e1"}') -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body


class _Connection:
    instances: list[_Connection] = []
    response: _Response | None = None
    request_error: BaseException | None = None

    def __init__(
        self, host: str, *, port: int | None = None, timeout: float | None = None
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.requests: list[dict[str, object]] = []
        self.closed = False
        type(self).instances.append(self)

    def request(
        self, method: str, path: str, *, body: bytes = b"", headers: dict | None = None
    ) -> None:
        if type(self).request_error is not None:
            raise type(self).request_error
        self.requests.append(
            {
                "method": method,
                "path": path,
                "body": body,
                "headers": dict(headers or {}),
            }
        )

    def getresponse(self) -> _Response:
        assert type(self).response is not None
        return type(self).response

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_connection() -> None:
    _Connection.instances = []
    _Connection.response = _Response()
    _Connection.request_error = None


def _outcome(was_correct: bool = True) -> RecallOutcome:
    return RecallOutcome(query="capital of France?", was_correct=was_correct, by="t")


# --- configuration guards ---------------------------------------------------


def test_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="timeout_s must be > 0"):
        RemanentiaCorrectnessClient(timeout_s=0)
    with pytest.raises(ValueError, match="scheme must be http or https"):
        RemanentiaCorrectnessClient("ftp://127.0.0.1:8001")
    with pytest.raises(ValueError, match="include a host"):
        RemanentiaCorrectnessClient("http:///missing-host")
    with pytest.raises(ValueError, match="params, query, or fragment"):
        RemanentiaCorrectnessClient("http://127.0.0.1:8001/x?debug=1")


# --- success path -----------------------------------------------------------


def test_record_posts_payload_and_returns_event_id() -> None:
    _Connection.response = _Response(
        200, b'{"event_id": "evt-42", "was_correct": true}'
    )
    with patch("http.client.HTTPConnection", _Connection):
        client = RemanentiaCorrectnessClient("http://127.0.0.1:8001/api", token="tok")
        event_id = client.record(_outcome(was_correct=False))

    assert event_id == "evt-42"
    request = _Connection.instances[0].requests[0]
    assert request["method"] == "POST"
    assert request["path"] == "/api/recall/correctness"
    assert json.loads(request["body"]) == {
        "query": "capital of France?",
        "was_correct": False,
        "by": "t",
    }
    headers = request["headers"]
    assert headers["Authorization"] == "Bearer tok"
    assert headers["Content-Type"] == "application/json"
    assert _Connection.instances[0].closed is True


def test_token_absent_sends_no_authorization_header() -> None:
    with patch("http.client.HTTPConnection", _Connection):
        client = RemanentiaCorrectnessClient("http://127.0.0.1:8001")
        client.record(_outcome())
    assert "Authorization" not in _Connection.instances[0].requests[0]["headers"]


def test_https_uses_https_connection() -> None:
    with patch("http.client.HTTPSConnection", _Connection):
        client = RemanentiaCorrectnessClient("https://memory.example:8443", token="s")
        assert client.record(_outcome()) == "e1"
    assert _Connection.instances[0].host == "memory.example"
    assert _Connection.instances[0].port == 8443


# --- 404 no-prior-recall ----------------------------------------------------


def test_record_404_returns_none() -> None:
    _Connection.response = _Response(404, b'{"detail": "no prior recall for: q"}')
    with patch("http.client.HTTPConnection", _Connection):
        client = RemanentiaCorrectnessClient(token="t")
        assert client.record(_outcome()) is None


# --- hard failures ----------------------------------------------------------


def test_record_non_2xx_raises_with_detail() -> None:
    _Connection.response = _Response(500, b'{"detail": "boom"}')
    with patch("http.client.HTTPConnection", _Connection):
        client = RemanentiaCorrectnessClient(token="t")
        with pytest.raises(RemanentiaCorrectnessError, match="HTTP 500: boom"):
            client.record(_outcome())


def test_record_422_raises() -> None:
    _Connection.response = _Response(422, b'{"detail": "field required"}')
    with patch("http.client.HTTPConnection", _Connection):
        client = RemanentiaCorrectnessClient(token="t")
        with pytest.raises(RemanentiaCorrectnessError, match="HTTP 422"):
            client.record(_outcome())


def test_record_non_2xx_without_detail_raises_plain() -> None:
    _Connection.response = _Response(503, b"")
    with patch("http.client.HTTPConnection", _Connection):
        client = RemanentiaCorrectnessClient(token="t")
        with pytest.raises(RemanentiaCorrectnessError, match=r"HTTP 503$"):
            client.record(_outcome())


def test_record_missing_event_id_raises() -> None:
    _Connection.response = _Response(200, b'{"was_correct": true}')
    with patch("http.client.HTTPConnection", _Connection):
        client = RemanentiaCorrectnessClient(token="t")
        with pytest.raises(RemanentiaCorrectnessError, match="missing event_id"):
            client.record(_outcome())


def test_record_invalid_json_raises() -> None:
    _Connection.response = _Response(200, b"not json")
    with patch("http.client.HTTPConnection", _Connection):
        client = RemanentiaCorrectnessClient(token="t")
        with pytest.raises(RemanentiaCorrectnessError, match="not valid JSON"):
            client.record(_outcome())


def test_record_transport_error_raises() -> None:
    _Connection.request_error = OSError("connection refused")
    with patch("http.client.HTTPConnection", _Connection):
        client = RemanentiaCorrectnessClient(token="t")
        with pytest.raises(RemanentiaCorrectnessError, match="request failed"):
            client.record(_outcome())
    assert _Connection.instances[0].closed is True


def test_record_non_object_json_treated_as_missing_event_id() -> None:
    _Connection.response = _Response(200, b"[1, 2, 3]")
    with patch("http.client.HTTPConnection", _Connection):
        client = RemanentiaCorrectnessClient(token="t")
        with pytest.raises(RemanentiaCorrectnessError, match="missing event_id"):
            client.record(_outcome())


# --- try_record hot-path safety ---------------------------------------------


def test_try_record_returns_event_id_on_success() -> None:
    _Connection.response = _Response(200, b'{"event_id": "ok"}')
    with patch("http.client.HTTPConnection", _Connection):
        client = RemanentiaCorrectnessClient(token="t")
        assert client.try_record(_outcome()) == "ok"


def test_try_record_swallows_protocol_error() -> None:
    _Connection.response = _Response(500, b'{"detail": "x"}')
    with patch("http.client.HTTPConnection", _Connection):
        client = RemanentiaCorrectnessClient(token="t")
        assert client.try_record(_outcome()) is None


def test_try_record_swallows_transport_error() -> None:
    _Connection.request_error = OSError("down")
    with patch("http.client.HTTPConnection", _Connection):
        client = RemanentiaCorrectnessClient(token="t")
        assert client.try_record(_outcome()) is None


def test_try_record_returns_none_on_404() -> None:
    _Connection.response = _Response(404, b'{"detail": "none"}')
    with patch("http.client.HTTPConnection", _Connection):
        client = RemanentiaCorrectnessClient(token="t")
        assert client.try_record(_outcome()) is None
