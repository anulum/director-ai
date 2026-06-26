# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - recall-correctness HTTP real-surface tests
"""Real HTTP protocol coverage for the recall-correctness client."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from director_ai.core.calibration.recall_correctness import RecallOutcome
from director_ai.core.calibration.recall_correctness_client import (
    RemanentiaCorrectnessClient,
    RemanentiaCorrectnessError,
)


@dataclass(frozen=True)
class _RecordedRequest:
    path: str
    headers: dict[str, str]
    body: bytes


@dataclass(frozen=True)
class _ServerState:
    base_url: str
    requests: list[_RecordedRequest]


@contextmanager
def _serve_correctness_response(
    *,
    status: int,
    body: bytes,
) -> Iterator[_ServerState]:
    requests: list[_RecordedRequest] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            request_body = self.rfile.read(length)
            requests.append(
                _RecordedRequest(
                    path=self.path,
                    headers={key: value for key, value in self.headers.items()},
                    body=request_body,
                )
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield _ServerState(
            base_url=f"http://127.0.0.1:{server.server_port}",
            requests=requests,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _outcome() -> RecallOutcome:
    return RecallOutcome(
        query="What was retrieved?",
        was_correct=True,
        by="director-ai",
    )


def test_recall_correctness_client_posts_to_real_http_endpoint() -> None:
    """The client should POST the production JSON payload over real HTTP."""
    with _serve_correctness_response(
        status=201,
        body=b'{"event_id": "evt-real"}',
    ) as server:
        client = RemanentiaCorrectnessClient(
            f"{server.base_url}/memory",
            token="token-1",
            timeout_s=2.0,
        )

        assert client.record(_outcome()) == "evt-real"

    assert len(server.requests) == 1
    request = server.requests[0]
    assert request.path == "/memory/recall/correctness"
    assert request.headers["Authorization"] == "Bearer token-1"
    assert request.headers["Accept"] == "application/json"
    assert request.headers["Content-Type"] == "application/json"
    assert json.loads(request.body) == {
        "query": "What was retrieved?",
        "was_correct": True,
        "by": "director-ai",
    }


def test_recall_correctness_client_treats_real_404_as_no_prior_recall() -> None:
    """The client should map REMANENTIA 404 responses to ``None``."""
    with _serve_correctness_response(
        status=404,
        body=b'{"detail": "no prior recall"}',
    ) as server:
        client = RemanentiaCorrectnessClient(server.base_url, token="token-1")

        assert client.record(_outcome()) is None

    assert server.requests[0].path == "/recall/correctness"


def test_recall_correctness_client_raises_real_error_detail() -> None:
    """The client should surface non-2xx REMANENTIA detail messages."""
    with _serve_correctness_response(
        status=409,
        body=b'{"detail": "ledger locked"}',
    ) as server:
        client = RemanentiaCorrectnessClient(server.base_url, token="token-1")

        with pytest.raises(RemanentiaCorrectnessError, match="HTTP 409: ledger locked"):
            client.record(_outcome())


def test_recall_correctness_client_rejects_embedded_url_credentials() -> None:
    """The client should reject credentials embedded in the endpoint URL."""
    with pytest.raises(ValueError, match="must not include credentials"):
        RemanentiaCorrectnessClient("http://user:pass@127.0.0.1:8001")
