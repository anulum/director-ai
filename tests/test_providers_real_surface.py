# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - provider adapter real-surface tests
"""Real localhost protocol coverage for provider adapters."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Literal, cast

from director_ai.integrations.providers import LocalProvider, OpenAIProvider


@dataclass(frozen=True)
class _CapturedRequest:
    """HTTP request captured by the local provider protocol server."""

    path: str
    headers: dict[str, str]
    payload: dict[str, object]


class _ProviderHTTPServer(ThreadingHTTPServer):
    """Threaded localhost server that records provider adapter requests."""

    requests: list[_CapturedRequest]
    mode: Literal["json", "sse"]

    def __init__(
        self,
        server_address: tuple[str, int],
        mode: Literal["json", "sse"],
    ) -> None:
        super().__init__(server_address, _ProviderHandler)
        self.requests = []
        self.mode = mode


class _ProviderHandler(BaseHTTPRequestHandler):
    """Serve OpenAI-compatible JSON and SSE responses for adapter tests."""

    def do_POST(self) -> None:
        """Record the request and return a protocol-compatible response."""
        server = cast(_ProviderHTTPServer, self.server)
        payload = self._read_json_payload()
        server.requests.append(
            _CapturedRequest(
                path=self.path,
                headers=dict(self.headers.items()),
                payload=payload,
            )
        )

        if server.mode == "sse":
            self._send_sse_response()
            return
        self._send_json_response(payload)

    def log_message(self, format: str, *args: object) -> None:
        """Suppress access logs during tests."""

    def _read_json_payload(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        raw_payload = json.loads(body.decode("utf-8")) if body else {}
        if not isinstance(raw_payload, dict):
            return {}
        return cast(dict[str, object], raw_payload)

    def _send_json_response(self, payload: dict[str, object]) -> None:
        requested_count = payload.get("n", 1)
        count = requested_count if isinstance(requested_count, int) else 1
        response = {
            "choices": [
                {"message": {"content": f"candidate-{index}"}}
                for index in range(count)
            ]
        }
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse_response(self) -> None:
        body = b"".join(
            [
                b'data: {"choices":[{"delta":{"content":"guarded "}}]}\n\n',
                b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n',
                b"data: [DONE]\n\n",
            ]
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def _provider_server(mode: Literal["json", "sse"]) -> Iterator[_ProviderHTTPServer]:
    server = _ProviderHTTPServer(("127.0.0.1", 0), mode)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_openai_provider_uses_real_chat_completions_endpoint() -> None:
    """OpenAI-compatible provider should POST the real chat-completions shape."""
    with _provider_server("json") as server:
        provider = OpenAIProvider(
            api_key="local-token",
            model="local-chat-model",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            temperature=0.2,
            max_tokens=17,
            timeout=5,
        )

        candidates = provider.generate_candidates("Check this answer.", n=2)

    assert candidates == [
        {"text": "candidate-0", "source": "openai/local-chat-model"},
        {"text": "candidate-1", "source": "openai/local-chat-model"},
    ]
    request = server.requests[0]
    assert request.path == "/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer local-token"
    assert request.payload["model"] == "local-chat-model"
    assert request.payload["temperature"] == 0.2
    assert request.payload["max_tokens"] == 17
    assert request.payload["n"] == 2
    assert request.payload["messages"] == [
        {"role": "user", "content": "Check this answer."}
    ]


def test_local_provider_streams_real_sse_tokens_from_endpoint() -> None:
    """Local provider should parse streamed SSE chunks from a real endpoint."""
    with _provider_server("sse") as server:
        provider = LocalProvider(
            api_url=f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            model="local-stream-model",
            temperature=0.1,
            max_tokens=23,
            timeout=5,
        )

        tokens = list(provider.stream_generate("Stream this answer."))

    assert tokens == ["guarded ", "answer"]
    request = server.requests[0]
    assert request.path == "/v1/chat/completions"
    assert request.payload["model"] == "local-stream-model"
    assert request.payload["stream"] is True
    assert request.payload["temperature"] == 0.1
    assert request.payload["max_tokens"] == 23
    assert request.payload["messages"] == [
        {"role": "user", "content": "Stream this answer."}
    ]
