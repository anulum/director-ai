# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Actor real-surface tests
"""Real HTTP/SSE coverage for the actor generator boundary."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Self, cast

import pytest

import director_ai
from director_ai.core.actor import LLMGenerator


@dataclass(frozen=True, slots=True)
class _RecordedRequest:
    """Request captured by the local actor protocol server."""

    path: str
    payload: dict[str, object]


@dataclass(slots=True)
class _ActorServerState:
    """Mutable response and request log for the local actor server."""

    completion_payloads: tuple[dict[str, object], ...]
    response_status: int = 200
    stream_events: tuple[dict[str, str], ...] = ()
    requests: list[_RecordedRequest] = field(default_factory=list)

    def response_for_request(self, path: str, payload: dict[str, object]) -> bytes:
        """Return the JSON or SSE response body for a recorded request."""
        self.requests.append(_RecordedRequest(path=path, payload=payload))
        if payload.get("stream") is True:
            return b"".join(
                f"data: {json.dumps(event)}\n\n".encode()
                for event in self.stream_events
            )
        index = min(
            sum(
                1
                for request in self.requests
                if request.payload.get("stream") is not True
            )
            - 1,
            len(self.completion_payloads) - 1,
        )
        return json.dumps(self.completion_payloads[index]).encode("utf-8")


class _ActorHTTPServer(ThreadingHTTPServer):
    """Threading HTTP server carrying typed actor test state."""

    state: _ActorServerState

    def __init__(self, state: _ActorServerState) -> None:
        super().__init__(("127.0.0.1", 0), _ActorHandler)
        self.state = state


class _ActorHandler(BaseHTTPRequestHandler):
    """Serve a minimal completion endpoint compatible with ``LLMGenerator``."""

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        """Record the POST payload and emit completion JSON or SSE frames."""
        state = cast(_ActorHTTPServer, self.server).state
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(length)
        payload = json.loads(raw_body.decode("utf-8"))
        if not isinstance(payload, dict):
            self.send_error(400, "JSON object required")
            return

        body = state.response_for_request(self.path, cast(dict[str, object], payload))
        if payload.get("stream") is True:
            content_type = "text/event-stream"
        else:
            content_type = "application/json"
        self.send_response(state.response_status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Suppress per-request stderr logging from the test server."""


class _ActorServer:
    """Context manager for a localhost actor protocol server."""

    def __init__(
        self,
        completion_payloads: tuple[dict[str, object], ...],
        *,
        response_status: int = 200,
        stream_events: tuple[dict[str, str], ...] = (),
    ) -> None:
        self.state = _ActorServerState(
            completion_payloads=completion_payloads,
            response_status=response_status,
            stream_events=stream_events,
        )
        self._server = _ActorHTTPServer(self.state)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        """Return the bound completion URL."""
        host, port = cast(tuple[str, int], self._server.socket.getsockname())
        return f"http://{host}:{port}/completion"

    def __enter__(self) -> Self:
        """Start serving requests in a background thread."""
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop the actor server and wait for the thread to exit."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _collect_stream(generator: LLMGenerator, prompt: str) -> list[str]:
    """Collect all tokens emitted by the async stream API."""

    async def collect() -> list[str]:
        tokens: list[str] = []
        stream: AsyncIterator[str] = generator.stream_tokens(prompt)
        async for token in stream:
            tokens.append(token)
        return tokens

    return asyncio.run(collect())


def test_llm_generator_uses_real_completion_http_contract() -> None:
    """LLMGenerator should post production payloads to a real HTTP endpoint."""
    payloads: tuple[dict[str, object], ...] = (
        {"content": "The release is blocked until evidence is signed."},
        {"choices": [{"text": "The second candidate uses choices text."}]},
    )
    with _ActorServer(payloads) as server:
        generator = LLMGenerator(
            server.url,
            max_retries=1,
            base_delay=0.0,
            timeout=5.0,
            max_tokens=64,
            temperature=0.25,
            stop_sequences=("END",),
        )

        candidates = generator.generate_candidates("Summarise the release gate.", n=2)

    assert candidates == [
        {
            "text": "The release is blocked until evidence is signed.",
            "source": "LLM",
        },
        {"text": "The second candidate uses choices text.", "source": "LLM"},
    ]
    assert [request.path for request in server.state.requests] == [
        "/completion",
        "/completion",
    ]
    assert server.state.requests[0].payload == {
        "prompt": "Summarise the release gate.",
        "n_predict": 64,
        "temperature": 0.25,
        "stop": ["END"],
    }


def test_llm_generator_streams_tokens_from_real_sse_endpoint() -> None:
    """The async stream API should consume real SSE completion frames."""
    with _ActorServer(
        ({"content": "unused fallback"},),
        stream_events=({"content": "verified"}, {"token": "receipt"}),
    ) as server:
        generator = LLMGenerator(server.url, max_retries=1, base_delay=0.0, timeout=5.0)

        tokens = _collect_stream(generator, "Stream the release receipt.")

    assert tokens == ["verified", "receipt"]
    assert server.state.requests == [
        _RecordedRequest(
            path="/completion",
            payload={
                "prompt": "Stream the release receipt.",
                "n_predict": 128,
                "temperature": 0.8,
                "stop": ["\nUser:", "\nSystem:"],
                "stream": True,
            },
        )
    ]


def test_llm_generator_logs_real_http_error_with_truncated_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """LLMGenerator should bound logged response text from real HTTP errors."""
    error_tail = "unlogged-tail-marker"
    error_payload: tuple[dict[str, object], ...] = (
        {"error": ("X" * 600) + error_tail},
    )
    with _ActorServer(error_payload, response_status=500) as server:
        generator = LLMGenerator(server.url, max_retries=1, base_delay=0.0, timeout=5.0)

        with caplog.at_level("ERROR", logger="LLMGenerator"):
            candidates = generator.generate_candidates(
                "Summarise the failed call.",
                n=1,
            )

    assert candidates == [{"text": "[Error: LLM unavailable]", "source": "System"}]
    assert len(server.state.requests) == 1
    error_messages = [record.getMessage() for record in caplog.records]
    assert any(message.startswith("LLM Error 500: ") for message in error_messages)
    assert all(error_tail not in message for message in error_messages)


def test_public_package_lazy_exports_actor_generators() -> None:
    """The package root should expose the same actor generator classes."""
    assert director_ai.LLMGenerator is LLMGenerator
    assert director_ai.MockGenerator.__module__ == "director_ai.core.actor"
