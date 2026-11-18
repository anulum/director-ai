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
import socket
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
    stream_body: bytes | None = None
    completion_body: bytes | None = None
    stream_status: int | None = None
    stream_content_type: str = "text/event-stream"
    completion_hold: tuple[threading.Event, threading.Event] | None = None

    def response_for_request(self, path: str, payload: dict[str, object]) -> bytes:
        """Return the JSON or SSE response body for a recorded request."""
        self.requests.append(_RecordedRequest(path=path, payload=payload))
        if payload.get("stream") is True:
            if self.stream_body is not None:
                return self.stream_body
            return b"".join(
                f"data: {json.dumps(event)}\n\n".encode()
                for event in self.stream_events
            )
        if self.completion_hold is not None:
            entered, release = self.completion_hold
            entered.set()
            release.wait(timeout=2.0)
        if self.completion_body is not None:
            return self.completion_body
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
            content_type = state.stream_content_type
            status = state.stream_status or state.response_status
        else:
            content_type = "application/json"
            status = state.response_status
        self.send_response(status)
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


def _closed_localhost_completion_url() -> str:
    """Reserve and release a localhost port so connection refusal is local."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        host, port = cast(tuple[str, int], sock.getsockname())
    return f"http://{host}:{port}/completion"


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


@pytest.mark.parametrize(
    ("body", "status", "content_type", "tokens", "requests"),
    [
        (
            b'data:{"content":"first"}\n\ndata: [DONE]\n\n',
            200,
            "text/event-stream",
            ["first"],
            1,
        ),
        (b"data: [DONE]\n\n", 200, "text/event-stream", [], 1),
        (
            b'data: {"stop":false}\n\ndata: {"token":"receipt"}\n\ndata: [DONE]\n\n',
            200,
            "text/event-stream",
            ["receipt"],
            1,
        ),
        (
            b": keep-alive\n\ndata:\n\n",
            200,
            "text/event-stream",
            ["fallback", "response"],
            2,
        ),
        (
            b'data: {"content":"unavailable"}\n\n',
            503,
            "text/event-stream",
            ["fallback", "response"],
            2,
        ),
        (
            b'{"content":"original response"}',
            200,
            "application/json",
            ["original", "response"],
            1,
        ),
        (b'{"content":""}', 200, "application/json", [], 1),
        (
            b'{"choices":[{"text":null}]}',
            200,
            "application/json",
            ["fallback", "response"],
            2,
        ),
        (b"null", 200, "application/json", ["fallback", "response"], 2),
    ],
)
def test_stream_terminators_and_available_fallbacks(
    body: bytes, status: int, content_type: str, tokens: list[str], requests: int
) -> None:
    """Terminate, reuse, or retry according to the actual upstream response."""
    with _ActorServer(({"content": "fallback response"},)) as server:
        server.state.stream_body = body
        server.state.stream_status = status
        server.state.stream_content_type = content_type
        generator = LLMGenerator(server.url, max_retries=1, timeout=5.0)
        assert _collect_stream(generator, "Stream the release receipt.") == tokens
        assert len(server.state.requests) == requests


@pytest.mark.parametrize(
    "failure",
    [
        b"data: incomplete\n\n",
        b'data: {"error":"interrupted"}\n\n',
        b'data: {"content":42}\n\n',
    ],
)
def test_stream_error_after_output_never_replays_another_generation(
    failure: bytes,
) -> None:
    """An interrupted stream keeps its prefix and reports its failure."""

    async def collect(generator: LLMGenerator) -> None:
        """Read the approved prefix and then observe the protocol error."""
        stream = generator.stream_tokens("Stream the release receipt.")
        assert await anext(stream) == "first"
        with pytest.raises(RuntimeError, match="after output started"):
            await anext(stream)

    with _ActorServer(({"content": "different response"},)) as server:
        server.state.stream_body = b'data: {"content":"first"}\n\n' + failure
        asyncio.run(collect(LLMGenerator(server.url, max_retries=1, timeout=5.0)))
        assert len(server.state.requests) == 1


def test_unavailable_stream_and_completion_surface_an_error() -> None:
    """An unavailable upstream cannot generate an ordinary text event."""
    with _ActorServer(({"error": "unavailable"},), response_status=503) as server:
        with pytest.raises(RuntimeError, match="completion unavailable"):
            _collect_stream(
                LLMGenerator(server.url, max_retries=1, timeout=5.0),
                "Stream the release receipt.",
            )
        assert len(server.state.requests) == 2


@pytest.mark.parametrize("completion", [b'{"error":"unavailable"}', b"incomplete"])
def test_missing_completion_text_cannot_look_like_an_empty_success(
    completion: bytes,
) -> None:
    """An unusable completion body fails the pre-output retry."""
    with _ActorServer(({"error": "unavailable"},)) as server:
        server.state.stream_body = b""
        server.state.completion_body = completion
        with pytest.raises(RuntimeError, match="completion unavailable"):
            _collect_stream(
                LLMGenerator(server.url, max_retries=1, timeout=5.0),
                "Stream the release receipt.",
            )
        assert len(server.state.requests) == 2


@pytest.mark.asyncio
async def test_actor_whole_completion_fallback_leaves_the_loop_available() -> None:
    """The replay request can stay in flight while the event loop runs."""
    entered = threading.Event()
    release = threading.Event()
    with _ActorServer(({"content": "fallback response"},)) as server:
        server.state.completion_hold = (entered, release)
        server.state.stream_status = 503
        generator = LLMGenerator(server.url, max_retries=1, timeout=5.0)

        async def collect() -> list[str]:
            """Drain the public token stream after the upstream is released."""
            return [
                token async for token in generator.stream_tokens("Release receipt.")
            ]

        task = asyncio.create_task(collect())
        try:
            assert await asyncio.to_thread(entered.wait, 5.0)
            assert not task.done()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
        assert task.result() == ["fallback", "response"]


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


def test_llm_generator_reports_real_connection_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Connection failures should return the public unavailable candidate."""
    generator = LLMGenerator(
        _closed_localhost_completion_url(),
        max_retries=1,
        base_delay=0.0,
        timeout=0.2,
    )

    with caplog.at_level("WARNING", logger="LLMGenerator"):
        candidates = generator.generate_candidates(
            "Summarise the closed endpoint.", n=1
        )

    assert candidates == [{"text": "[Error: LLM unavailable]", "source": "System"}]
    assert any(
        "LLM connection error" in record.getMessage() for record in caplog.records
    )


def test_public_package_lazy_exports_actor_generators() -> None:
    """The package root should expose the same actor generator classes."""
    assert director_ai.LLMGenerator is LLMGenerator
    assert director_ai.MockGenerator.__module__ == "director_ai.core.actor"
