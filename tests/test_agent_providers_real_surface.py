# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - agent provider real-surface tests
"""Real provider-protocol coverage for ``CoherenceAgent`` routing."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Self, cast

from director_ai.core.agent import CoherenceAgent
from director_ai.core.types import ReviewResult
from director_ai.integrations.providers import OpenAIProvider


@dataclass(frozen=True, slots=True)
class _RecordedChatRequest:
    """OpenAI-compatible chat request captured by the local server."""

    path: str
    headers: dict[str, str]
    payload: dict[str, object]


@dataclass(slots=True)
class _ChatServerState:
    """Mutable response and request log for the local provider server."""

    replies: tuple[str, ...]
    requests: list[_RecordedChatRequest] = field(default_factory=list)

    def response_body(self, path: str, headers: dict[str, str], body: bytes) -> bytes:
        """Record one request and return a chat-completions JSON response."""
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            payload = {}
        self.requests.append(
            _RecordedChatRequest(
                path=path,
                headers=headers,
                payload=cast(dict[str, object], payload),
            )
        )
        requested = payload.get("n", 1)
        count = requested if isinstance(requested, int) and requested > 0 else 1
        choices = [
            {"message": {"content": self.replies[index % len(self.replies)]}}
            for index in range(count)
        ]
        return json.dumps({"choices": choices}).encode("utf-8")


class _ChatHTTPServer(ThreadingHTTPServer):
    """Threading server carrying typed provider test state."""

    state: _ChatServerState

    def __init__(self, state: _ChatServerState) -> None:
        super().__init__(("127.0.0.1", 0), _ChatHandler)
        self.state = state


class _ChatHandler(BaseHTTPRequestHandler):
    """Serve a minimal OpenAI-compatible chat-completions endpoint."""

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        """Record the POST payload and emit chat-completions JSON."""
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = cast(_ChatHTTPServer, self.server).state.response_body(
            self.path,
            dict(self.headers.items()),
            self.rfile.read(length),
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Suppress access logs during tests."""


class _ChatServer:
    """Context manager for a localhost provider protocol server."""

    def __init__(self, replies: tuple[str, ...]) -> None:
        self.state = _ChatServerState(replies=replies)
        self._server = _ChatHTTPServer(self.state)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        """Return the OpenAI-compatible ``/v1`` base URL."""
        host, port = cast(tuple[str, int], self._server.socket.getsockname())
        return f"http://{host}:{port}/v1"

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
        """Stop the server and wait for the thread to exit."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def test_agent_openai_provider_process_uses_real_chat_completions_protocol() -> None:
    """Provider-backed agents should drive a real chat-completions endpoint."""
    env = {key: value for key, value in os.environ.items() if key != "OPENAI_API_KEY"}
    replies = (
        "The verification receipt was signed.",
        "The verification receipt was countersigned.",
    )

    with _ChatServer(replies) as server, _PatchedEnviron(env):
        agent = CoherenceAgent(
            provider="openai",
            api_key="local-provider-token",
            use_nli=False,
            production_mode=True,
            max_candidates=2,
        )
        provider = cast(OpenAIProvider, agent.generator)
        provider.base_url = server.base_url
        provider.timeout = 5
        agent.scorer.threshold = 0.0

        result = agent.process("Was the verification receipt signed?")
        assert "OPENAI_API_KEY" not in os.environ

    assert len(server.state.requests) == 1
    request = server.state.requests[0]
    assert isinstance(result, ReviewResult)
    assert result.candidates_evaluated == 2
    assert request.path == "/v1/chat/completions"
    assert request.payload == {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": "Was the verification receipt signed?",
            }
        ],
        "n": 2,
        "temperature": 0.8,
        "max_tokens": 512,
    }
    assert request.headers["Authorization"] == "Bearer local-provider-token"


class _PatchedEnviron:
    """Temporary process environment replacement for provider leakage checks."""

    def __init__(self, values: dict[str, str]) -> None:
        self._values = values
        self._original: dict[str, str] = {}

    def __enter__(self) -> None:
        """Replace ``os.environ`` with the provided mapping."""
        self._original = dict(os.environ)
        os.environ.clear()
        os.environ.update(self._values)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore the original process environment."""
        os.environ.clear()
        os.environ.update(self._original)
