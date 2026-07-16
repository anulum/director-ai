# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Consumer API real-surface tests
"""Real public consumer API coverage for package exports and LLM endpoints."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Self, cast

import director_ai


@dataclass(frozen=True, slots=True)
class _RecordedRequest:
    """Request captured by the local completion protocol server."""

    path: str
    payload: dict[str, object]


@dataclass(slots=True)
class _CompletionState:
    """Mutable response and request log for the local completion server."""

    response_payload: dict[str, object]
    status_code: int = 200
    requests: list[_RecordedRequest] = field(default_factory=list)


class _CompletionHTTPServer(ThreadingHTTPServer):
    """Threading HTTP server carrying typed completion test state."""

    state: _CompletionState

    def __init__(self, state: _CompletionState) -> None:
        super().__init__(("127.0.0.1", 0), _CompletionHandler)
        self.state = state


class _CompletionHandler(BaseHTTPRequestHandler):
    """Serve a minimal OpenAI-compatible completion endpoint."""

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        """Record the completion request and return the configured payload."""
        state = cast(_CompletionHTTPServer, self.server).state
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(length)
        payload = json.loads(raw_body.decode("utf-8"))
        if not isinstance(payload, dict):
            self.send_error(400, "JSON object required")
            return

        state.requests.append(
            _RecordedRequest(
                path=self.path,
                payload=cast(dict[str, object], payload),
            ),
        )
        response_body = json.dumps(state.response_payload).encode("utf-8")
        self.send_response(state.status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format: str, *args: object) -> None:
        """Suppress per-request stderr logging from the test server."""


class _CompletionServer:
    """Context manager for a localhost completion protocol server."""

    def __init__(
        self,
        response_payload: dict[str, object],
        *,
        status_code: int = 200,
    ) -> None:
        self.state = _CompletionState(
            response_payload=response_payload,
            status_code=status_code,
        )
        self._server = _CompletionHTTPServer(self.state)
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
        """Stop the completion server and wait for the thread to exit."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def test_public_consumer_exports_import_in_clean_subprocess() -> None:
    """The installed package should expose the documented consumer API."""
    script = """
import json
import director_ai

exports = [
    "CoherenceAgent",
    "CoherenceScorer",
    "LLMGenerator",
    "GroundTruthStore",
    "CoherenceScore",
    "ReviewResult",
    "SafetyKernel",
]

print(json.dumps({
    "version": director_ai.__version__,
    "exports": {name: hasattr(director_ai, name) for name in exports},
}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload == {
        "version": "3.18.0",
        "exports": {
            "CoherenceAgent": True,
            "CoherenceScorer": True,
            "LLMGenerator": True,
            "GroundTruthStore": True,
            "CoherenceScore": True,
            "ReviewResult": True,
            "SafetyKernel": True,
        },
    }


def test_public_llm_generator_uses_real_completion_protocol() -> None:
    """LLMGenerator should POST completion payloads to a real HTTP endpoint."""
    with _CompletionServer({"content": "The sky is blue."}) as server:
        generator = director_ai.LLMGenerator(
            api_url=server.url,
            max_retries=1,
            base_delay=0,
            timeout=5,
            max_tokens=48,
            temperature=0.2,
            stop_sequences=("\\nUser:",),
        )

        candidates = generator.generate_candidates("What colour is the sky?", n=2)

    assert candidates == [
        {"text": "The sky is blue.", "source": "LLM"},
        {"text": "The sky is blue.", "source": "LLM"},
    ]
    assert [request.path for request in server.state.requests] == [
        "/completion",
        "/completion",
    ]
    assert server.state.requests[0].payload == {
        "prompt": "What colour is the sky?",
        "n_predict": 48,
        "temperature": 0.2,
        "stop": ["\\nUser:"],
    }


def test_public_llm_generator_surfaces_real_http_failure() -> None:
    """LLMGenerator should return the documented error candidate on HTTP 500."""
    with _CompletionServer({"error": "unavailable"}, status_code=500) as server:
        generator = director_ai.LLMGenerator(
            api_url=server.url,
            max_retries=1,
            base_delay=0,
            timeout=5,
        )

        candidates = generator.generate_candidates("What colour is the sky?", n=1)

    assert candidates == [{"text": "[Error: LLM unavailable]", "source": "System"}]
    assert server.state.requests[0].payload["prompt"] == "What colour is the sky?"


def test_public_agent_processes_real_http_completion() -> None:
    """CoherenceAgent should generate and score through a real HTTP endpoint."""
    with _CompletionServer({"content": "The sky is blue."}) as server:
        agent = director_ai.CoherenceAgent(
            llm_api_url=server.url,
            use_nli=False,
            max_candidates=1,
            llm_max_tokens=32,
            llm_temperature=0.0,
        )
        agent.store.add("sky", "The sky is blue.")

        result = agent.process("What colour is the sky?")

    assert isinstance(result, director_ai.ReviewResult)
    assert result.output == "The sky is blue."
    assert result.coherence is not None
    assert result.candidates_evaluated == 1
    assert server.state.requests[0].payload["prompt"] == "What colour is the sky?"
