# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - agent real-surface tests
"""Real completion-endpoint coverage for ``CoherenceAgent`` orchestration."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Self, cast

from director_ai.core.agent import CoherenceAgent
from director_ai.core.runtime.batch import BatchProcessor
from director_ai.core.types import ReviewResult
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


@dataclass(frozen=True, slots=True)
class _RecordedCompletionRequest:
    """Completion request captured by the local LLM endpoint."""

    path: str
    payload: dict[str, object]


@dataclass(slots=True)
class _CompletionServerState:
    """Mutable response plan and request log for the completion endpoint."""

    replies: tuple[str, ...]
    requests: list[_RecordedCompletionRequest] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def response_body(self, path: str, body: bytes) -> bytes:
        """Record one request and return an OpenAI-compatible JSON body."""
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            payload = {}
        with self._lock:
            index = min(len(self.requests), len(self.replies) - 1)
            self.requests.append(
                _RecordedCompletionRequest(
                    path=path,
                    payload=cast(dict[str, object], payload),
                )
            )
        return json.dumps({"content": self.replies[index]}).encode("utf-8")


class _CompletionHTTPServer(ThreadingHTTPServer):
    """Threading HTTP server carrying typed completion test state."""

    state: _CompletionServerState

    def __init__(self, state: _CompletionServerState) -> None:
        super().__init__(("127.0.0.1", 0), _CompletionHandler)
        self.state = state


class _CompletionHandler(BaseHTTPRequestHandler):
    """Serve a minimal ``LLMGenerator`` completion endpoint."""

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        """Record the POST payload and emit completion JSON."""
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = cast(_CompletionHTTPServer, self.server).state.response_body(
            self.path,
            self.rfile.read(length),
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Suppress access logs during tests."""


class _CompletionServer:
    """Context manager for a localhost completion endpoint."""

    def __init__(self, replies: tuple[str, ...]) -> None:
        self.state = _CompletionServerState(replies=replies)
        self._server = _CompletionHTTPServer(self.state)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        """Return the bound ``/completion`` URL."""
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
        """Stop the server and wait for the thread to exit."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _build_agent(url: str, *, max_candidates: int) -> CoherenceAgent:
    """Construct a production-mode agent backed by the local completion URL."""
    agent = CoherenceAgent(
        llm_api_url=url,
        use_nli=False,
        production_mode=True,
        llm_max_tokens=48,
        llm_temperature=0.0,
        max_candidates=max_candidates,
    )
    agent.scorer.threshold = 0.0
    agent.scorer.soft_limit = 0.0
    return agent


def test_phase3_hardening_unit_guard_declares_real_surface_companions() -> None:
    """The phase3 hardening unit guard is backed by public workflow tests."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_phase3_hardening.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_agent_real_surface.py" in reason
    assert "tests/test_actor_real_surface.py" in reason
    assert "tests/test_config_real_surface.py" in reason
    assert "tests/test_cli_serve_real_surface.py" in reason


def test_agent_process_uses_real_completion_endpoint() -> None:
    """``process`` should drive real completion POSTs through ``LLMGenerator``."""
    replies = (
        "The release receipt was signed by the verification owner.",
        "The release receipt is awaiting the second signature.",
    )

    with _CompletionServer(replies) as server:
        result = _build_agent(server.url, max_candidates=2).process(
            "Was the release receipt signed?"
        )

    assert isinstance(result, ReviewResult)
    assert result.candidates_evaluated == 2
    assert result.halted is False
    assert result.output in replies
    assert [request.path for request in server.state.requests] == [
        "/completion",
        "/completion",
    ]
    assert server.state.requests[0].payload == {
        "prompt": "Was the release receipt signed?",
        "n_predict": 48,
        "temperature": 0.0,
        "stop": ["\nUser:", "\nSystem:"],
    }


def test_batch_processor_runs_real_agent_completion_workflow() -> None:
    """BatchProcessor should drive real agent processing over HTTP in order."""
    prompts = [
        "Was the first batch receipt signed?",
        "Was the second batch receipt signed?",
    ]
    replies = (
        "The first batch receipt was signed.",
        "The second batch receipt was signed.",
    )

    with _CompletionServer(replies) as server:
        processor = BatchProcessor(
            _build_agent(server.url, max_candidates=1),
            max_concurrency=1,
            item_timeout=5.0,
        )
        result = processor.process_batch(prompts, record_metrics=False)

    assert result.total == 2
    assert result.succeeded == 2
    assert result.failed == 0
    review_results = [item for item in result.results if isinstance(item, ReviewResult)]
    assert len(review_results) == 2
    assert [item.output for item in review_results] == list(replies)
    assert [request.payload["prompt"] for request in server.state.requests] == prompts


def test_agent_aprocess_uses_same_real_completion_endpoint() -> None:
    """``aprocess`` should preserve completion wiring through the async facade."""

    async def run(server_url: str) -> ReviewResult:
        return await _build_agent(server_url, max_candidates=1).aprocess(
            "Was the async release receipt signed?"
        )

    with _CompletionServer(("The async release receipt was signed.",)) as server:
        result = asyncio.run(run(server.url))

    assert isinstance(result, ReviewResult)
    assert result.candidates_evaluated == 1
    assert result.halted is False
    assert result.output == "The async release receipt was signed."
    assert server.state.requests == [
        _RecordedCompletionRequest(
            path="/completion",
            payload={
                "prompt": "Was the async release receipt signed?",
                "n_predict": 48,
                "temperature": 0.0,
                "stop": ["\nUser:", "\nSystem:"],
            },
        )
    ]
