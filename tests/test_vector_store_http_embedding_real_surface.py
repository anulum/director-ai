# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — HTTP embedding real-surface tests
"""Real public-surface coverage for the HTTP embedding adapter."""

from __future__ import annotations

import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import ClassVar, Self, cast
from urllib.parse import urlparse

import pytest

from director_ai.core.retrieval.vector_store.http_embedding import HttpEmbeddingFunction
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


class _PublicEmbeddingHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible loopback embedding endpoint for tests."""

    vectors_by_text: ClassVar[dict[str, list[float]]] = {}
    requests_seen: ClassVar[list[dict[str, object]]] = []
    headers_seen: ClassVar[list[dict[str, str]]] = []
    paths_seen: ClassVar[list[str]] = []

    def do_POST(self) -> None:
        """Handle a batch embedding request and return configured vectors."""
        length = int(self.headers.get("Content-Length", "0"))
        request_payload = cast(
            dict[str, object],
            json.loads(self.rfile.read(length).decode("utf-8")),
        )
        raw_inputs = request_payload.get("input")
        if not isinstance(raw_inputs, list) or not all(
            isinstance(item, str) for item in raw_inputs
        ):
            self.send_response(400)
            self.end_headers()
            return

        inputs = [cast(str, item) for item in raw_inputs]
        self.__class__.requests_seen.append(request_payload)
        self.__class__.headers_seen.append(dict(self.headers.items()))
        self.__class__.paths_seen.append(self.path)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        data = [{"embedding": self.__class__.vectors_by_text[text]} for text in inputs]
        self.wfile.write(json.dumps({"data": data}).encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:
        """Silence loopback server logging during tests."""
        return


class _PublicEmbeddingServer:
    """Context manager for the loopback embedding endpoint."""

    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self._vectors_by_text = vectors_by_text
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.url = ""

    def __enter__(self) -> Self:
        """Start a loopback server with the configured vector map."""
        _PublicEmbeddingHandler.vectors_by_text = self._vectors_by_text
        _PublicEmbeddingHandler.requests_seen = []
        _PublicEmbeddingHandler.headers_seen = []
        _PublicEmbeddingHandler.paths_seen = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _PublicEmbeddingHandler)
        host, port = cast(tuple[str, int], self._server.server_address)
        self.url = f"http://{host}:{port}"
        self._thread = threading.Thread(target=self._server.serve_forever)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop the loopback server."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def requests(self) -> list[dict[str, object]]:
        """Return embedding requests received by the loopback endpoint."""
        return _PublicEmbeddingHandler.requests_seen

    @property
    def headers(self) -> list[dict[str, str]]:
        """Return request headers received by the loopback endpoint."""
        return _PublicEmbeddingHandler.headers_seen

    @property
    def paths(self) -> list[str]:
        """Return request paths received by the loopback endpoint."""
        return _PublicEmbeddingHandler.paths_seen


def test_vector_store_http_embedding_unit_guard_has_real_surface_companion() -> None:
    """The private helper unit guard should be backed by public adapter coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_vector_store_http_embedding.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_vector_store_http_embedding_real_surface.py" in category


def test_loopback_endpoint_rejects_malformed_embedding_payload() -> None:
    """The loopback endpoint should fail closed on malformed input payloads."""
    with _PublicEmbeddingServer({}) as server:
        parsed = urlparse(server.url)
        assert parsed.hostname is not None
        assert parsed.port is not None
        connection = http.client.HTTPConnection(
            parsed.hostname,
            port=parsed.port,
            timeout=2.0,
        )
        try:
            connection.request(
                "POST",
                "/v1/embeddings",
                body=json.dumps({"input": [123]}),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            response.read()
        finally:
            connection.close()

    assert response.status == 400
    assert server.requests == []


def test_loopback_server_exit_before_enter_is_noop() -> None:
    """The loopback server context manager should tolerate early cleanup."""
    server = _PublicEmbeddingServer({})

    server.__exit__(None, None, None)

    assert server.url == ""


def test_http_embedding_public_client_posts_batch_with_auth_header() -> None:
    """The public adapter should post batches, auth, and endpoint path correctly."""
    with _PublicEmbeddingServer({"alpha": [3.0, 4.0], "beta": [0.0, 5.0]}) as server:
        embed = HttpEmbeddingFunction(
            base_url=f"{server.url}/api",
            model="local-embedder",
            api_key="secret-token",
            vector_size=2,
        )

        vectors = embed(["alpha", "beta"])

    assert vectors[0] == pytest.approx([0.6, 0.8])
    assert vectors[1] == pytest.approx([0.0, 1.0])
    assert server.paths == ["/api/v1/embeddings"]
    assert server.headers[0]["Authorization"] == "Bearer secret-token"
    assert server.requests[0]["model"] == "local-embedder"
    assert server.requests[0]["input"] == ["alpha", "beta"]


def test_http_embedding_from_env_uses_public_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public environment constructor should reach the configured endpoint."""
    with _PublicEmbeddingServer({"query": [0.0, 2.0]}) as server:
        monkeypatch.setenv("DIRECTOR_AI_EMBEDDING_BASE_URL", server.url)
        monkeypatch.setenv("DIRECTOR_AI_EMBEDDING_MODEL", "local-embedder")
        monkeypatch.setenv("DIRECTOR_AI_EMBEDDING_VECTOR_SIZE", "2")

        embed = HttpEmbeddingFunction.from_env()
        vector = embed("query")

    assert vector == pytest.approx([0.0, 1.0])
    assert server.requests[0]["input"] == ["query"]
