# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — HTTP embedding adapter tests

"""Multi-angle tests for the HTTP embedding adapter."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

import pytest

from director_ai.core.config import DirectorConfig
from director_ai.core.vector_store import (
    FAISSBackend,
    HttpEmbeddingConnectionError,
    HttpEmbeddingDimensionError,
    HttpEmbeddingFunction,
    HttpEmbeddingResponseError,
    RemanentiaBackendError,
    RemanentiaVectorBackend,
)


class _EmbeddingHandler(BaseHTTPRequestHandler):
    response_payload: ClassVar[dict[str, Any]] = {"data": []}
    response_factory: ClassVar[Callable[[dict[str, Any]], dict[str, Any]] | None] = None
    raw_response: ClassVar[bytes | None] = None
    status_code: ClassVar[int] = 200
    delay_s: ClassVar[float] = 0.0
    requests_seen: ClassVar[list[dict[str, Any]]] = []
    headers_seen: ClassVar[list[dict[str, str]]] = []
    paths_seen: ClassVar[list[str]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        request_payload = json.loads(body.decode("utf-8"))
        self.__class__.requests_seen.append(request_payload)
        self.__class__.headers_seen.append(dict(self.headers.items()))
        self.__class__.paths_seen.append(self.path)

        if self.delay_s:
            time.sleep(self.delay_s)

        self.send_response(self.status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        if self.raw_response is not None:
            self.wfile.write(self.raw_response)
            return

        factory = _EmbeddingHandler.response_factory
        payload = (
            factory(request_payload) if factory is not None else self.response_payload
        )
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:
        return


class _EmbeddingServer:
    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        factory: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        raw_response: bytes | None = None,
        status_code: int = 200,
        delay_s: float = 0.0,
    ) -> None:
        self._payload = payload or {"data": []}
        self._factory = factory
        self._raw_response = raw_response
        self._status_code = status_code
        self._delay_s = delay_s
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.url = ""

    @property
    def requests(self) -> list[dict[str, Any]]:
        return _EmbeddingHandler.requests_seen

    @property
    def headers(self) -> list[dict[str, str]]:
        return _EmbeddingHandler.headers_seen

    @property
    def paths(self) -> list[str]:
        return _EmbeddingHandler.paths_seen

    def __enter__(self) -> _EmbeddingServer:
        _EmbeddingHandler.response_payload = self._payload
        _EmbeddingHandler.response_factory = self._factory
        _EmbeddingHandler.raw_response = self._raw_response
        _EmbeddingHandler.status_code = self._status_code
        _EmbeddingHandler.delay_s = self._delay_s
        _EmbeddingHandler.requests_seen = []
        _EmbeddingHandler.headers_seen = []
        _EmbeddingHandler.paths_seen = []

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _EmbeddingHandler)
        host, port = self._server.server_address
        self.url = f"http://{host}:{port}"
        self._thread = threading.Thread(target=self._server.serve_forever)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class _RemanentiaHandler(BaseHTTPRequestHandler):
    status_code: ClassVar[int] = 200
    requests_seen: ClassVar[list[dict[str, Any]]] = []

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps({"episodic_traces": 2, "semantic_memories": 3}).encode("utf-8")
        )

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        request_payload = json.loads(body.decode("utf-8"))
        self.__class__.requests_seen.append(request_payload)
        self.send_response(self.status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(
                {
                    "query": request_payload["query"],
                    "results": [
                        {
                            "chunk_id": "paper:0",
                            "text": "verified public evidence",
                            "score": 0.8,
                            "metadata": {"source": "paper"},
                        }
                    ],
                }
            ).encode("utf-8")
        )

    def log_message(self, format: str, *args: object) -> None:
        return


class _RemanentiaServer:
    def __init__(self) -> None:
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.url = ""

    def __enter__(self) -> _RemanentiaServer:
        _RemanentiaHandler.requests_seen = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _RemanentiaHandler)
        host, port = self._server.server_address
        self.url = f"http://{host}:{port}"
        self._thread = threading.Thread(target=self._server.serve_forever)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def requests(self) -> list[dict[str, Any]]:
        return _RemanentiaHandler.requests_seen


def test_successful_single_embedding_normalises_vector() -> None:
    with _EmbeddingServer({"data": [{"embedding": [3.0, 4.0]}]}) as server:
        embed = HttpEmbeddingFunction(
            base_url=server.url,
            model="embedding-model",
            vector_size=2,
        )

        vector = embed("hello")

    assert vector == pytest.approx([0.6, 0.8])
    assert server.requests[0]["model"] == "embedding-model"
    assert server.requests[0]["input"] == ["hello"]


def test_successful_batch_embedding_uses_embeddings_schema() -> None:
    payload = {"embeddings": [[1.0, 0.0], [0.0, 2.0]]}
    with _EmbeddingServer(payload) as server:
        embed = HttpEmbeddingFunction(
            base_url=server.url,
            model="embedding-model",
            vector_size=2,
        )

        vectors = embed(["first", "second"])

    assert vectors[0] == pytest.approx([1.0, 0.0])
    assert vectors[1] == pytest.approx([0.0, 1.0])
    assert server.requests[0]["input"] == ["first", "second"]


def test_base_url_with_version_path_appends_embeddings_endpoint() -> None:
    with _EmbeddingServer({"data": [{"embedding": [1.0, 0.0]}]}) as server:
        embed = HttpEmbeddingFunction(
            base_url=f"{server.url}/v1",
            model="embedding-model",
            vector_size=2,
        )

        embed("hello")

    assert server.paths == ["/v1/embeddings"]


def test_optional_bearer_token_header() -> None:
    with _EmbeddingServer({"data": [{"embedding": [1.0, 0.0]}]}) as server:
        embed = HttpEmbeddingFunction(
            base_url=server.url,
            model="embedding-model",
            api_key="token",
            vector_size=2,
        )

        embed("hello")

    assert server.headers[0]["Authorization"] == "Bearer token"


def test_response_count_mismatch_raises() -> None:
    with _EmbeddingServer({"data": [{"embedding": [1.0, 0.0]}]}) as server:
        embed = HttpEmbeddingFunction(
            base_url=server.url,
            model="embedding-model",
            vector_size=2,
        )

        with pytest.raises(HttpEmbeddingResponseError, match="expected 2"):
            embed(["first", "second"])


def test_dimension_mismatch_raises() -> None:
    with _EmbeddingServer({"data": [{"embedding": [1.0, 0.0]}]}) as server:
        embed = HttpEmbeddingFunction(
            base_url=server.url,
            model="embedding-model",
            vector_size=3,
        )

        with pytest.raises(HttpEmbeddingDimensionError, match="expected dimension 3"):
            embed("hello")


def test_malformed_json_raises() -> None:
    with _EmbeddingServer(raw_response=b"{not-json") as server:
        embed = HttpEmbeddingFunction(
            base_url=server.url,
            model="embedding-model",
            vector_size=2,
        )

        with pytest.raises(HttpEmbeddingResponseError, match="invalid JSON"):
            embed("hello")


def test_missing_embedding_field_raises() -> None:
    with _EmbeddingServer({"data": [{"not_embedding": [1.0, 0.0]}]}) as server:
        embed = HttpEmbeddingFunction(
            base_url=server.url,
            model="embedding-model",
            vector_size=2,
        )

        with pytest.raises(HttpEmbeddingResponseError, match="embedding field"):
            embed("hello")


def test_http_status_error_raises() -> None:
    with _EmbeddingServer({"error": "no"}, status_code=503) as server:
        embed = HttpEmbeddingFunction(
            base_url=server.url,
            model="embedding-model",
            vector_size=2,
        )

        with pytest.raises(HttpEmbeddingResponseError, match="HTTP 503"):
            embed("hello")


def test_timeout_maps_to_connection_error() -> None:
    with _EmbeddingServer(
        {"data": [{"embedding": [1.0, 0.0]}]},
        delay_s=0.2,
    ) as server:
        embed = HttpEmbeddingFunction(
            base_url=server.url,
            model="embedding-model",
            timeout_s=0.05,
            vector_size=2,
        )

        with pytest.raises(HttpEmbeddingConnectionError):
            embed("hello")


def test_invalid_scheme_rejected() -> None:
    with pytest.raises(ValueError, match="scheme"):
        HttpEmbeddingFunction(
            base_url="ftp://example.invalid",
            model="embedding-model",
        )


def test_from_env_reads_public_configuration_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _EmbeddingServer({"data": [{"embedding": [0.0, 3.0]}]}) as server:
        monkeypatch.setenv("DIRECTOR_AI_EMBEDDING_BASE_URL", server.url)
        monkeypatch.setenv("DIRECTOR_AI_EMBEDDING_MODEL", "embedding-model")
        monkeypatch.setenv("DIRECTOR_AI_EMBEDDING_VECTOR_SIZE", "2")

        embed = HttpEmbeddingFunction.from_env()
        vector = embed("hello")

    assert vector == pytest.approx([0.0, 1.0])


def test_director_config_redacts_embedding_api_key() -> None:
    cfg = DirectorConfig(embedding_api_key="secret")

    assert cfg.to_dict()["embedding_api_key"] == "***"


def test_director_config_http_faiss_requires_base_url() -> None:
    with pytest.raises(ValueError, match="embedding_base_url"):
        DirectorConfig(vector_backend="http-faiss")


def test_faiss_pipeline_with_http_embedding() -> None:
    pytest.importorskip("faiss")

    vectors = {
        "alpha document": [1.0, 0.0],
        "beta document": [0.0, 1.0],
        "alpha query": [1.0, 0.0],
    }

    def _factory(request: dict[str, Any]) -> dict[str, Any]:
        return {
            "data": [{"embedding": vectors[text]} for text in request["input"]],
        }

    with _EmbeddingServer(factory=_factory) as server:
        embed = HttpEmbeddingFunction(
            base_url=server.url,
            model="embedding-model",
            vector_size=2,
        )
        backend = FAISSBackend(embed_fn=embed, vector_size=2)
        backend.add("alpha", "alpha document")
        backend.add("beta", "beta document")

        results = backend.query("alpha query", n_results=1)

    assert results[0]["id"] == "alpha"


def test_director_config_builds_http_faiss_store() -> None:
    pytest.importorskip("faiss")

    vectors = {
        "alpha: alpha document": [1.0, 0.0],
        "beta: beta document": [0.0, 1.0],
        "alpha query": [1.0, 0.0],
    }

    def _factory(request: dict[str, Any]) -> dict[str, Any]:
        return {
            "data": [{"embedding": vectors[text]} for text in request["input"]],
        }

    with _EmbeddingServer(factory=_factory) as server:
        cfg = DirectorConfig(
            vector_backend="http-faiss",
            embedding_base_url=server.url,
            embedding_model="embedding-model",
            embedding_vector_size=2,
            hybrid_retrieval=False,
            reranker_enabled=False,
        )
        store = cfg.build_store()
        store.add_fact("alpha", "alpha document")
        store.add_fact("beta", "beta document")

        context = store.retrieve_context("alpha query", top_k=1)

    assert context is not None
    assert "alpha document" in context


def test_remanentia_backend_queries_public_vector_api() -> None:
    with _RemanentiaServer() as server:
        backend = RemanentiaVectorBackend(
            base_url=server.url,
            timeout_s=2.0,
            source="paper",
        )
        results = backend.query("evidence", n_results=1)

    assert results == [
        {
            "id": "paper:0",
            "text": "verified public evidence",
            "distance": pytest.approx(0.2),
            "metadata": {"source": "paper"},
        }
    ]
    assert server.requests[0]["source"] == "paper"


def test_remanentia_backend_count_reads_status() -> None:
    with _RemanentiaServer() as server:
        backend = RemanentiaVectorBackend(base_url=server.url)
        count = backend.count()

    assert count == 5


def test_remanentia_backend_is_read_only() -> None:
    backend = RemanentiaVectorBackend()

    with pytest.raises(RemanentiaBackendError, match="read-only"):
        backend.add("doc", "text")


def test_director_config_builds_remanentia_store() -> None:
    with _RemanentiaServer() as server:
        cfg = DirectorConfig(
            vector_backend="remanentia",
            remanentia_base_url=server.url,
            remanentia_source="paper",
            hybrid_retrieval=True,
            reranker_enabled=True,
        )
        store = cfg.build_store()
        context = store.retrieve_context("evidence", top_k=1)

    assert context == "verified public evidence"
    assert server.requests[0]["source"] == "paper"


def test_director_config_remanentia_requires_base_url() -> None:
    with pytest.raises(ValueError, match="remanentia_base_url"):
        DirectorConfig(vector_backend="remanentia", remanentia_base_url="")
