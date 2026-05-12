# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — HTTP embedding adapter

"""HTTP embedding function for vector backends.

The adapter is intentionally a callable embedding function rather than
a storage backend. It can feed existing backends such as FAISS, Qdrant,
Pinecone, Weaviate, or Elasticsearch without introducing a new index
type or hard-coding deployment details.
"""

from __future__ import annotations

import http.client
import json
import math
import os
from collections.abc import Sequence
from typing import overload
from urllib.parse import urlparse

__all__ = [
    "HttpEmbeddingConnectionError",
    "HttpEmbeddingDimensionError",
    "HttpEmbeddingError",
    "HttpEmbeddingFunction",
    "HttpEmbeddingResponseError",
]


class HttpEmbeddingError(RuntimeError):
    """Base error for HTTP embedding failures."""


class HttpEmbeddingConnectionError(HttpEmbeddingError):
    """Raised when the embedding endpoint cannot be reached."""


class HttpEmbeddingResponseError(HttpEmbeddingError):
    """Raised when the embedding endpoint returns malformed data."""


class HttpEmbeddingDimensionError(HttpEmbeddingResponseError):
    """Raised when returned vectors have unexpected dimensions."""


class HttpEmbeddingFunction:
    """Callable batch-capable embedding client for HTTP endpoints."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout_s: float = 10.0,
        vector_size: int | None = None,
        endpoint_path: str = "/v1/embeddings",
        normalize: bool = True,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must be set")
        if not model.strip():
            raise ValueError("model must be set")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        if vector_size is not None and vector_size < 1:
            raise ValueError("vector_size must be >= 1")

        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("base_url scheme must be http or https")
        if not parsed.netloc:
            raise ValueError("base_url must include a host")
        if parsed.params or parsed.query or parsed.fragment:
            raise ValueError("base_url must not include params, query, or fragment")

        if not endpoint_path.startswith("/"):
            raise ValueError("endpoint_path must start with '/'")
        path = _embedding_path(parsed.path, endpoint_path)

        self._scheme = parsed.scheme
        self._host = parsed.hostname or ""
        self._port = parsed.port
        self._path = path
        self._model = model
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._vector_size = vector_size
        self._normalize = normalize

    @classmethod
    def from_env(cls, prefix: str = "DIRECTOR_AI_EMBEDDING_") -> HttpEmbeddingFunction:
        """Build from environment variables with the given prefix."""
        base_url = os.environ.get(f"{prefix}BASE_URL", "")
        model = os.environ.get(f"{prefix}MODEL", "")
        api_key = os.environ.get(f"{prefix}API_KEY", "")
        timeout_s = float(os.environ.get(f"{prefix}TIMEOUT_S", "10.0"))
        vector_size_raw = os.environ.get(f"{prefix}VECTOR_SIZE", "")
        vector_size = int(vector_size_raw) if vector_size_raw else None
        return cls(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
            vector_size=vector_size,
        )

    @overload
    def __call__(self, texts: str) -> list[float]: ...

    @overload
    def __call__(self, texts: list[str] | tuple[str, ...]) -> list[list[float]]: ...

    def __call__(
        self,
        texts: str | list[str] | tuple[str, ...],
    ) -> list[float] | list[list[float]]:
        """Embed one string or a batch of strings."""
        if isinstance(texts, str):
            return self.embed(texts)
        return self.embed_many(list(texts))

    def embed(self, text: str) -> list[float]:
        """Embed one text string."""
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a non-empty batch of text strings."""
        batch = list(texts)
        if not batch:
            return []
        if not all(isinstance(text, str) for text in batch):
            raise TypeError("all embedding inputs must be strings")
        if any(not text.strip() for text in batch):
            raise ValueError("embedding inputs must be non-empty strings")

        payload = self._request_embeddings(batch)
        vectors = self._extract_vectors(payload)
        if len(vectors) != len(batch):
            raise HttpEmbeddingResponseError(
                f"expected {len(batch)} embeddings, got {len(vectors)}",
            )
        return [self._validate_vector(vector) for vector in vectors]

    def _request_embeddings(self, texts: list[str]) -> object:
        body = json.dumps({"model": self._model, "input": texts}).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        connection_cls = (
            http.client.HTTPSConnection
            if self._scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_cls(
            self._host,
            port=self._port,
            timeout=self._timeout_s,
        )
        try:
            connection.request("POST", self._path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read()
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            raise HttpEmbeddingConnectionError(
                "embedding endpoint request failed",
            ) from exc
        finally:
            connection.close()

        if response.status < 200 or response.status >= 300:
            raise HttpEmbeddingResponseError(
                f"embedding endpoint returned HTTP {response.status}",
            )
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HttpEmbeddingResponseError(
                "embedding endpoint returned invalid JSON",
            ) from exc

    def _extract_vectors(self, payload: object) -> list[list[object]]:
        if not isinstance(payload, dict):
            raise HttpEmbeddingResponseError("embedding response must be an object")

        data = payload.get("data")
        if isinstance(data, list):
            vectors = []
            for item in data:
                if not isinstance(item, dict) or "embedding" not in item:
                    raise HttpEmbeddingResponseError(
                        "embedding data items must contain an embedding field",
                    )
                embedding = item["embedding"]
                if not isinstance(embedding, list):
                    raise HttpEmbeddingResponseError("embedding must be a list")
                vectors.append(embedding)
            return vectors

        embeddings = payload.get("embeddings")
        if isinstance(embeddings, list):
            if not all(isinstance(vector, list) for vector in embeddings):
                raise HttpEmbeddingResponseError("embeddings must be a list of lists")
            return embeddings

        raise HttpEmbeddingResponseError(
            "embedding response must contain data or embeddings",
        )

    def _validate_vector(self, vector: list[object]) -> list[float]:
        values: list[float] = []
        for item in vector:
            if isinstance(item, bool) or not isinstance(item, int | float):
                raise HttpEmbeddingResponseError("embedding values must be numeric")
            values.append(float(item))

        if self._vector_size is not None and len(values) != self._vector_size:
            raise HttpEmbeddingDimensionError(
                f"expected dimension {self._vector_size}, got {len(values)}",
            )

        if not self._normalize:
            return values

        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            raise HttpEmbeddingDimensionError("embedding vector norm must be non-zero")
        return [value / norm for value in values]


def _embedding_path(base_path: str, endpoint_path: str) -> str:
    if not endpoint_path.startswith("/"):
        raise ValueError("endpoint_path must start with '/'")
    clean_endpoint = endpoint_path.rstrip("/") or "/"
    clean_base = base_path.rstrip("/")
    if not clean_base:
        return clean_endpoint
    if clean_base == clean_endpoint:
        return clean_base
    if clean_endpoint.startswith(f"{clean_base}/"):
        return clean_endpoint
    return f"{clean_base}{clean_endpoint}"
