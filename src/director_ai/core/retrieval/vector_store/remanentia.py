# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Remanentia vector backend

"""Read-only vector backend backed by the local Remanentia API."""

from __future__ import annotations

import http.client
import json
from typing import Any
from urllib.parse import urlparse

from .base import VectorBackend

__all__ = [
    "RemanentiaBackendError",
    "RemanentiaVectorBackend",
]


class RemanentiaBackendError(RuntimeError):
    """Raised when the Remanentia API cannot serve a vector query."""


class RemanentiaVectorBackend(VectorBackend):
    """Query Remanentia's public-safe vector endpoint as a backend.

    The backend is intentionally read-only. Remanentia owns ingestion,
    indexing, public-corpus allowlisting, and redaction. Director-AI
    consumes the returned evidence as grounding context.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001",
        timeout_s: float = 5.0,
        source: str = "",
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("base_url scheme must be http or https")
        if not parsed.netloc:
            raise ValueError("base_url must include a host")
        if parsed.params or parsed.query or parsed.fragment:
            raise ValueError("base_url must not include params, query, or fragment")

        self._scheme = parsed.scheme
        self._host = parsed.hostname or ""
        self._port = parsed.port
        self._base_path = parsed.path.rstrip("/")
        self._timeout_s = timeout_s
        self._source = source

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        raise RemanentiaBackendError(
            "RemanentiaVectorBackend is read-only; ingest into Remanentia instead",
        )

    def query(
        self,
        text: str,
        n_results: int = 3,
        tenant_id: str = "",
    ) -> list[dict[str, Any]]:
        if n_results <= 0:
            return []
        payload = {
            "query": text,
            "top_k": n_results,
            "source": self._source or tenant_id,
        }
        response = self._request_json("POST", "/vector/search/public", payload)
        results = response.get("results")
        if not isinstance(results, list):
            raise RemanentiaBackendError("Remanentia vector response missing results")

        docs: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                raise RemanentiaBackendError(
                    "Remanentia vector result must be an object"
                )
            chunk_id = str(item.get("chunk_id", ""))
            score = _float_or_zero(item.get("score"))
            metadata = item.get("metadata")
            docs.append(
                {
                    "id": chunk_id,
                    "text": str(item.get("text", "")),
                    "distance": max(0.0, 1.0 - score),
                    "metadata": metadata if isinstance(metadata, dict) else {},
                }
            )
        return docs

    def count(self) -> int:
        response = self._request_json("GET", "/status", None)
        semantic = int(response.get("semantic_memories", 0))
        traces = int(response.get("episodic_traces", 0))
        return semantic + traces

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        body = b"" if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"

        connection_cls = (
            http.client.HTTPSConnection
            if self._scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_cls(
            self._host, port=self._port, timeout=self._timeout_s
        )
        try:
            connection.request(
                method, f"{self._base_path}{path}", body=body, headers=headers
            )
            response = connection.getresponse()
            raw = response.read()
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            raise RemanentiaBackendError("Remanentia API request failed") from exc
        finally:
            connection.close()

        if response.status < 200 or response.status >= 300:
            raise RemanentiaBackendError(
                f"Remanentia API returned HTTP {response.status}"
            )
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemanentiaBackendError(
                "Remanentia API returned invalid JSON"
            ) from exc
        if not isinstance(decoded, dict):
            raise RemanentiaBackendError("Remanentia API response must be an object")
        return decoded


def _float_or_zero(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)
