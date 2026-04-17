# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vector backend protocol and registry

"""VectorBackend protocol, registry, and the dependency-free
in-memory backend.

Keeping the ABC and the baseline backend together means any client
can instantiate a working store with zero optional dependencies —
useful for tests and examples that should not require
``sentence-transformers``, ``chromadb``, or a cloud vendor SDK.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger("DirectorAI.VectorStore")

# Re-export recommended model name for documentation
RECOMMENDED_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

__all__ = [
    "RECOMMENDED_EMBEDDING_MODEL",
    "VectorBackend",
    "InMemoryBackend",
    "register_vector_backend",
    "get_vector_backend",
    "list_vector_backends",
]

_VECTOR_REGISTRY: dict[str, type[VectorBackend]] = {}
_VECTOR_EP_LOADED = False


def register_vector_backend(name: str, cls: type[VectorBackend]) -> None:
    """Register a vector backend class under *name*."""
    if not (isinstance(cls, type) and issubclass(cls, VectorBackend)):
        raise TypeError(f"{cls!r} must be a VectorBackend subclass")
    _VECTOR_REGISTRY[name] = cls
    logger.debug("Registered vector backend: %s", name)


def get_vector_backend(name: str) -> type[VectorBackend]:
    """Look up a registered vector backend by name."""
    _load_vector_entry_points()
    if name not in _VECTOR_REGISTRY:
        raise KeyError(
            f"Unknown vector backend {name!r}. "
            f"Available: {list(_VECTOR_REGISTRY)}",
        )
    return _VECTOR_REGISTRY[name]


def list_vector_backends() -> dict[str, type[VectorBackend]]:
    """Return all registered vector backends."""
    _load_vector_entry_points()
    return dict(_VECTOR_REGISTRY)


def _load_vector_entry_points() -> None:
    """Discover backends from ``director_ai.vector_backends`` entry points."""
    global _VECTOR_EP_LOADED
    if _VECTOR_EP_LOADED:
        return
    _VECTOR_EP_LOADED = True
    try:
        from importlib.metadata import entry_points

        eps = entry_points()
        group: Any = (
            eps.get("director_ai.vector_backends", [])
            if isinstance(eps, dict)
            else eps.select(group="director_ai.vector_backends")
        )
        for ep in group:
            try:
                cls = ep.load()
                if ep.name not in _VECTOR_REGISTRY:  # pragma: no cover
                    register_vector_backend(ep.name, cls)
            except (ImportError, AttributeError, TypeError) as exc:
                # pragma: no cover
                logger.warning(
                    "Failed to load vector backend entry point %s: %s",
                    ep.name,
                    exc,
                )
    except ImportError:
        pass


class VectorBackend(ABC):
    """Protocol for vector database backends."""

    @abstractmethod
    def add(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...  # pragma: no cover

    @abstractmethod
    def query(
        self,
        text: str,
        n_results: int = 3,
        tenant_id: str = "",
    ) -> list[dict[str, Any]]: ...  # pragma: no cover

    @abstractmethod
    def count(self) -> int: ...  # pragma: no cover

    async def aadd(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Async add — delegates to sync add via executor by default."""
        import asyncio

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.add, doc_id, text, metadata)

    async def aquery(
        self,
        text: str,
        n_results: int = 3,
        tenant_id: str = "",
    ) -> list[dict[str, Any]]:
        """Async query — delegates to sync query via executor by default."""
        import asyncio
        import functools

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(
                self.query,
                text,
                n_results=n_results,
                tenant_id=tenant_id,
            ),
        )


class InMemoryBackend(VectorBackend):
    """Simple in-memory cosine-similarity backend (no external deps).

    Uses TF-IDF-like word overlap for embedding approximation.
    Suitable for testing and small fact stores.
    """

    def __init__(self) -> None:
        self._docs: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._docs.append({"id": doc_id, "text": text, "metadata": metadata or {}})

    def query(
        self,
        text: str,
        n_results: int = 3,
        tenant_id: str = "",
    ) -> list[dict[str, Any]]:
        with self._lock:
            snapshot = list(self._docs)

        if not snapshot:
            return []
        docs = [
            d
            for d in snapshot
            if not tenant_id or d["metadata"].get("tenant_id") == tenant_id
        ]
        if not docs:
            return []
        import re

        _strip = re.compile(r"[^\w\s]")
        query_words = set(_strip.sub("", text).lower().split())
        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in docs:
            doc_words = set(_strip.sub("", doc["text"]).lower().split())
            overlap = len(query_words & doc_words)
            total = max(len(query_words | doc_words), 1)
            similarity = overlap / total
            scored.append((similarity, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[dict[str, Any]] = []
        scored_sliced = scored[:n_results]
        for sim, doc2 in scored_sliced:
            if sim > 0:
                results.append({**doc2, "distance": 1.0 - sim})
        return results

    def count(self) -> int:
        with self._lock:
            return len(self._docs)

