# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real-surface coverage for embedding-backed vector stores."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import pytest

pytest.importorskip("chromadb", reason="chromadb required for embedding real surface")

from director_ai.core.retrieval.vector_store.embedding import ChromaBackend
from director_ai.core.retrieval.vector_store.store import VectorGroundTruthStore
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


class _LocalEmbeddingFunction:
    """Deterministic local Chroma embedding function for offline retrieval."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed Chroma document batches without network or model downloads."""
        vectors: list[list[float]] = []
        for text in input:
            normalized = text.lower()
            vectors.append(
                [
                    1.0 if "refund" in normalized else 0.0,
                    1.0 if "billing" in normalized else 0.0,
                    1.0 if "audit" in normalized or "signed" in normalized else 0.0,
                    1.0 if "rollback" in normalized else 0.0,
                ]
            )
        return vectors


class _NativeEmbeddingFunction(_LocalEmbeddingFunction):
    """Chroma-compatible embedding function with provider metadata."""

    @staticmethod
    def name() -> str:
        """Return the Chroma provider name."""
        return "native-local-embedding"

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """Return query embeddings for Chroma query calls."""
        return self(input)

    def default_space(self) -> str:
        """Return Chroma's default distance space."""
        return "l2"

    def supported_spaces(self) -> list[str]:
        """Return distance spaces accepted by this provider."""
        return ["cosine", "l2", "ip"]

    def get_config(self) -> dict[str, str]:
        """Return serialisable Chroma provider metadata."""
        return {"provider": self.name()}

    @staticmethod
    def build_from_config(config: dict[str, str]) -> _NativeEmbeddingFunction:
        """Rebuild the native local embedder from Chroma metadata."""
        return _NativeEmbeddingFunction()

    def is_legacy(self) -> bool:
        """Return False because Chroma metadata hooks are implemented."""
        return False


class _EmbeddingProviderMetadata(Protocol):
    """Metadata surface Chroma expects on embedding providers."""

    def name(self) -> str:
        """Return the Chroma provider name."""
        ...

    def get_config(self) -> dict[str, str]:
        """Return serialisable provider metadata."""
        ...

    def build_from_config(self, config: dict[str, str]) -> object:
        """Build an embedding provider from metadata."""
        ...


def _ids(rows: list[dict[str, object]]) -> list[str]:
    """Return result identifiers as strings for stable assertions."""
    return [str(row["id"]) for row in rows]


def _chroma_backend(tmp_path: Path, collection_name: str) -> ChromaBackend:
    """Create a real persistent Chroma backend with a local embedder."""
    return ChromaBackend(
        collection_name=collection_name,
        persist_directory=str(tmp_path / "chroma"),
        embedding_function=_LocalEmbeddingFunction(),
    )


def test_vector_store_embedding_unit_guard_declares_this_companion() -> None:
    """The embedding guard must point at this production-surface companion."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_vector_store_embedding.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_vector_store_embedding_real_surface.py" in reason


def test_real_chroma_backend_uses_local_embeddings_and_persistence(
    tmp_path: Path,
) -> None:
    """ChromaBackend should persist and query with an injected local embedder."""
    backend = _chroma_backend(tmp_path, "tenant_facts")

    backend.add(
        "alpha-refund",
        "Alpha refund policy requires signed operator evidence.",
        {"tenant_id": "tenant-alpha", "source_id": "runbook-alpha"},
    )
    backend.add(
        "beta-billing",
        "Beta billing policy requires finance desk evidence.",
        {"tenant_id": "tenant-beta", "source_id": "runbook-beta"},
    )
    backend.add(
        "alpha-audit",
        "Alpha audit packets retain signed source hashes.",
        {"tenant_id": "tenant-alpha", "source_id": "runbook-alpha"},
    )

    alpha_rows = backend.query(
        "refund signed evidence",
        n_results=3,
        tenant_id="tenant-alpha",
    )
    beta_rows = backend.query(
        "billing finance evidence",
        n_results=2,
        tenant_id="tenant-beta",
    )
    reopened = _chroma_backend(tmp_path, "tenant_facts")
    reopened_rows = reopened.query(
        "audit signed evidence",
        n_results=2,
        tenant_id="tenant-alpha",
    )

    assert backend.count() == 3
    assert _ids(alpha_rows)[:2] == ["alpha-refund", "alpha-audit"]
    assert _ids(beta_rows) == ["beta-billing"]
    assert _ids(reopened_rows)[:2] == ["alpha-audit", "alpha-refund"]
    assert all(row["metadata"]["tenant_id"] == "tenant-alpha" for row in alpha_rows)
    assert reopened.delete(["alpha-audit"]) == 1
    assert reopened.count() == 2


def test_real_chroma_backend_exposes_adapter_metadata(
    tmp_path: Path,
) -> None:
    """Injected local callables should expose Chroma provider metadata."""
    backend = _chroma_backend(tmp_path, "metadata")
    provider = cast(
        _EmbeddingProviderMetadata,
        backend._collection._embedding_function,
    )

    assert provider.name() == "director-ai-local-embedding"
    assert provider.get_config() == {"provider": "director-ai-local-embedding"}
    with pytest.raises(ValueError, match="provided at runtime"):
        provider.build_from_config(provider.get_config())


def test_real_chroma_backend_accepts_native_embedding_function(
    tmp_path: Path,
) -> None:
    """Chroma-native embedding functions should pass through and query storage."""
    backend = ChromaBackend(
        collection_name="native",
        persist_directory=str(tmp_path / "native"),
        embedding_function=_NativeEmbeddingFunction(),
    )

    backend.add(
        "alpha-refund",
        "Alpha refund policy requires signed operator evidence.",
        {"tenant_id": "tenant-alpha"},
    )
    rows = backend.query(
        "refund signed evidence",
        n_results=1,
        tenant_id="tenant-alpha",
    )

    assert _ids(rows) == ["alpha-refund"]


def test_real_chroma_backend_wires_into_vector_ground_truth_store(
    tmp_path: Path,
) -> None:
    """VectorGroundTruthStore should retrieve tenant facts through real Chroma."""
    store = VectorGroundTruthStore(
        backend=_chroma_backend(tmp_path, "ground_truth"),
        tenant_id="tenant-alpha",
    )

    store.add_fact(
        "refund-policy",
        "Alpha refund policy requires signed approval evidence.",
        metadata={"source_id": "chroma-real"},
    )
    store.add_fact(
        "refund-policy",
        "Beta billing policy requires finance desk approval.",
        tenant_id="tenant-beta",
        metadata={"source_id": "chroma-real"},
    )

    alpha_context = store.retrieve_context(
        "signed refund evidence",
        top_k=2,
    )
    beta_context = store.retrieve_context(
        "billing approval evidence",
        top_k=2,
        tenant_id="tenant-beta",
    )

    assert alpha_context == (
        "refund-policy: Alpha refund policy requires signed approval evidence."
    )
    assert beta_context == (
        "refund-policy: Beta billing policy requires finance desk approval."
    )


def test_chroma_backend_rejects_ambiguous_embedding_configuration(
    tmp_path: Path,
) -> None:
    """Operators should not configure two Chroma embedding providers at once."""
    with pytest.raises(ValueError, match="embedding_model and embedding_function"):
        ChromaBackend(
            collection_name="ambiguous",
            persist_directory=str(tmp_path / "ambiguous"),
            embedding_model="local-model",
            embedding_function=_LocalEmbeddingFunction(),
        )
