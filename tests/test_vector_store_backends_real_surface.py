# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real-surface coverage for in-process vector store backend adapters."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("faiss", reason="faiss required for vector backend real surface")

from director_ai.core.retrieval.vector_store import FAISSBackend, get_vector_backend
from director_ai.core.retrieval.vector_store.store import VectorGroundTruthStore
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def _token_embedding(text: str) -> list[float]:
    """Return a deterministic local embedding for FAISS adapter tests."""
    normalized = text.lower()
    return [
        1.0 if "refund" in normalized else 0.0,
        1.0 if "billing" in normalized else 0.0,
        1.0 if "audit" in normalized or "signed" in normalized else 0.0,
        1.0 if "async" in normalized or "fallback" in normalized else 0.0,
    ]


def _result_ids(rows: list[dict[str, Any]]) -> list[str]:
    """Extract stable document identifiers from vector backend rows."""
    return [str(row["id"]) for row in rows]


def test_vector_store_backends_unit_guard_declares_this_companion() -> None:
    """The vendor-adapter guard must point at this production-surface companion."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_vector_store_backends.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_vector_store_backends_real_surface.py" in reason


def test_real_faiss_flat_index_preserves_tenant_boundaries() -> None:
    """FAISSBackend should search the real FAISS index without tenant leakage."""
    assert get_vector_backend("faiss") is FAISSBackend
    backend = FAISSBackend(embed_fn=_token_embedding, vector_size=4)

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

    alpha_results = backend.query(
        "alpha refund signed evidence",
        n_results=3,
        tenant_id="tenant-alpha",
    )
    beta_results = backend.query(
        "beta billing evidence",
        n_results=2,
        tenant_id="tenant-beta",
    )

    assert backend.count() == 3
    assert _result_ids(alpha_results)[:2] == ["alpha-refund", "alpha-audit"]
    assert _result_ids(beta_results) == ["beta-billing"]
    assert all(row["metadata"]["tenant_id"] == "tenant-alpha" for row in alpha_results)
    assert alpha_results[0]["distance"] < alpha_results[-1]["distance"]


def test_real_faiss_backend_wires_into_vector_ground_truth_store() -> None:
    """VectorGroundTruthStore should retrieve evidence through the FAISS backend."""
    store = VectorGroundTruthStore(
        backend=FAISSBackend(embed_fn=_token_embedding, vector_size=4),
        tenant_id="tenant-alpha",
    )

    store.add_fact(
        "refund-policy",
        "Alpha refund policy requires signed approval evidence.",
        metadata={"source_id": "faiss-real"},
    )
    store.add_fact(
        "refund-policy",
        "Beta refund policy requires billing desk approval.",
        tenant_id="tenant-beta",
        metadata={"source_id": "faiss-real"},
    )

    alpha_context = store.retrieve_context(
        "alpha signed refund evidence",
        top_k=2,
    )
    beta_context = store.retrieve_context(
        "beta billing refund evidence",
        top_k=2,
        tenant_id="tenant-beta",
    )

    assert alpha_context == (
        "refund-policy: Alpha refund policy requires signed approval evidence."
    )
    assert beta_context == (
        "refund-policy: Beta refund policy requires billing desk approval."
    )


@pytest.mark.parametrize(
    ("ivf_nlist", "message"),
    [(True, "ivf_nlist must be an integer"), (0, "ivf_nlist must be positive")],
)
def test_real_faiss_backend_rejects_invalid_ivf_nlist(
    ivf_nlist: Any,
    message: str,
) -> None:
    """FAISSBackend should reject unusable IVF cluster-count settings."""
    with pytest.raises(ValueError, match=message):
        FAISSBackend(
            embed_fn=_token_embedding,
            vector_size=4,
            index_type="ivf",
            ivf_nlist=ivf_nlist,
        )


def test_real_faiss_ivf_bootstraps_small_indexes_before_training() -> None:
    """IVF mode should remain queryable before enough vectors exist to train."""
    backend = FAISSBackend(
        embed_fn=_token_embedding,
        vector_size=4,
        index_type="ivf",
        ivf_nlist=2,
    )

    backend.add(
        "alpha-refund",
        "Alpha refund policy requires signed operator evidence.",
        {"tenant_id": "tenant-alpha"},
    )
    before_training = backend.query(
        "alpha refund signed evidence",
        n_results=1,
        tenant_id="tenant-alpha",
    )
    backend.add(
        "beta-billing",
        "Beta billing policy requires finance desk evidence.",
        {"tenant_id": "tenant-beta"},
    )
    after_training = backend.query(
        "beta billing evidence",
        n_results=1,
        tenant_id="tenant-beta",
    )

    assert _result_ids(before_training) == ["alpha-refund"]
    assert _result_ids(after_training) == ["beta-billing"]
    assert backend.count() == 2


async def _exercise_async_faiss_backend() -> list[str]:
    """Run the VectorBackend async helpers against a real FAISS backend."""
    backend = FAISSBackend(embed_fn=_token_embedding, vector_size=4)
    await backend.aadd(
        "async-alpha",
        "Alpha async retrieval keeps signed evidence available.",
        {"tenant_id": "tenant-alpha"},
    )
    rows = await backend.aquery(
        "alpha async signed evidence",
        n_results=1,
        tenant_id="tenant-alpha",
    )
    return _result_ids(rows)


def test_real_faiss_backend_uses_vector_backend_async_contract() -> None:
    """Default VectorBackend async helpers should work with real FAISS methods."""
    assert asyncio.run(_exercise_async_faiss_backend()) == ["async-alpha"]
