# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real-surface coverage for contextual-compression retrieval wiring."""

from __future__ import annotations

import director_ai
from director_ai.core.config import DirectorConfig
from director_ai.core.retrieval.contextual_compression import (
    ContextualCompressionBackend,
)
from director_ai.core.retrieval.vector_store import VectorGroundTruthStore
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def _contextual_compression_store() -> VectorGroundTruthStore:
    """Build the dependency-free production store with compression enabled."""
    config = DirectorConfig(
        hybrid_retrieval=False,
        reranker_enabled=False,
        contextual_compression_enabled=True,
        contextual_compression_strategy="heuristic",
    )
    store = config.build_store()
    assert isinstance(store, VectorGroundTruthStore)
    return store


def test_contextual_compression_unit_guard_declares_this_companion() -> None:
    """The legacy unit guard should point to this production-surface companion."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_contextual_compression.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_contextual_compression_real_surface.py" in reason


def test_config_build_store_compresses_retrieved_context() -> None:
    """Configured stores should return only query-relevant retrieved sentences."""
    store = _contextual_compression_store()
    store.add(
        "refund-policy",
        (
            "Refund approval requires manager approval. "
            "Shipping windows are managed by operations. "
            "Weather alerts are handled by facilities."
        ),
    )

    context = store.retrieve_context("refund approval", top_k=1)

    assert context == "refund-policy: Refund approval requires manager approval."
    assert "Shipping windows" not in context
    assert "Weather alerts" not in context


def test_compression_preserves_original_text_audit_metadata() -> None:
    """Compressed backend results should keep original text metadata."""
    store = _contextual_compression_store()
    store.add(
        "refund-policy",
        (
            "Refund approval requires manager approval. "
            "Shipping windows are managed by operations. "
            "Weather alerts are handled by facilities."
        ),
    )

    results = store.backend.query("refund approval", n_results=1)

    assert results
    result = results[0]
    metadata = result["metadata"]
    assert isinstance(metadata, dict)
    assert result["text"] == "refund-policy: Refund approval requires manager approval."
    assert "Shipping windows" in metadata["original_text"]
    assert 0.0 < metadata["compression_ratio"] < 1.0


def test_contextual_compression_store_preserves_tenant_filtering() -> None:
    """Compression must not leak retrieved evidence across tenant filters."""
    store = _contextual_compression_store()
    store.add(
        "refund-policy",
        (
            "Alpha refund approval requires legal review. "
            "Alpha shipping notes are not relevant."
        ),
        tenant_id="tenant-alpha",
    )
    store.add(
        "refund-policy",
        (
            "Beta refund approval requires finance review. "
            "Beta shipping notes are not relevant."
        ),
        tenant_id="tenant-beta",
    )

    alpha_context = store.retrieve_context(
        "refund approval",
        top_k=1,
        tenant_id="tenant-alpha",
    )
    beta_context = store.retrieve_context(
        "refund approval",
        top_k=1,
        tenant_id="tenant-beta",
    )
    missing_context = store.retrieve_context(
        "refund approval",
        top_k=1,
        tenant_id="tenant-gamma",
    )

    assert (
        alpha_context == "refund-policy: Alpha refund approval requires legal review."
    )
    assert (
        beta_context == "refund-policy: Beta refund approval requires finance review."
    )
    assert missing_context is None


def test_public_contextual_compression_export_matches_backend_class() -> None:
    """The package-level lazy export should expose the production backend class."""
    assert director_ai.ContextualCompressionBackend is ContextualCompressionBackend
