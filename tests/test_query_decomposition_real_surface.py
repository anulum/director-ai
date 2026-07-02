# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real-surface coverage for query-decomposition retrieval wiring."""

from __future__ import annotations

import director_ai
from director_ai.core.config import DirectorConfig
from director_ai.core.retrieval.knowledge import GroundTruthStore
from director_ai.core.retrieval.query_decomposition import QueryDecompositionBackend
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def _query_decomposition_store() -> GroundTruthStore:
    """Build the dependency-free production store path with decomposition on."""
    config = DirectorConfig(
        hybrid_retrieval=False,
        reranker_enabled=False,
        query_decomposition_enabled=True,
        query_decomposition_strategy="heuristic",
    )
    return config.build_store()


def test_query_decomposition_unit_guard_declares_this_companion() -> None:
    """The legacy unit guard should point to this production-surface companion."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_query_decomposition.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_query_decomposition_real_surface.py" in reason


def test_config_build_store_retrieves_both_compound_query_intents() -> None:
    """Configured stores should retrieve both halves of a compound question."""
    store = _query_decomposition_store()

    store.add(
        "refund-approval",
        "Refund approval requires signed manager evidence.",
    )
    store.add(
        "shipping-sla",
        "Shipping delivery window is three business days.",
    )

    context = store.retrieve_context(
        "What controls refund approval and shipping delivery window?",
        top_k=2,
    )

    assert context is not None
    assert "Refund approval requires signed manager evidence." in context
    assert "Shipping delivery window is three business days." in context


def test_query_decomposition_store_preserves_tenant_filtering() -> None:
    """Sub-query fanout must keep tenant filters on every backend query."""
    store = _query_decomposition_store()
    store.add(
        "refund-approval",
        "Alpha refund approvals require legal evidence.",
        tenant_id="tenant-alpha",
    )
    store.add(
        "shipping-policy",
        "Beta shipping policies require warehouse evidence.",
        tenant_id="tenant-beta",
    )

    alpha_context = store.retrieve_context(
        "refund approvals and shipping policies",
        top_k=2,
        tenant_id="tenant-alpha",
    )
    beta_context = store.retrieve_context(
        "refund approvals and shipping policies",
        top_k=2,
        tenant_id="tenant-beta",
    )
    missing_context = store.retrieve_context(
        "refund approvals and shipping policies",
        top_k=2,
        tenant_id="tenant-gamma",
    )

    assert (
        alpha_context
        == "refund-approval: Alpha refund approvals require legal evidence."
    )
    assert (
        beta_context
        == "shipping-policy: Beta shipping policies require warehouse evidence."
    )
    assert missing_context is None


def test_public_query_decomposition_export_matches_backend_class() -> None:
    """The package-level lazy export should expose the production backend class."""
    assert director_ai.QueryDecompositionBackend is QueryDecompositionBackend
