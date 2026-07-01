# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real-surface coverage for reranked vector retrieval."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import pytest

from director_ai.core.retrieval.vector_store import (
    InMemoryBackend,
    RerankedBackend,
    VectorGroundTruthStore,
)
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


class _LocalCrossEncoder:
    """Deterministic local reranker with the CrossEncoder predict contract."""

    calls: ClassVar[list[list[tuple[str, str]]]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score query/document pairs by local keyword evidence."""
        self.calls.append(list(pairs))
        return [_rerank_score(query, document) for query, document in pairs]


def _rerank_score(query: str, document: str) -> float:
    """Return a stable local relevance score for test documents."""
    query_terms = set(query.lower().split())
    document_terms = set(document.lower().split())
    score = float(len(query_terms & document_terms))
    if {"refund", "signed", "approval"} <= document_terms:
        score += 5.0
    if {"billing", "finance"} <= document_terms:
        score -= 1.0
    return score


def _reranked_backend() -> RerankedBackend:
    """Create a real reranked backend with a local deterministic reranker."""
    _LocalCrossEncoder.calls = []
    base = InMemoryBackend()
    base.add(
        "alpha-general",
        "Refund policy overview for customer support.",
        {"tenant_id": "tenant-alpha", "source_id": "support"},
    )
    base.add(
        "alpha-signed",
        "Refund signed approval packet for regulated customer evidence.",
        {"tenant_id": "tenant-alpha", "source_id": "runbook"},
    )
    base.add(
        "beta-billing",
        "Billing finance desk owns invoice queue triage.",
        {"tenant_id": "tenant-beta", "source_id": "finance"},
    )
    return RerankedBackend(
        base,
        reranker=_LocalCrossEncoder(),
        top_k_multiplier=3,
    )


@pytest.mark.parametrize(
    "unit_guard",
    ["tests/test_reranker.py", "tests/test_vector_store_reranker.py"],
)
def test_reranker_unit_guards_declare_this_companion(unit_guard: str) -> None:
    """Reranker unit guards must point at this production-surface companion."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[unit_guard]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_vector_store_reranker_real_surface.py" in reason


def test_real_reranked_backend_uses_injected_reranker_and_tenant_filter() -> None:
    """RerankedBackend should rerank real base results without model downloads."""
    backend = _reranked_backend()

    results = backend.query(
        "refund signed approval",
        n_results=2,
        tenant_id="tenant-alpha",
    )

    assert [result["id"] for result in results] == ["alpha-signed", "alpha-general"]
    assert all(result["metadata"]["tenant_id"] == "tenant-alpha" for result in results)
    assert len(_LocalCrossEncoder.calls) == 1
    assert len(_LocalCrossEncoder.calls[0]) == 2
    assert backend.count() == 3


def test_real_reranked_backend_wires_into_vector_ground_truth_store() -> None:
    """VectorGroundTruthStore should retrieve context through real reranking."""
    store = VectorGroundTruthStore(
        backend=_reranked_backend(),
        tenant_id="tenant-alpha",
    )

    context = store.retrieve_context("refund signed approval", top_k=1)

    assert context == "Refund signed approval packet for regulated customer evidence."


def test_reranked_backend_rejects_invalid_injected_reranker() -> None:
    """Injected rerankers must provide the CrossEncoder predict contract."""
    with pytest.raises(ValueError, match="reranker"):
        RerankedBackend(
            InMemoryBackend(),
            reranker=cast(Any, object()),
        )
