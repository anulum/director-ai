# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real-surface coverage for multi-vector retrieval store wiring."""

from __future__ import annotations

from director_ai.core.retrieval.multi_vector import MultiVectorBackend
from director_ai.core.retrieval.vector_store.base import InMemoryBackend
from director_ai.core.retrieval.vector_store.store import VectorGroundTruthStore
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def _store_with_backend(
    backend: MultiVectorBackend,
    *,
    tenant_id: str = "",
) -> VectorGroundTruthStore:
    """Build the production vector store around a concrete multi-vector backend."""
    return VectorGroundTruthStore(backend=backend, tenant_id=tenant_id)


def test_multi_vector_unit_guard_declares_this_real_surface_companion() -> None:
    """The unit guard manifest must point at this production-surface companion."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_multi_vector.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_multi_vector_real_surface.py" in reason


def test_store_retrieval_uses_configured_multi_vector_representations() -> None:
    """Vector store retrieval reaches a MultiVectorBackend without fake adapters."""
    backend = MultiVectorBackend(
        InMemoryBackend(),
        representations=["summary", "title"],
    )
    store = _store_with_backend(backend, tenant_id="tenant-alpha")
    fact_value = (
        "Rollback approvals require signed change evidence. "
        "Operators retain the deployment checklist for audit review."
    )

    store.add(
        "rollback-runbook",
        fact_value,
        metadata={"source_id": "runbook.md"},
    )

    expected = f"rollback-runbook: {fact_value}"
    context = store.retrieve_context("rollback signed evidence", top_k=1)
    chunks = store.retrieve_context_with_chunks("rollback signed evidence", top_k=1)

    assert backend.count() == 2
    assert backend.document_count == 1
    assert context == expected
    assert [chunk.text for chunk in chunks] == [expected]
    assert chunks[0].source.startswith("vector:tenant-alpha::rollback-runbook::")


def test_multi_vector_store_deduplicates_representation_hits() -> None:
    """Multiple matching representations return one evidence row per document."""
    backend = MultiVectorBackend(InMemoryBackend())
    store = _store_with_backend(backend)
    fact_value = (
        "Recovery evidence includes rollback approval, rollback log export, "
        "and rollback operator attestation."
    )

    store.add("recovery-evidence", fact_value)

    chunks = store.retrieve_context_with_chunks(
        "rollback evidence approval attestation",
        top_k=3,
    )

    assert [chunk.text for chunk in chunks] == [f"recovery-evidence: {fact_value}"]


def test_multi_vector_query_preserves_existing_backend_rows() -> None:
    """Pre-existing backend rows stay queryable when no original text is tracked."""
    base = InMemoryBackend()
    base.add(
        "external-doc::content",
        "External retained vector row with escalation evidence.",
        {"doc_id": "external-doc", "representation": "content"},
    )
    backend = MultiVectorBackend(base)

    results = backend.query("retained escalation evidence", n_results=1)

    assert len(results) == 1
    assert (
        results[0]["text"] == "External retained vector row with escalation evidence."
    )
    assert results[0]["metadata"]["doc_id"] == "external-doc"


def test_multi_vector_store_preserves_tenant_isolation() -> None:
    """Tenant filters survive the store, multi-vector decorator, and base backend."""
    backend = MultiVectorBackend(InMemoryBackend())
    store = _store_with_backend(backend)
    alpha_value = "Alpha incident response requires saffron approval evidence."
    beta_value = "Beta incident response requires cobalt approval evidence."

    store.add("incident-runbook", alpha_value, tenant_id="tenant-alpha")
    store.add("incident-runbook", beta_value, tenant_id="tenant-beta")

    alpha_context = store.retrieve_context(
        "saffron approval",
        top_k=2,
        tenant_id="tenant-alpha",
    )
    beta_context = store.retrieve_context(
        "cobalt approval",
        top_k=2,
        tenant_id="tenant-beta",
    )
    cross_tenant_context = store.retrieve_context(
        "saffron",
        top_k=2,
        tenant_id="tenant-beta",
    )

    assert alpha_context == f"incident-runbook: {alpha_value}"
    assert beta_context == f"incident-runbook: {beta_value}"
    assert cross_tenant_context is None
