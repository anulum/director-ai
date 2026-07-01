# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real-surface coverage for the core vector store production path."""

from __future__ import annotations

import pytest

from director_ai.core.retrieval.vector_store.base import InMemoryBackend
from director_ai.core.retrieval.vector_store.store import VectorGroundTruthStore
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def test_vector_store_unit_guard_declares_this_real_surface_companion() -> None:
    """The unit guard manifest must point at this production-surface companion."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_vector_store.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_vector_store_real_surface.py" in reason


def test_vector_store_tracks_versions_snapshots_and_tenant_retrieval() -> None:
    """VectorGroundTruthStore should return tenant-scoped evidence with metadata."""
    store = VectorGroundTruthStore(tenant_id="tenant-alpha")
    alpha_value = "Refund approvals require signed operator evidence."
    beta_value = "Refund approvals require finance desk evidence."

    store.add_fact(
        "refund-policy",
        alpha_value,
        metadata={
            "source_id": "runbook-alpha",
            "external_id": "kb-alpha-1",
            "source_timestamp": "1710000000.0",
            "updated_timestamp": "1710000300.0",
            "citation_status": "fresh",
            "status_observed_at": "1710000600.0",
        },
    )
    store.add_fact(
        "refund-policy",
        beta_value,
        tenant_id="tenant-beta",
        metadata={"source_id": "runbook-beta"},
    )

    alpha_context = store.retrieve_context("signed refund evidence", top_k=2)
    beta_context = store.retrieve_context(
        "finance refund evidence",
        top_k=2,
        tenant_id="tenant-beta",
    )
    alpha_chunks = store.retrieve_context_with_chunks(
        "signed refund evidence",
        top_k=2,
    )
    alpha_manifest = store.version_manifest("tenant-alpha")
    alpha_record = alpha_manifest["tenant-alpha::refund-policy"]
    snapshot_audit = store.kb_snapshot_audit_record("tenant-alpha")
    freshness = store.freshness_status_signals("tenant-alpha", key="refund-policy")

    assert alpha_context == f"refund-policy: {alpha_value}"
    assert beta_context == f"refund-policy: {beta_value}"
    assert [chunk.text for chunk in alpha_chunks] == [f"refund-policy: {alpha_value}"]
    assert alpha_chunks[0].source == "vector:tenant-alpha::refund-policy"
    assert alpha_record["version"] == "1.0.0"
    assert alpha_record["source_id"] == "runbook-alpha"
    assert alpha_record["external_id"] == "kb-alpha-1"
    assert snapshot_audit["record_count"] == 1
    assert snapshot_audit["tenant_id"] == "tenant-alpha"
    assert isinstance(snapshot_audit["merkle_root"], str)
    assert len(snapshot_audit["merkle_root"]) == 64
    assert freshness == [
        {
            "source_id": "kb-alpha-1",
            "status": "fresh",
            "status_source": "",
            "published_at": 1710000000.0,
            "updated_at": 1710000300.0,
            "observed_at": 1710000600.0,
        }
    ]


def test_retraction_and_replacement_filter_stale_backend_rows() -> None:
    """Retracted and replaced facts should not leak stale vector rows."""
    store = VectorGroundTruthStore()
    original_value = "Rollback approvals require paper records."
    replacement_value = "Rollback approvals require signed deployment evidence."

    store.add_fact(
        "rollback-policy",
        original_value,
        metadata={
            "signed_fact_id": "signed-fact-1",
            "claim_source": "signed_fact",
        },
    )
    root_before_retraction = store.kb_snapshot_root()

    retraction = store.retract_fact(
        "rollback-policy",
        reason="superseded by signed deployment controls",
    )
    after_retraction = store.retrieve_context("paper rollback records", top_k=3)
    after_retraction_chunks = store.retrieve_context_with_chunks(
        "paper rollback records",
        top_k=3,
    )
    replacement = store.replace_fact(
        "rollback-policy",
        replacement_value,
        reason="signed deployment controls adopted",
        metadata={
            "signed_fact_id": "signed-fact-1",
            "claim_source": "signed_fact",
            "kb_version_bump": "minor",
        },
    )
    replacement_context = store.retrieve_context("signed deployment evidence", top_k=3)
    replacement_chunks = store.retrieve_context_with_chunks(
        "signed deployment evidence",
        top_k=3,
    )
    manifest = store.version_manifest()

    assert retraction["event"] == "retracted"
    assert retraction["content_hash"]
    assert after_retraction is None
    assert after_retraction_chunks == []
    assert replacement["event"] == "replaced"
    assert replacement["from_version"] == "1.0.0"
    assert replacement["to_version"] == "1.1.0"
    assert replacement_context == f"rollback-policy: {replacement_value}"
    assert [chunk.text for chunk in replacement_chunks] == [
        f"rollback-policy: {replacement_value}"
    ]
    assert manifest["rollback-policy"]["replacement_reason"] == (
        "signed deployment controls adopted"
    )
    assert store.retraction_records() == [retraction]
    assert store.replacement_records() == [replacement]
    assert store.kb_snapshot_root() != root_before_retraction


@pytest.mark.asyncio
async def test_in_memory_backend_async_methods_use_real_sync_backend() -> None:
    """VectorBackend async helpers should execute the real in-memory backend."""
    backend = InMemoryBackend()

    await backend.aadd(
        "async-policy",
        "Async rollback checks retain signed evidence.",
        {"tenant_id": "tenant-alpha"},
    )
    results = await backend.aquery(
        "signed rollback evidence",
        n_results=1,
        tenant_id="tenant-alpha",
    )

    assert backend.count() == 1
    assert [result["id"] for result in results] == ["async-policy"]
    assert results[0]["text"] == "Async rollback checks retain signed evidence."
