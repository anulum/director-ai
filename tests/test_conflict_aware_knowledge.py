# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — conflict-aware KB guard tests

from __future__ import annotations

from director_ai.core.retrieval.conflict_guard import (
    ConflictAwareKnowledgeGuard,
    KnowledgeFact,
)
from director_ai.core.retrieval.knowledge import GroundTruthStore
from director_ai.core.retrieval.vector_store import (
    InMemoryBackend,
    VectorGroundTruthStore,
)


def test_guard_blocks_same_key_conflict_before_plain_store_mutation() -> None:
    store = GroundTruthStore()
    store.add_fact("refund_policy", "Refunds are available within 30 days.")
    guard = ConflictAwareKnowledgeGuard(store)

    result = guard.add_fact(
        KnowledgeFact(
            key="refund_policy",
            value="Refunds are never available.",
            metadata={"claim_id": "claim-refund"},
        )
    )

    assert result.decision == "block"
    assert result.conflicts[0].conflict_type == "same_key_value_mismatch"
    assert result.conflicts[0].incoming_hash
    assert result.conflicts[0].existing_hash
    assert "never available" not in str(result.to_dict())
    assert store.facts["refund_policy"] == "Refunds are available within 30 days."


def test_guard_allows_idempotent_fact_and_records_retrieval_safe_evidence() -> None:
    store = GroundTruthStore()
    guard = ConflictAwareKnowledgeGuard(store)

    first = guard.add_fact(
        KnowledgeFact(key="boiling_point", value="Water boils at 100 C.")
    )
    second = guard.add_fact(
        KnowledgeFact(key="boiling_point", value="Water boils at 100 C.")
    )

    assert first.decision == "allow"
    assert second.decision == "allow"
    assert second.evidence_refs == ("kb://boiling_point",)
    assert store.retrieve_context("boiling_point") == "Water boils at 100 C."


def test_guard_detects_explicit_contradiction_reference_before_vector_add() -> None:
    store = VectorGroundTruthStore(backend=InMemoryBackend())
    store.add_fact(
        "policy_v1",
        "Refunds are available within 30 days.",
        metadata={"claim_id": "claim-refund-v1"},
    )
    guard = ConflictAwareKnowledgeGuard(store)

    result = guard.add_fact(
        KnowledgeFact(
            key="policy_v2",
            value="Refunds are never available.",
            metadata={"contradicts": "claim-refund-v1"},
        )
    )

    assert result.decision == "block"
    assert result.conflicts[0].conflict_type == "explicit_contradiction"
    assert store.fact_version("policy_v2") is None
    assert store.conflict_reports() == []


def test_guard_allowed_vector_fact_keeps_keyword_and_version_paths() -> None:
    store = VectorGroundTruthStore(backend=InMemoryBackend())
    guard = ConflictAwareKnowledgeGuard(store)

    result = guard.add_fact(
        KnowledgeFact(
            key="policy_v1",
            value="Refunds are available within 30 days.",
            metadata={"claim_id": "claim-refund-v1"},
        )
    )

    assert result.decision == "allow"
    assert store.fact_version("policy_v1") == "1.0.0"
    assert store.facts["policy_v1"] == "Refunds are available within 30 days."
    assert "Refunds are available within 30 days." in store.retrieve_context(
        "policy_v1"
    )


def test_guard_can_warn_on_scorer_conflict_without_storing_raw_values() -> None:
    store = GroundTruthStore()
    store.add_fact("availability", "The service is available in Zurich.")
    guard = ConflictAwareKnowledgeGuard(
        store,
        score_fn=lambda existing, incoming: 0.72,
        warn_threshold=0.6,
        block_threshold=0.9,
    )

    result = guard.check_fact(
        KnowledgeFact(key="coverage", value="The service is unavailable in Zurich.")
    )

    assert result.decision == "warn"
    assert result.conflicts[0].conflict_type == "semantic_contradiction"
    assert result.conflicts[0].score == 0.72
    assert "unavailable" not in str(result.to_dict())


def test_lazy_import_export() -> None:
    from director_ai import ConflictAwareKnowledgeGuard as RootGuard

    assert RootGuard is ConflictAwareKnowledgeGuard
