# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — conflict-aware KB guard tests

from __future__ import annotations

import pytest

from director_ai.core.retrieval.conflict_guard import (
    ConflictAwareKnowledgeGuard,
    KnowledgeConflict,
    KnowledgeConflictCheck,
    KnowledgeFact,
    _dedupe_conflicts,
    _metadata_refs,
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


def test_conflict_records_validate_type_score_and_decision() -> None:
    with pytest.raises(ValueError, match="conflict_type is required"):
        KnowledgeConflict(
            conflict_type=" ",
            incoming_key="incoming",
            existing_key="existing",
            incoming_hash="a",
            existing_hash="b",
            score=0.5,
            evidence_refs=("kb://existing",),
            reason="empty type",
        )
    with pytest.raises(ValueError, match="score"):
        KnowledgeConflict(
            conflict_type="semantic_contradiction",
            incoming_key="incoming",
            existing_key="existing",
            incoming_hash="a",
            existing_hash="b",
            score=1.5,
            evidence_refs=("kb://existing",),
            reason="bad score",
        )
    with pytest.raises(ValueError, match="unsupported decision"):
        KnowledgeConflictCheck(
            decision="defer",
            incoming_key="incoming",
            tenant_id="tenant-a",
            incoming_hash="a",
        )


def test_guard_rejects_invalid_threshold_configuration() -> None:
    store = GroundTruthStore()

    with pytest.raises(ValueError, match="warn_threshold"):
        ConflictAwareKnowledgeGuard(store, warn_threshold=float("nan"))
    with pytest.raises(ValueError, match="block_threshold"):
        ConflictAwareKnowledgeGuard(store, block_threshold=1.5)
    with pytest.raises(ValueError, match="warn_threshold must be <= block_threshold"):
        ConflictAwareKnowledgeGuard(store, warn_threshold=0.9, block_threshold=0.7)


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


def test_guard_still_blocks_same_key_mismatch_by_score_when_flag_disabled() -> None:
    store = GroundTruthStore()
    store.add_fact("tenant-a:refund_policy", "Refunds are available within 30 days.")
    guard = ConflictAwareKnowledgeGuard(store, block_on_same_key_mismatch=False)

    result = guard.check_fact(
        KnowledgeFact(
            key="refund_policy",
            value="Refunds are unavailable.",
            tenant_id="tenant-a",
        )
    )

    assert result.decision == "block"
    assert result.blocked is True
    assert result.tenant_id == "tenant-a"
    assert result.conflicts[0].conflict_type == "same_key_value_mismatch"


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


def test_guard_still_blocks_explicit_contradiction_by_score_when_flag_disabled() -> (
    None
):
    store = VectorGroundTruthStore(backend=InMemoryBackend())
    store.add_fact(
        "policy_v1",
        "Refunds are available within 30 days.",
        metadata={"external_id": "external-refund-policy"},
    )
    guard = ConflictAwareKnowledgeGuard(
        store,
        block_on_explicit_contradiction=False,
    )

    result = guard.check_fact(
        KnowledgeFact(
            key="policy_v2",
            value="Refunds are unavailable.",
            metadata={"contradicts": ["external-refund-policy"]},
        )
    )

    assert result.decision == "block"
    assert result.conflicts[0].conflict_type == "explicit_contradiction"
    assert result.conflicts[0].evidence_refs == ("kb://policy_v1",)


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


def test_guard_filters_sensitive_metadata_before_store_mutation() -> None:
    store = VectorGroundTruthStore(backend=InMemoryBackend())
    guard = ConflictAwareKnowledgeGuard(store)

    result = guard.add_fact(
        KnowledgeFact(
            key="safe_fact",
            value="Public documentation exists.",
            metadata={
                "claim_id": "claim-safe",
                "raw_prompt": "discard",
                "private_key_hint": "discard",
            },
        )
    )

    record = store.fact_version_record("safe_fact")
    assert result.decision == "allow"
    assert record is not None
    assert record["claim_id"] == "claim-safe"
    assert "raw_prompt" not in record
    assert "private_key_hint" not in record


def test_guard_uses_legacy_store_add_when_add_fact_lacks_metadata() -> None:
    class LegacyStore:
        def __init__(self) -> None:
            self.facts: dict[str, str] = {}
            self.add_calls: list[tuple[str, str, dict[str, object], str]] = []

        def add_fact(self, key: str, value: str, *, tenant_id: str = "") -> None:
            self.facts[f"{tenant_id}:{key}" if tenant_id else key] = value

        def add(
            self,
            key: str,
            value: str,
            *,
            metadata: dict[str, object],
            tenant_id: str = "",
        ) -> None:
            self.add_calls.append((key, value, metadata, tenant_id))
            self.facts[f"{tenant_id}:{key}" if tenant_id else key] = value

    store = LegacyStore()
    guard = ConflictAwareKnowledgeGuard(store)  # type: ignore[arg-type]

    result = guard.add_fact(
        KnowledgeFact(
            key="legacy_fact",
            value="Legacy stores can still receive safe metadata.",
            tenant_id="tenant-a",
            metadata={"claim_id": "claim-legacy"},
        )
    )

    assert result.decision == "allow"
    assert store.add_calls == [
        (
            "legacy_fact",
            "Legacy stores can still receive safe metadata.",
            {"claim_id": "claim-legacy"},
            "tenant-a",
        )
    ]


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


def test_guard_ignores_unmatched_explicit_refs_and_low_semantic_scores() -> None:
    store = VectorGroundTruthStore(backend=InMemoryBackend())
    store.add_fact(
        "policy_v1",
        "Refunds are available within 30 days.",
        metadata={"claim_id": "claim-refund-v1"},
    )
    guard = ConflictAwareKnowledgeGuard(
        store,
        score_fn=lambda _existing, _incoming: 0.2,
        warn_threshold=0.6,
        block_threshold=0.9,
    )

    result = guard.check_fact(
        KnowledgeFact(
            key="policy_v2",
            value="Refunds are available within 14 days.",
            metadata={"contradicts": "different-claim"},
        )
    )

    assert result.decision == "allow"
    assert result.conflicts == ()


def test_guard_blocks_high_scoring_semantic_conflict() -> None:
    store = GroundTruthStore()
    store.add_fact("tenant-a:availability", "The service is available in Zurich.")
    store.add_fact("tenant-b:availability", "The service is available in Basel.")
    calls: list[tuple[str, str]] = []

    def score(existing: str, incoming: str) -> float:
        calls.append((existing, incoming))
        return 0.92

    guard = ConflictAwareKnowledgeGuard(
        store,
        score_fn=score,
        warn_threshold=0.6,
        block_threshold=0.9,
    )

    result = guard.check_fact(
        KnowledgeFact(
            key="coverage",
            value="The service is unavailable in Zurich.",
            tenant_id="tenant-a",
        )
    )

    assert result.decision == "block"
    assert result.blocked is True
    assert result.conflicts[0].existing_key == "availability"
    assert calls == [
        ("The service is available in Zurich.", "The service is unavailable in Zurich.")
    ]


def test_guard_uses_version_record_hash_when_fact_text_is_missing() -> None:
    store = VectorGroundTruthStore(backend=InMemoryBackend())
    store.add_fact(
        "policy_v1",
        "Refunds are available within 30 days.",
        metadata={"claim_id": "claim-refund-v1"},
    )
    stored_record = store.fact_version_record("policy_v1")
    assert stored_record is not None
    del store.facts["policy_v1"]
    guard = ConflictAwareKnowledgeGuard(store)

    result = guard.check_fact(
        KnowledgeFact(
            key="policy_v2",
            value="Refunds are unavailable.",
            metadata={"contradicts": "claim-refund-v1"},
        )
    )

    assert result.decision == "block"
    assert result.conflicts[0].existing_hash


def test_guard_uses_empty_value_for_missing_version_record() -> None:
    class ManifestStore(GroundTruthStore):
        def version_manifest(self, tenant_id: str = ""):
            return {"ghost": {"key": "ghost", "claim_id": "missing-claim"}}

        def fact_version_record(self, key: str, tenant_id: str = ""):
            return None

    guard = ConflictAwareKnowledgeGuard(ManifestStore())

    result = guard.check_fact(
        KnowledgeFact(
            key="incoming",
            value="Incoming claim.",
            metadata={"contradicts": "missing-claim"},
        )
    )

    assert result.decision == "block"
    assert result.conflicts[0].existing_hash


def test_metadata_refs_normalise_empty_sequences_and_scalar_values() -> None:
    assert _metadata_refs(None) == set()
    assert _metadata_refs("") == set()
    assert _metadata_refs((" claim-a ", "", "claim-b")) == {"claim-a", "claim-b"}
    assert _metadata_refs(123) == {"123"}


def test_guard_deduplicates_conflicts_from_multiple_detection_paths() -> None:
    store = GroundTruthStore()
    store.add_fact("availability", "The service is available in Zurich.")
    guard = ConflictAwareKnowledgeGuard(
        store,
        score_fn=lambda _existing, _incoming: 1.0,
        block_on_same_key_mismatch=False,
    )

    result = guard.check_fact(
        KnowledgeFact(
            key="availability",
            value="The service is unavailable in Zurich.",
        )
    )

    assert result.decision == "block"
    assert [conflict.conflict_type for conflict in result.conflicts] == [
        "same_key_value_mismatch"
    ]


def test_dedupe_conflicts_removes_duplicate_markers() -> None:
    first = KnowledgeConflict(
        conflict_type="semantic_contradiction",
        incoming_key="incoming",
        existing_key="existing",
        incoming_hash="in",
        existing_hash="ex",
        score=0.7,
        evidence_refs=("kb://existing",),
        reason="first",
    )
    duplicate = KnowledgeConflict(
        conflict_type="semantic_contradiction",
        incoming_key="incoming",
        existing_key="existing",
        incoming_hash="in",
        existing_hash="ex",
        score=0.9,
        evidence_refs=("kb://existing-2",),
        reason="duplicate",
    )
    unique = KnowledgeConflict(
        conflict_type="semantic_contradiction",
        incoming_key="incoming",
        existing_key="other",
        incoming_hash="in",
        existing_hash="other",
        score=0.7,
        evidence_refs=("kb://other",),
        reason="unique",
    )

    assert _dedupe_conflicts([first, duplicate, unique]) == [first, unique]


def test_lazy_import_export() -> None:
    from director_ai import ConflictAwareKnowledgeGuard as RootGuard

    assert RootGuard is ConflictAwareKnowledgeGuard
