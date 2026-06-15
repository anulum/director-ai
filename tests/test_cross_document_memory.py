# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Cross-Document Consistency Memory Tests

import pytest

from director_ai.core import CrossDocumentConsistencyMemory
from director_ai.core.memory.consistency import (
    CrossDocumentConflict,
    CrossDocumentConsistencyReport,
)


def _contradiction_score(previous: str, incoming: str) -> float:
    previous_l = previous.lower()
    incoming_l = incoming.lower()
    if "approved" in previous_l and "not approved" in incoming_l:
        return 0.93
    if "not approved" in previous_l and "approved" in incoming_l:
        return 0.93
    return 0.05


def test_detects_tenant_scoped_cross_document_contradiction(tmp_path):
    memory = CrossDocumentConsistencyMemory(
        tmp_path / "consistency.sqlite",
        score_fn=_contradiction_score,
        contradiction_threshold=0.8,
    )
    memory.record_document(
        tenant_id="tenant-a",
        document_id="doc-1",
        text="Procedure A is approved for use.",
        metadata={"source": "policy-v1"},
    )
    memory.record_document(
        tenant_id="tenant-b",
        document_id="doc-2",
        text="Procedure A is approved for use.",
    )

    report = memory.check_document(
        tenant_id="tenant-a",
        document_id="doc-3",
        text="Procedure A is not approved for use.",
    )
    isolated = memory.check_document(
        tenant_id="tenant-b",
        document_id="doc-4",
        text="Procedure A is not approved for use.",
    )

    assert report.decision == "block"
    assert report.conflicts[0].existing_document_id == "doc-1"
    assert report.conflicts[0].score == 0.93
    assert isolated.conflicts[0].existing_document_id == "doc-2"
    assert all(c.tenant_id == "tenant-a" for c in report.conflicts)


def test_report_serialisation_is_tenant_safe_by_default(tmp_path):
    memory = CrossDocumentConsistencyMemory(
        tmp_path / "consistency.sqlite",
        score_fn=_contradiction_score,
        contradiction_threshold=0.8,
    )
    memory.record_document("tenant-a", "doc-1", "Secret launch is approved.")

    report = memory.check_document(
        "tenant-a",
        "doc-2",
        "Secret launch is not approved.",
    )

    payload = report.to_dict()
    assert "Secret launch" not in str(payload)
    assert payload["conflicts"][0]["existing_hash"]
    assert payload["conflicts"][0]["incoming_hash"]

    payload_with_text = report.to_dict(include_text=True)
    assert "Secret launch is approved" in str(payload_with_text)


def test_record_document_returns_report_and_persists_when_allowed(tmp_path):
    memory = CrossDocumentConsistencyMemory(
        tmp_path / "consistency.sqlite",
        score_fn=_contradiction_score,
        contradiction_threshold=0.8,
    )

    first = memory.record_document("tenant-a", "doc-1", "Policy remains stable.")
    second = memory.record_document("tenant-a", "doc-2", "Policy remains stable.")

    assert first.decision == "allow"
    assert second.decision == "allow"
    assert memory.count(tenant_id="tenant-a") == 2


def test_blocked_document_is_not_persisted(tmp_path):
    memory = CrossDocumentConsistencyMemory(
        tmp_path / "consistency.sqlite",
        score_fn=_contradiction_score,
        contradiction_threshold=0.8,
    )
    memory.record_document("tenant-a", "doc-1", "Procedure A is approved.")

    report = memory.record_document(
        "tenant-a",
        "doc-2",
        "Procedure A is not approved.",
    )

    assert report.decision == "block"
    assert memory.get_document("tenant-a", "doc-2") is None


def test_retention_limit_and_right_to_delete(tmp_path):
    memory = CrossDocumentConsistencyMemory(
        tmp_path / "consistency.sqlite",
        score_fn=_contradiction_score,
        max_documents_per_tenant=2,
    )

    memory.record_document("tenant-a", "doc-1", "one")
    memory.record_document("tenant-a", "doc-2", "two")
    memory.record_document("tenant-a", "doc-3", "three")

    assert memory.get_document("tenant-a", "doc-1") is None
    assert memory.count(tenant_id="tenant-a") == 2
    assert memory.delete_tenant("tenant-a") == 2
    assert memory.count(tenant_id="tenant-a") == 0


def test_builtin_similarity_flags_identical_documents(tmp_path):
    # _builtin_similarity now delegates to the shared text_overlap helper
    # (dispatch + mandatory-failure covered by test_text_overlap); identical
    # documents have overlap 1.0 and must trip the contradiction threshold.
    memory = CrossDocumentConsistencyMemory(
        tmp_path / "consistency.sqlite",
        use_builtin_similarity=True,
        warn_threshold=0.8,
        contradiction_threshold=0.95,
    )
    memory.record_document("tenant-a", "doc-1", "Policy remains approved.")
    report = memory.check_document("tenant-a", "doc-2", "Policy remains approved.")
    assert report.decision in {"warn", "block"}


def test_validation_rejects_invalid_tenant_document_text_scores_and_thresholds(
    tmp_path,
) -> None:
    memory = CrossDocumentConsistencyMemory(tmp_path / "consistency.sqlite")

    with pytest.raises(ValueError, match="tenant_id must match"):
        memory.record_document("bad tenant", "doc-1", "Policy text")

    with pytest.raises(ValueError, match="document_id must be non-empty"):
        memory.record_document("tenant-a", " ", "Policy text")

    with pytest.raises(ValueError, match="text must be non-empty"):
        memory.record_document("tenant-a", "doc-1", " ")

    with pytest.raises(ValueError, match="consistency score"):
        CrossDocumentConflict(
            tenant_id="tenant-a",
            incoming_document_id="incoming",
            existing_document_id="existing",
            incoming_hash="incoming-hash",
            existing_hash="existing-hash",
            score=1.1,
        )

    with pytest.raises(ValueError, match="unsupported decision"):
        CrossDocumentConsistencyReport(
            decision="escalate",
            tenant_id="tenant-a",
            document_id="doc-1",
            incoming_hash="hash",
            checked_documents=0,
        )

    with pytest.raises(ValueError, match="warn_threshold"):
        CrossDocumentConsistencyMemory(
            tmp_path / "bad-thresholds.sqlite",
            warn_threshold=0.9,
            contradiction_threshold=0.8,
        )

    with pytest.raises(ValueError, match="max_documents_per_tenant"):
        CrossDocumentConsistencyMemory(
            tmp_path / "bad-retention.sqlite",
            max_documents_per_tenant=0,
        )


def test_document_serialisation_count_and_close_paths(tmp_path) -> None:
    memory = CrossDocumentConsistencyMemory(tmp_path / "consistency.sqlite")

    memory.record_document(
        "tenant-a",
        "doc-1",
        "Stored tenant policy.",
        metadata={"source": "manual"},
    )
    document = memory.get_document("tenant-a", "doc-1")

    assert document is not None
    assert document.to_dict()["text"] is None
    assert document.to_dict(include_text=True)["text"] == "Stored tenant policy."
    assert document.to_dict()["metadata"] == {"source": "manual"}
    assert memory.count() == 1

    memory.close()


def test_builtin_similarity_scores_partial_overlap(tmp_path) -> None:
    # _builtin_similarity delegates to the shared text_overlap helper; "policy
    # approved" vs "policy pending" share one of three tokens -> 1/3.
    memory = CrossDocumentConsistencyMemory(
        tmp_path / "consistency.sqlite",
        use_builtin_similarity=True,
        warn_threshold=0.2,
        contradiction_threshold=0.9,
    )
    memory.record_document("tenant-a", "doc-1", "policy approved")
    report = memory.check_document("tenant-a", "doc-2", "policy pending")

    assert report.decision == "warn"
    assert report.conflicts[0].score == pytest.approx(1 / 3)


def test_check_document_without_scoring_allows_and_skip_same_document(tmp_path) -> None:
    memory = CrossDocumentConsistencyMemory(tmp_path / "consistency.sqlite")
    memory.record_document("tenant-a", "doc-1", "Policy remains approved.")

    allow_without_scoring = memory.check_document(
        "tenant-a",
        "doc-2",
        "Policy is now rejected.",
    )
    same_doc = CrossDocumentConsistencyMemory(
        tmp_path / "same-doc.sqlite",
        score_fn=lambda _previous, _incoming: 0.99,
        contradiction_threshold=0.9,
    )
    same_doc.record_document("tenant-a", "doc-1", "Policy remains approved.")
    same_doc_report = same_doc.check_document(
        "tenant-a",
        "doc-1",
        "Policy is now rejected.",
    )

    assert allow_without_scoring.decision == "allow"
    assert allow_without_scoring.checked_documents == 1
    assert allow_without_scoring.conflicts == ()
    assert same_doc_report.decision == "allow"
    assert same_doc_report.checked_documents == 1
    assert same_doc_report.conflicts == ()


def test_builtin_similarity_jaccard_values(tmp_path):
    # The overlap that backs _builtin_similarity is Jaccard token overlap (now
    # via the shared text_overlap helper; empties score 0).
    memory = CrossDocumentConsistencyMemory(tmp_path / "consistency.sqlite")
    assert memory._builtin_similarity(
        "the quick brown fox", "the lazy brown dog"
    ) == pytest.approx(2 / 6)
    assert memory._builtin_similarity("no shared tokens", "completely different") == 0.0
