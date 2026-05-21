# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Cross-Document Consistency Memory Tests

import director_ai.core.memory.consistency as consistency_mod
from director_ai.core import CrossDocumentConsistencyMemory


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


def test_builtin_similarity_uses_rust_kernel_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(consistency_mod, "_RUST_CONSISTENCY", True)
    called = {"count": 0}

    def _rust_overlap(text_a: str, text_b: str) -> float:
        called["count"] += 1
        return 0.9

    monkeypatch.setattr(consistency_mod, "rust_word_overlap", _rust_overlap, raising=True)
    memory = CrossDocumentConsistencyMemory(
        tmp_path / "consistency.sqlite",
        use_builtin_similarity=True,
        warn_threshold=0.8,
        contradiction_threshold=0.95,
    )
    memory.record_document("tenant-a", "doc-1", "Policy remains approved.")
    report = memory.check_document("tenant-a", "doc-2", "Policy remains approved.")
    assert called["count"] >= 1
    assert report.decision == "warn"


def test_builtin_similarity_type_error_falls_back_to_python(tmp_path, monkeypatch):
    monkeypatch.setattr(consistency_mod, "_RUST_CONSISTENCY", True)
    monkeypatch.setattr(
        consistency_mod,
        "rust_word_overlap",
        lambda _text_a, _text_b: (_ for _ in ()).throw(
            TypeError("ffi signature mismatch")
        ),
        raising=True,
    )
    memory = CrossDocumentConsistencyMemory(
        tmp_path / "consistency.sqlite",
        use_builtin_similarity=True,
        warn_threshold=0.2,
        contradiction_threshold=0.95,
    )
    memory.record_document("tenant-a", "doc-1", "Policy remains approved.")
    report = memory.check_document("tenant-a", "doc-2", "Policy remains approved.")
    assert report.decision in {"warn", "block", "allow"}
