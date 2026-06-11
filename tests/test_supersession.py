# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — KnowledgeSupersessionPolicy tests

"""Tests for the human-gated supersession policy.

Covers candidate detection from explicit hints, same-source revisions, and
contradiction scores; exclusion of the incoming document; candidate
ordering by score; the default human-approval gate; opt-in auto-promotion
and its all-candidates threshold rule; and parameter/candidate validation."""

from __future__ import annotations

import pytest

from director_ai.core.provenance import (
    KnowledgeSupersessionPolicy,
    SupersessionCandidate,
    SupersessionDecision,
)
from director_ai.core.retrieval.doc_registry import DocRecord


def _record(doc_id: str, *, source: str, tenant_id: str = "t1") -> DocRecord:
    return DocRecord(
        doc_id=doc_id,
        source=source,
        tenant_id=tenant_id,
        created_at=0.0,
        updated_at=0.0,
        chunk_count=1,
        chunk_ids=[f"{doc_id}:c0"],
        content_hash=f"hash-{doc_id}",
    )


def _evaluate(policy: KnowledgeSupersessionPolicy, **kwargs) -> SupersessionDecision:
    base = {
        "incoming_doc_id": "new",
        "incoming_source": "new.md",
        "tenant_id": "t1",
        "existing": [],
    }
    base.update(kwargs)
    return policy.evaluate(**base)


class TestCandidateDetection:
    def test_no_existing_yields_none(self):
        decision = _evaluate(KnowledgeSupersessionPolicy())
        assert decision.action == "none"
        assert not decision.has_candidates
        assert decision.requires_human_approval is False

    def test_explicit_hint_by_doc_id(self):
        decision = _evaluate(
            KnowledgeSupersessionPolicy(),
            existing=[_record("old", source="old.md")],
            explicit_supersedes=["old"],
        )
        assert decision.superseded_doc_ids == ("old",)
        candidate = decision.candidates[0]
        assert candidate.reason == "explicit_supersedes"
        assert candidate.score == 1.0
        assert candidate.evidence_ref == "doc://t1/old"

    def test_explicit_hint_by_source(self):
        decision = _evaluate(
            KnowledgeSupersessionPolicy(),
            existing=[_record("old", source="legacy.md")],
            explicit_supersedes=["legacy.md"],
        )
        assert decision.superseded_doc_ids == ("old",)
        assert decision.candidates[0].reason == "explicit_supersedes"

    def test_same_source_revision(self):
        decision = _evaluate(
            KnowledgeSupersessionPolicy(),
            incoming_source="manual.md",
            existing=[_record("old", source="manual.md")],
        )
        candidate = decision.candidates[0]
        assert candidate.reason == "same_source_revision"
        assert candidate.score == 0.9

    def test_contradiction_above_threshold(self):
        decision = _evaluate(
            KnowledgeSupersessionPolicy(min_contradiction_score=0.6),
            existing=[_record("old", source="other.md")],
            contradiction_scores={"old": 0.8},
        )
        candidate = decision.candidates[0]
        assert candidate.reason == "contradiction"
        assert candidate.score == 0.8

    def test_unrelated_record_is_not_a_candidate(self):
        decision = _evaluate(
            KnowledgeSupersessionPolicy(),
            incoming_source="manual.md",
            existing=[_record("other", source="unrelated.md")],
            contradiction_scores={},
        )
        assert decision.action == "none"

    def test_contradiction_below_threshold_ignored(self):
        decision = _evaluate(
            KnowledgeSupersessionPolicy(min_contradiction_score=0.6),
            existing=[_record("old", source="other.md")],
            contradiction_scores={"old": 0.4},
        )
        assert decision.action == "none"

    def test_incoming_excluded_from_candidates(self):
        decision = _evaluate(
            KnowledgeSupersessionPolicy(),
            incoming_doc_id="new",
            incoming_source="manual.md",
            existing=[_record("new", source="manual.md")],
        )
        assert decision.action == "none"

    def test_candidates_sorted_by_score_desc(self):
        decision = _evaluate(
            KnowledgeSupersessionPolicy(min_contradiction_score=0.5),
            incoming_source="manual.md",
            existing=[
                _record("same", source="manual.md"),  # 0.9
                _record("contra", source="other.md"),  # 0.7
                _record("explicit", source="x.md"),  # 1.0
            ],
            explicit_supersedes=["explicit"],
            contradiction_scores={"contra": 0.7},
        )
        scores = [candidate.score for candidate in decision.candidates]
        assert scores == sorted(scores, reverse=True)
        assert decision.candidates[0].superseded_doc_id == "explicit"


class TestGating:
    def test_default_requires_human_approval(self):
        decision = _evaluate(
            KnowledgeSupersessionPolicy(),
            existing=[_record("old", source="old.md")],
            explicit_supersedes=["old"],
        )
        assert decision.action == "recommend"
        assert decision.requires_human_approval is True

    def test_auto_promote_all_above_threshold(self):
        decision = _evaluate(
            KnowledgeSupersessionPolicy(auto_promote=True, auto_promote_threshold=0.95),
            existing=[_record("old", source="old.md")],
            explicit_supersedes=["old"],  # score 1.0 >= 0.95
        )
        assert decision.action == "promote"
        assert decision.requires_human_approval is False

    def test_auto_promote_withheld_when_one_below_threshold(self):
        decision = _evaluate(
            KnowledgeSupersessionPolicy(
                auto_promote=True,
                auto_promote_threshold=0.95,
                min_contradiction_score=0.5,
            ),
            incoming_source="manual.md",
            existing=[
                _record("same", source="manual.md"),  # 0.9 < 0.95
                _record("explicit", source="x.md"),  # 1.0
            ],
            explicit_supersedes=["explicit"],
        )
        assert decision.action == "recommend"
        assert decision.requires_human_approval is True


class TestValidation:
    def test_min_score_out_of_range(self):
        with pytest.raises(ValueError, match="min_contradiction_score"):
            KnowledgeSupersessionPolicy(min_contradiction_score=1.5)

    def test_auto_threshold_below_min(self):
        with pytest.raises(ValueError, match="auto_promote_threshold must be >="):
            KnowledgeSupersessionPolicy(
                min_contradiction_score=0.8, auto_promote_threshold=0.5
            )

    def test_contradiction_score_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="contradiction_score"):
            _evaluate(
                KnowledgeSupersessionPolicy(),
                existing=[_record("old", source="o.md")],
                contradiction_scores={"old": 2.0},
            )

    def test_candidate_empty_id_rejected(self):
        with pytest.raises(ValueError, match="superseded_doc_id"):
            SupersessionCandidate(
                superseded_doc_id="", reason="x", score=0.5, evidence_ref="r"
            )

    def test_candidate_score_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="score"):
            SupersessionCandidate(
                superseded_doc_id="d", reason="x", score=1.2, evidence_ref="r"
            )
