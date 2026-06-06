# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CredibilityFeedbackLoop tests

"""Tests for online credibility learning from human feedback.

Covers approval/rejection signal direction, distinct/blank source handling,
correction adapters and batch ingestion, the relevance-vs-credibility
re-rank with its weight extremes, parameter validation, and the shared-tracker
wiring that makes a ProvenanceVerifier's trust score learn from feedback."""

from __future__ import annotations

import pytest

from director_ai.core.calibration.feedback_store import Correction
from director_ai.core.provenance import (
    CitationFact,
    CredibilityFeedbackLoop,
    ProvenanceChain,
    ProvenanceVerifier,
    SourceCredibility,
)
from director_ai.core.types import EvidenceChunk

_SECRET = b"director-ai-credibility-feedback-secret-key"


def _const_credibility() -> SourceCredibility:
    """A tracker with a frozen clock — first observation is decay-free."""
    return SourceCredibility(clock=lambda: 0.0)


def _correction(*, human_approved: bool, review_id: str = "r") -> Correction:
    return Correction(
        prompt="q",
        response="a",
        guardrail_score=0.0,
        guardrail_approved=True,
        human_approved=human_approved,
        timestamp=0.0,
        review_id=review_id,
    )


class TestObserve:
    def test_approval_raises_above_prior(self):
        loop = CredibilityFeedbackLoop(credibility=_const_credibility())
        (score,) = loop.observe(source_ids=["wiki"], human_approved=True)
        assert score.source_id == "wiki"
        assert score.score == pytest.approx(0.75)

    def test_rejection_falls_below_prior(self):
        loop = CredibilityFeedbackLoop(credibility=_const_credibility())
        (score,) = loop.observe(source_ids=["spam"], human_approved=False)
        assert score.score == pytest.approx(0.25)

    def test_distinct_sources_each_observed_once(self):
        loop = CredibilityFeedbackLoop(credibility=_const_credibility())
        scores = loop.observe(source_ids=["a", "b", "a"], human_approved=True)
        assert {s.source_id for s in scores} == {"a", "b"}
        assert len(scores) == 2

    def test_blank_sources_ignored(self):
        loop = CredibilityFeedbackLoop(credibility=_const_credibility())
        assert loop.observe(source_ids=["", "  ".strip()], human_approved=True) == ()

    def test_repeated_rejection_drifts_down(self):
        clock_state = {"t": 0.0}

        def clock() -> float:
            now = clock_state["t"]
            clock_state["t"] += 1.0
            return now

        credibility = SourceCredibility(half_life_seconds=1.0, clock=clock)
        loop = CredibilityFeedbackLoop(credibility=credibility)
        first = loop.observe(source_ids=["spam"], human_approved=False)[0].score
        second = loop.observe(source_ids=["spam"], human_approved=False)[0].score
        assert second < first


class TestCorrections:
    def test_observe_correction_uses_human_verdict(self):
        loop = CredibilityFeedbackLoop(credibility=_const_credibility())
        (score,) = loop.observe_correction(
            _correction(human_approved=False), source_ids=["src"]
        )
        assert score.score == pytest.approx(0.25)

    def test_ingest_corrections_counts_applied(self):
        loop = CredibilityFeedbackLoop(credibility=_const_credibility())
        corrections = [
            _correction(human_approved=True, review_id="r0"),
            _correction(human_approved=False, review_id="r1"),
            _correction(human_approved=True, review_id="r2"),
        ]
        sources = {"r0": ["a"], "r1": ["b"], "r2": []}
        applied = loop.ingest_corrections(
            corrections,
            source_resolver=lambda c: sources[c.review_id],
        )
        assert applied == 2  # r2 resolves to no source and is skipped

    def test_credibility_of_passthrough(self):
        loop = CredibilityFeedbackLoop(credibility=_const_credibility())
        loop.observe(source_ids=["src"], human_approved=True)
        assert loop.credibility_of("src") == pytest.approx(0.75)
        assert loop.credibility_of("unknown") == pytest.approx(0.5)


class TestRerank:
    def _loop_with_two_sources(self) -> CredibilityFeedbackLoop:
        loop = CredibilityFeedbackLoop(credibility=_const_credibility())
        loop.observe(source_ids=["high"], human_approved=True)  # 0.75
        loop.observe(source_ids=["low"], human_approved=False)  # 0.25
        return loop

    def _chunks(self) -> list[EvidenceChunk]:
        return [
            EvidenceChunk(text="closer", distance=0.1, source="low"),
            EvidenceChunk(text="farther", distance=0.5, source="high"),
        ]

    def test_weight_zero_is_pure_relevance(self):
        loop = self._loop_with_two_sources()
        ranked = loop.rerank(self._chunks(), credibility_weight=0.0)
        assert [c.text for c in ranked] == ["closer", "farther"]

    def test_weight_one_is_pure_credibility(self):
        loop = self._loop_with_two_sources()
        ranked = loop.rerank(self._chunks(), credibility_weight=1.0)
        assert [c.text for c in ranked] == ["farther", "closer"]

    def test_blend_lets_credibility_outrank_relevance(self):
        loop = self._loop_with_two_sources()
        ranked = loop.rerank(self._chunks(), credibility_weight=0.5)
        assert ranked[0].source == "high"

    def test_empty_chunks(self):
        loop = self._loop_with_two_sources()
        assert loop.rerank([]) == []

    def test_unsourced_chunk_uses_prior(self):
        loop = self._loop_with_two_sources()
        chunks = [
            EvidenceChunk(text="anon", distance=0.4, source=""),
            EvidenceChunk(text="trusted", distance=0.4, source="high"),
        ]
        ranked = loop.rerank(chunks, credibility_weight=1.0)
        # Equal distance, so credibility decides: high (0.75) > prior (0.5).
        assert ranked[0].source == "high"


class TestValidation:
    def test_bad_approve_signal(self):
        with pytest.raises(ValueError, match="approve_signal"):
            CredibilityFeedbackLoop(
                credibility=_const_credibility(), approve_signal=2.0
            )

    def test_bad_reject_signal(self):
        with pytest.raises(ValueError, match="reject_signal"):
            CredibilityFeedbackLoop(
                credibility=_const_credibility(), reject_signal=-0.1
            )

    def test_bad_rerank_weight(self):
        loop = CredibilityFeedbackLoop(credibility=_const_credibility())
        with pytest.raises(ValueError, match="credibility_weight"):
            loop.rerank([], credibility_weight=1.5)


class TestVerifierWiring:
    def test_feedback_lowers_verifier_trust_score(self):
        credibility = _const_credibility()
        loop = CredibilityFeedbackLoop(credibility=credibility)
        loop.observe(source_ids=["spam"], human_approved=False)

        verifier = ProvenanceVerifier(
            chain=ProvenanceChain(secret=_SECRET),
            credibility=credibility,
        )
        rejected = verifier.verify(
            [CitationFact(source_id="spam", content="claim", timestamp=0.0)]
        )
        trusted = verifier.verify(
            [CitationFact(source_id="fresh", content="claim", timestamp=0.0)]
        )
        # The rejected source's learned credibility (0.25) is consumed by the
        # verifier and sits below a never-seen source's prior (0.5).
        assert rejected.trust_score == pytest.approx(0.25)
        assert rejected.trust_score < trusted.trust_score
