# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — HaltEvidence Tests
"""Multi-angle tests for HaltEvidence and ReviewResult dataclasses.

Covers: field presence, defaults, evidence chunks, NLI scores,
suggested action, ReviewResult integration, agent pipeline halt
evidence population, parametrised construction, and pipeline
performance documentation.
"""

from __future__ import annotations

import pytest

from director_ai.core.types import EvidenceChunk, HaltEvidence, ReviewResult

# ── HaltEvidence construction ────────────────────────────────────


class TestHaltEvidenceConstruction:
    """HaltEvidence must store all fields correctly."""

    def test_full_construction(self):
        ev = HaltEvidence(
            reason="all_candidates_rejected",
            last_score=0.32,
            evidence_chunks=[
                EvidenceChunk(text="sky is blue", distance=0.1, source="builtin"),
            ],
            nli_scores=[0.85],
            suggested_action="Add facts.",
        )
        assert ev.reason == "all_candidates_rejected"
        assert ev.last_score == 0.32
        assert len(ev.evidence_chunks) == 1
        assert ev.nli_scores == [0.85]
        assert ev.suggested_action == "Add facts."

    def test_defaults(self):
        ev = HaltEvidence(reason="test", last_score=0.0, evidence_chunks=[])
        assert ev.nli_scores is None
        assert ev.suggested_action == ""

    @pytest.mark.parametrize(
        "reason",
        [
            "all_candidates_rejected",
            "coherence_below_threshold",
            "nli_divergence_high",
            "streaming_halt",
        ],
    )
    def test_various_reasons(self, reason):
        ev = HaltEvidence(reason=reason, last_score=0.0, evidence_chunks=[])
        assert ev.reason == reason

    @pytest.mark.parametrize("score", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_various_last_scores(self, score):
        ev = HaltEvidence(reason="test", last_score=score, evidence_chunks=[])
        assert ev.last_score == score

    def test_multiple_evidence_chunks(self):
        chunks = [
            EvidenceChunk(text=f"chunk {i}", distance=i * 0.1, source="test")
            for i in range(5)
        ]
        ev = HaltEvidence(reason="test", last_score=0.3, evidence_chunks=chunks)
        assert len(ev.evidence_chunks) == 5

    def test_multiple_nli_scores(self):
        ev = HaltEvidence(
            reason="test",
            last_score=0.3,
            evidence_chunks=[],
            nli_scores=[0.1, 0.5, 0.9],
        )
        assert len(ev.nli_scores) == 3


# ── EvidenceChunk ────────────────────────────────────────────────


class TestEvidenceChunk:
    """EvidenceChunk must store text, distance, and source."""

    def test_basic_construction(self):
        chunk = EvidenceChunk(text="sky is blue", distance=0.1, source="builtin")
        assert chunk.text == "sky is blue"
        assert chunk.distance == 0.1
        assert chunk.source == "builtin"

    @pytest.mark.parametrize("distance", [0.0, 0.01, 0.5, 0.99, 1.0])
    def test_various_distances(self, distance):
        chunk = EvidenceChunk(text="test", distance=distance, source="test")
        assert chunk.distance == distance

    def test_empty_text(self):
        chunk = EvidenceChunk(text="", distance=0.0, source="test")
        assert chunk.text == ""


# ── ReviewResult integration ─────────────────────────────────────


class TestReviewResultHaltEvidence:
    """ReviewResult must carry HaltEvidence through pipeline."""

    def test_halt_evidence_present_on_halt(self):
        ev = HaltEvidence(reason="test", last_score=0.4, evidence_chunks=[])
        result = ReviewResult(
            output="[HALT]",
            coherence=None,
            halted=True,
            candidates_evaluated=3,
            halt_evidence=ev,
        )
        assert result.halt_evidence is not None
        assert result.halt_evidence.reason == "test"

    def test_halt_evidence_none_by_default(self):
        result = ReviewResult(
            output="ok",
            coherence=None,
            halted=False,
            candidates_evaluated=1,
        )
        assert result.halt_evidence is None

    def test_halted_result_has_output(self):
        ev = HaltEvidence(reason="test", last_score=0.0, evidence_chunks=[])
        result = ReviewResult(
            output="[HALT]",
            coherence=None,
            halted=True,
            candidates_evaluated=0,
            halt_evidence=ev,
        )
        assert isinstance(result.output, str)
        assert result.halted is True

    @pytest.mark.parametrize("n_candidates", [0, 1, 3, 5, 10])
    def test_various_candidate_counts(self, n_candidates):
        result = ReviewResult(
            output="test",
            coherence=None,
            halted=False,
            candidates_evaluated=n_candidates,
        )
        assert result.candidates_evaluated == n_candidates


# ── Agent pipeline integration ───────────────────────────────────


class TestAgentHaltEvidence:
    """Agent must populate halt_evidence when halting."""

    def test_agent_populates_halt_evidence(self):
        from director_ai.core.agent import CoherenceAgent

        agent = CoherenceAgent()
        result = agent.process("Tell me about quantum gravity in 17 dimensions")
        if result.halted:
            assert result.halt_evidence is not None
            assert result.halt_evidence.reason == "all_candidates_rejected"
            assert isinstance(result.halt_evidence.evidence_chunks, list)
            assert result.halt_evidence.suggested_action != ""

    def test_agent_non_halted_no_evidence(self):
        from director_ai.core.agent import CoherenceAgent

        agent = CoherenceAgent()
        result = agent.process("What is 2+2?")
        if not result.halted:
            assert result.halt_evidence is None


# ── Pipeline performance ─────────────────────────────────────────


class TestHaltEvidencePerformance:
    """Document halt evidence pipeline characteristics."""

    def test_evidence_chunk_has_required_fields(self):
        chunk = EvidenceChunk(text="t", distance=0.1, source="s")
        assert hasattr(chunk, "text")
        assert hasattr(chunk, "distance")
        assert hasattr(chunk, "source")

    def test_halt_evidence_has_required_fields(self):
        ev = HaltEvidence(reason="r", last_score=0.0, evidence_chunks=[])
        assert hasattr(ev, "reason")
        assert hasattr(ev, "last_score")
        assert hasattr(ev, "evidence_chunks")
        assert hasattr(ev, "nli_scores")
        assert hasattr(ev, "suggested_action")
