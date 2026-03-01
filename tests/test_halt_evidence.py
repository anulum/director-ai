# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — HaltEvidence Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
from __future__ import annotations

from director_ai.core.types import EvidenceChunk, HaltEvidence, ReviewResult


def test_halt_evidence_dataclass():
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


def test_halt_evidence_defaults():
    ev = HaltEvidence(reason="test", last_score=0.0, evidence_chunks=[])
    assert ev.nli_scores is None
    assert ev.suggested_action == ""


def test_review_result_has_halt_evidence():
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


def test_review_result_halt_evidence_none_by_default():
    result = ReviewResult(
        output="ok", coherence=None, halted=False, candidates_evaluated=1
    )
    assert result.halt_evidence is None


def test_agent_populates_halt_evidence():
    from director_ai.core.agent import CoherenceAgent

    agent = CoherenceAgent()
    result = agent.process("Tell me about quantum gravity in 17 dimensions")
    if result.halted:
        assert result.halt_evidence is not None
        assert result.halt_evidence.reason == "all_candidates_rejected"
        assert isinstance(result.halt_evidence.evidence_chunks, list)
        assert result.halt_evidence.suggested_action != ""
