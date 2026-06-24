# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the verdict → recall-correctness label derivation.

Covers every verdict surface the seam accepts — ReviewResult (clean emit, halt,
and a VerifiedScorer second-gate downgrade), VerificationResult, the
contradiction-halt decision, and a pre-derived bool — plus the unknown-type
fail-closed guard, the RecallOutcome packaging, the empty-query rejection, and
custom provenance attribution. The correctness semantics are asserted directly:
a clean emit is correct, every refusal/downgrade/contradiction is not, and usage
is never inferred (RecallOutcome carries no was_used field).
"""

from __future__ import annotations

import pytest

from director_ai.core.calibration.recall_correctness import (
    DEFAULT_ATTRIBUTION,
    RecallOutcome,
    correctness_from_verdict,
    recall_outcome,
)
from director_ai.core.runtime.contradiction_halt import ContradictionHaltDecision
from director_ai.core.scoring.verified_scorer import VerificationResult
from director_ai.core.types import CoherenceScore, ReviewResult


def _coherence(*, verified_approved: bool | None) -> CoherenceScore:
    """Build a minimal CoherenceScore carrying only the verified-gate flag."""
    return CoherenceScore(
        score=0.9,
        approved=True,
        h_logical=0.1,
        h_factual=0.1,
        verified_approved=verified_approved,
    )


def _review(*, halted: bool, coherence: CoherenceScore | None) -> ReviewResult:
    """Build a ReviewResult with the fields the derivation reads."""
    return ReviewResult(
        output="" if halted else "an emitted answer",
        coherence=coherence,
        halted=halted,
        candidates_evaluated=1,
    )


# --- ReviewResult surface ---------------------------------------------------


def test_review_clean_emit_is_correct() -> None:
    """An emitted answer with no verified-gate downgrade is correct."""
    verdict = _review(halted=False, coherence=_coherence(verified_approved=True))
    assert correctness_from_verdict(verdict) is True


def test_review_clean_emit_without_coherence_is_correct() -> None:
    """A non-halted review with no coherence payload is still correct."""
    verdict = _review(halted=False, coherence=None)
    assert correctness_from_verdict(verdict) is True


def test_review_clean_emit_with_unevaluated_verified_gate_is_correct() -> None:
    """``verified_approved is None`` (gate not run) does not mark incorrect."""
    verdict = _review(halted=False, coherence=_coherence(verified_approved=None))
    assert correctness_from_verdict(verdict) is True


def test_review_halt_is_incorrect() -> None:
    """A halt is a refusal — the grounding contradicted the answer."""
    verdict = _review(halted=True, coherence=None)
    assert correctness_from_verdict(verdict) is False


def test_review_verified_downgrade_is_incorrect() -> None:
    """An emitted answer the VerifiedScorer downgraded is not correct."""
    verdict = _review(halted=False, coherence=_coherence(verified_approved=False))
    assert correctness_from_verdict(verdict) is False


# --- VerificationResult surface ---------------------------------------------


@pytest.mark.parametrize("approved", [True, False])
def test_verification_result_uses_approval(approved: bool) -> None:
    """The atomic VerifiedScorer's own approval is the label."""
    verdict = VerificationResult(
        approved=approved, overall_score=0.8, confidence="high"
    )
    assert correctness_from_verdict(verdict) is approved


# --- ContradictionHaltDecision surface --------------------------------------


def test_halt_decision_halt_is_incorrect() -> None:
    """A contradiction halt means the claim did not hold."""
    verdict = ContradictionHaltDecision(halt=True, contradiction=0.9, fact="f")
    assert correctness_from_verdict(verdict) is False


def test_halt_decision_no_halt_is_correct() -> None:
    """No contradiction halt means the claim held against grounding."""
    verdict = ContradictionHaltDecision(halt=False, contradiction=0.1)
    assert correctness_from_verdict(verdict) is True


# --- bool passthrough + unknown guard ---------------------------------------


@pytest.mark.parametrize("label", [True, False])
def test_bool_passthrough(label: bool) -> None:
    """An already-derived boolean passes through unchanged."""
    assert correctness_from_verdict(label) is label


def test_unknown_verdict_type_raises() -> None:
    """An unrecognised verdict fails closed rather than guessing a label."""
    with pytest.raises(TypeError, match="cannot derive a correctness label"):
        correctness_from_verdict("not a verdict")


# --- recall_outcome packaging -----------------------------------------------


def test_recall_outcome_packages_query_and_label() -> None:
    """recall_outcome derives the label and bundles it with the query."""
    verdict = _review(halted=True, coherence=None)
    outcome = recall_outcome("what is the capital of France?", verdict)
    assert outcome == RecallOutcome(
        query="what is the capital of France?",
        was_correct=False,
        by=DEFAULT_ATTRIBUTION,
    )


def test_recall_outcome_custom_attribution() -> None:
    """A caller may stamp its own provenance into the ``by`` field."""
    outcome = recall_outcome("q", True, by="director-ai/contradiction-halt")
    assert outcome.was_correct is True
    assert outcome.by == "director-ai/contradiction-halt"


def test_recall_outcome_has_no_usage_field() -> None:
    """RecallOutcome carries correctness only; was_used is REMANENTIA's."""
    outcome = recall_outcome("q", True)
    assert not hasattr(outcome, "was_used")


@pytest.mark.parametrize("bad_query", ["", "   ", "\t\n"])
def test_recall_outcome_rejects_blank_query(bad_query: str) -> None:
    """A blank query cannot be matched to a ledger record and is rejected."""
    with pytest.raises(ValueError, match="non-empty recall query"):
        recall_outcome(bad_query, True)


def test_recall_outcome_is_frozen() -> None:
    """The outcome is immutable once derived."""
    outcome = recall_outcome("q", True)
    with pytest.raises(AttributeError):
        outcome.was_correct = False  # type: ignore[misc]


def test_public_surface_reexports_derivation() -> None:
    """The derivation is reachable from the core package surface."""
    from director_ai.core import (
        RecallOutcome as ExportedOutcome,
    )
    from director_ai.core import (
        correctness_from_verdict as exported_fn,
    )
    from director_ai.core import (
        recall_outcome as exported_outcome,
    )

    assert ExportedOutcome is RecallOutcome
    assert exported_fn is correctness_from_verdict
    assert exported_outcome is recall_outcome
