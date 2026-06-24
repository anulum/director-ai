# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — recall correctness label derived from a guarded verdict

"""Turn a Director-AI verdict into a recall *correctness* label.

REMANENTIA's recall ledger carries two labels per recalled grounding, and they
are not interchangeable:

* ``was_used`` — did the downstream consumer actually use the recalled memory?
  Auto-derived by REMANENTIA's loop-closure heuristic. A legitimate cold-start
  and ranking signal: it tells you whether a memory is *reached for*.
* ``was_correct`` — did the recalled grounding actually yield a correct,
  non-hallucinated answer? This is the label a production conformal abstention
  gate must calibrate on. Calibrating coverage on *usage* instead of
  *correctness* gives a guarantee about the wrong event — the "calibration
  theatre" the abstention design exists to rule out
  (see :mod:`director_ai.core.calibration.miscoverage` and
  :mod:`director_ai.core.calibration.adaptive_conformal`).

Director-AI already produces a correctness judgement on every guarded answer:
the VerifiedScorer / contradiction-halt verdict on the emitted text against its
retrieved grounding. That verdict *is* the auto-derived correctness label — no
human annotation closes the loop, the guard does. The mapping is deliberately
strict and fail-closed:

* a clean emit means the grounding held → ``was_correct=True``;
* a halt, a high-confidence contradiction/fabrication, or a VerifiedScorer
  downgrade means the grounding did not hold → ``was_correct=False``.

This module is the *pure* derivation only: it maps each verdict surface
(:class:`~director_ai.core.types.ReviewResult`,
:class:`~director_ai.core.scoring.verified_scorer.VerificationResult`,
:class:`~director_ai.core.runtime.contradiction_halt.ContradictionHaltDecision`,
or an already-derived ``bool``) to that label. The transport that records the
label into REMANENTIA's ``recall_correctness`` seam and the cold-start reader
that primes the conformal predictors from the shared ledger build on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch

from ..runtime.contradiction_halt import ContradictionHaltDecision
from ..scoring.verified_scorer import VerificationResult
from ..types import ReviewResult

__all__ = [
    "RecallOutcome",
    "correctness_from_verdict",
    "recall_outcome",
]

#: Default attribution recorded in the ``by`` field of a correctness outcome.
#: Identifies Director-AI's verification gate as the origin of the label, so the
#: ledger keeps provenance distinct from human or other-system annotations.
DEFAULT_ATTRIBUTION = "director-ai/verified-scorer"


@dataclass(frozen=True)
class RecallOutcome:
    """A correctness label for one recalled grounding, ready for the seam.

    Mirrors the argument shape of REMANENTIA's
    ``handle_recall_correctness(query, was_correct, by)``: the recall
    ``query`` it pertains to, the derived ``was_correct`` boolean, and ``by``
    — the provenance string identifying which gate produced the label. It
    carries no usage signal: ``was_used`` is REMANENTIA's to populate and must
    not be inferred here.
    """

    query: str
    was_correct: bool
    by: str = DEFAULT_ATTRIBUTION


@singledispatch
def correctness_from_verdict(verdict: object) -> bool:
    """Derive ``was_correct`` from a Director-AI verdict (fail-closed).

    Dispatches on the verdict type. Unknown types raise rather than guess a
    label, because a wrong correctness label silently corrupts the conformal
    calibration set — there is no safe default for an unrecognised verdict.
    """
    raise TypeError(
        f"cannot derive a correctness label from {type(verdict).__name__}; "
        "expected ReviewResult, VerificationResult, ContradictionHaltDecision, "
        "or bool"
    )


@correctness_from_verdict.register
def _correctness_from_review(verdict: ReviewResult) -> bool:
    """Treat a full pipeline review as correct only on a clean emit.

    A halt is a refusal — the grounding contradicted the answer — so it is not
    correct. An emitted answer that the VerifiedScorer second gate downgraded
    (``verified_approved is False``) is also not correct even though it was
    emitted: the atomic verifier found a contradicted or fabricated claim.
    Everything else — emitted and not downgraded — is correct.

    ``verified_approved`` is read with ``getattr`` because the score may be the
    Rust-backed ``CoherenceScore``, which does not expose the verified-gate
    fields; a missing attribute (and a ``None`` coherence) reads as not
    downgraded, so only the halt decides.
    """
    if verdict.halted:
        return False
    downgraded = getattr(verdict.coherence, "verified_approved", None) is False
    return not downgraded


@correctness_from_verdict.register
def _correctness_from_verification(verdict: VerificationResult) -> bool:
    """Use the atomic VerifiedScorer's own approval as the label."""
    return verdict.approved


@correctness_from_verdict.register
def _correctness_from_halt(verdict: ContradictionHaltDecision) -> bool:
    """Mark a contradiction halt incorrect; no halt means the claim held."""
    return not verdict.halt


@correctness_from_verdict.register
def _correctness_from_bool(verdict: bool) -> bool:
    """Pass an already-derived correctness label through unchanged."""
    return verdict


def recall_outcome(
    query: str,
    verdict: ReviewResult | VerificationResult | ContradictionHaltDecision | bool,
    *,
    by: str = DEFAULT_ATTRIBUTION,
) -> RecallOutcome:
    """Bundle a recall ``query`` and a verdict into a :class:`RecallOutcome`.

    Convenience over :func:`correctness_from_verdict`: derives the label and
    packages it with the query and provenance for the transport seam. ``query``
    must be the non-empty recall query the grounding answered; an empty query
    cannot be matched back to a ledger record and is rejected.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty recall query")
    return RecallOutcome(
        query=query,
        was_correct=correctness_from_verdict(verdict),
        by=by,
    )
