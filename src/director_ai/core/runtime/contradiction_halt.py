# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — contradiction-driven streaming halt

"""Halt a stream when a completed claim contradicts the retrieved grounding.

This is the working streaming-halt mechanism. The coherence-based halt failed
because its score folds "unsupported" into "divergence" and false-halts correct
text; this fires only when a claim's ``P(contradiction)`` against the retrieved
knowledge-base facts crosses a threshold. A correct-but-unsupported claim is
neutral, not a contradiction, so it does not halt; only a claim that the model
judges to *contradict* governed facts does.

The claim is scored against each retrieved fact and the strongest contradiction
decides — a claim that contradicts any one governed fact halts. When retrieval
returns nothing the claim is ungrounded and is allowed (an absent fact is not a
contradiction). Pair with :class:`StreamingCoherenceGate` so only complete
claims, not half-finished sentences, reach the scorer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..scoring.contradiction import ContradictionScorer

__all__ = ["ContradictionHaltDecision", "ContradictionHalt"]

# A retriever maps a claim to its grounding: a list of facts, a single
# semicolon-joined context string (the GroundTruthStore format), or None.
Retriever = Callable[[str], "str | Sequence[str] | None"]


@dataclass(frozen=True)
class ContradictionHaltDecision:
    """Outcome of checking one claim against its grounding."""

    halt: bool
    contradiction: float
    fact: str = ""


def _as_facts(grounding: str | Sequence[str] | None) -> list[str]:
    if grounding is None:
        return []
    if isinstance(grounding, str):
        return [part.strip() for part in grounding.split(";") if part.strip()]
    return [str(f).strip() for f in grounding if str(f).strip()]


class ContradictionHalt:
    """Decide whether a completed claim should halt the stream.

    Parameters
    ----------
    scorer:
        A :class:`ContradictionScorer` exposing ``contradiction_batch``.
    retrieve:
        ``retrieve(claim)`` returning the grounding facts for the claim
        (typically ``GroundTruthStore.retrieve_context``).
    threshold:
        ``P(contradiction)`` at or above which to halt. Defaults to the
        scorer's own threshold.
    """

    def __init__(
        self,
        scorer: ContradictionScorer,
        retrieve: Retriever,
        *,
        threshold: float | None = None,
    ) -> None:
        self._scorer = scorer
        self._retrieve = retrieve
        self._threshold = (
            float(threshold) if threshold is not None else scorer.threshold
        )

    def should_halt(self, claim: str) -> ContradictionHaltDecision:
        """Halt when *claim* contradicts any retrieved grounding fact."""
        if not claim or not claim.strip():
            return ContradictionHaltDecision(halt=False, contradiction=0.0)
        facts = _as_facts(self._retrieve(claim))
        if not facts:
            # Ungrounded claim: not a contradiction, so never halt on it here.
            return ContradictionHaltDecision(halt=False, contradiction=0.0)
        scores = self._scorer.contradiction_batch([(f, claim) for f in facts])
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best = scores[best_idx]
        return ContradictionHaltDecision(
            halt=best >= self._threshold,
            contradiction=best,
            fact=facts[best_idx] if best >= self._threshold else "",
        )

    @property
    def threshold(self) -> float:
        return self._threshold
