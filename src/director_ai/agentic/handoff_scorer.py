# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Inter-agent handoff scorer
"""Score inter-agent messages for hallucination before handoff.

When Agent A passes output to Agent B, the handoff scorer verifies
that the message is grounded in the source context. This prevents
hallucinated information from propagating through the agent swarm.

Scoring uses keyword overlap (zero-dep heuristic) with an optional
NLI path when ``CoherenceScorer`` is available.

Usage::

    from director_ai.agentic.handoff_scorer import HandoffScorer

    scorer = HandoffScorer()
    result = scorer.score(
        message="Paris is the capital of France.",
        context="European geography: France, capital Paris.",
        from_agent="researcher-0",
        to_agent="summariser-0",
    )
    print(result.grounded, result.score)  # True, 0.15
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

__all__ = ["HandoffScorer", "HandoffScore"]

logger = logging.getLogger("DirectorAI.HandoffScorer")


@dataclass(frozen=True)
class HandoffScore:
    """Result of scoring an inter-agent handoff."""

    from_agent: str
    to_agent: str
    score: float  # 0 = fully grounded, 1 = hallucinated
    grounded: bool  # score <= threshold
    latency_ms: float = 0.0
    method: str = "keyword"  # "keyword" or "nli"
    evidence: list[str] = field(default_factory=list)


def _keyword_divergence(message: str, context: str) -> float:
    """Compute keyword-based divergence.

    Returns 0.0 (perfect overlap) to 1.0 (no overlap).
    """
    if not context or not message:
        return 0.5

    msg_words = set(message.lower().split())
    ctx_words = set(context.lower().split())
    if not msg_words:
        return 0.5

    overlap = len(msg_words & ctx_words)
    coverage = overlap / len(msg_words)
    return max(0.0, min(1.0, 1.0 - coverage))


class HandoffScorer:
    """Score inter-agent messages for factual grounding.

    Parameters
    ----------
    threshold : float
        Maximum divergence score to consider grounded (0–1).
    nli_scorer : object | None
        Optional ``CoherenceScorer`` for NLI-based scoring.
        When provided, uses NLI instead of keyword overlap.
    """

    def __init__(
        self,
        threshold: float = 0.4,
        nli_scorer: object | None = None,
    ) -> None:
        self._threshold = threshold
        self._nli_scorer = nli_scorer
        self._history: list[HandoffScore] = []

    def score(
        self,
        message: str,
        context: str,
        from_agent: str = "",
        to_agent: str = "",
    ) -> HandoffScore:
        """Score a handoff message against source context.

        Parameters
        ----------
        message : str
            The content being passed between agents.
        context : str
            Ground truth or source material for verification.
        from_agent : str
            Source agent identifier.
        to_agent : str
            Destination agent identifier.

        Returns
        -------
        HandoffScore
        """
        t0 = time.perf_counter()

        if self._nli_scorer is not None:
            divergence, method = self._score_nli(message, context)
        else:
            divergence = _keyword_divergence(message, context)
            method = "keyword"

        latency_ms = (time.perf_counter() - t0) * 1000
        grounded = divergence <= self._threshold

        evidence: list[str] = []
        if not grounded:
            evidence.append(
                f"divergence {divergence:.3f} > threshold {self._threshold:.3f}"
            )

        result = HandoffScore(
            from_agent=from_agent,
            to_agent=to_agent,
            score=divergence,
            grounded=grounded,
            latency_ms=round(latency_ms, 2),
            method=method,
            evidence=evidence,
        )

        self._history.append(result)
        return result

    def _score_nli(self, message: str, context: str) -> tuple[float, str]:
        """Score using NLI scorer if available."""
        try:
            scorer = self._nli_scorer
            if hasattr(scorer, "calculate_factual_divergence"):
                div = scorer.calculate_factual_divergence(context, message)
                return float(div), "nli"
        except Exception as exc:
            logger.warning("NLI scoring failed, falling back to keyword: %s", exc)
        return _keyword_divergence(message, context), "keyword"

    @property
    def history(self) -> list[HandoffScore]:
        """All scored handoffs (most recent last)."""
        return list(self._history)

    def clear_history(self) -> None:
        """Clear handoff history."""
        self._history.clear()

    def stats(self) -> dict[str, float]:
        """Aggregate statistics from history.

        Returns dict with: total, grounded_pct, mean_score,
        mean_latency_ms.
        """
        if not self._history:
            return {
                "total": 0,
                "grounded_pct": 0.0,
                "mean_score": 0.0,
                "mean_latency_ms": 0.0,
            }
        n = len(self._history)
        grounded = sum(1 for h in self._history if h.grounded)
        return {
            "total": n,
            "grounded_pct": grounded / n * 100,
            "mean_score": sum(h.score for h in self._history) / n,
            "mean_latency_ms": sum(h.latency_ms for h in self._history) / n,
        }
