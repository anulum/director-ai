# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Adaptive retrieval router
"""Adaptive retrieval router: decide whether a query needs KB lookup.

Factual queries ("What is the refund policy?") benefit from retrieval.
Creative or conversational queries ("Write me a poem") do not — they
waste latency and risk false KB matches.

The router classifies the query using the existing
``detect_task_type()`` heuristic and returns a routing decision.

Usage::

    from director_ai.core.retrieval.adaptive_router import AdaptiveRouter

    router = AdaptiveRouter()
    decision = router.should_retrieve("What is the capital of France?")
    # RoutingDecision(retrieve=True, task_type="qa", confidence=0.9)

    decision = router.should_retrieve("Write a haiku about spring")
    # RoutingDecision(retrieve=False, task_type="dialogue", confidence=0.8)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["AdaptiveRouter", "RoutingDecision"]

# Task types that benefit from KB retrieval
_RETRIEVAL_TYPES = frozenset({"rag", "fact_check", "qa"})

# Task types that do NOT benefit from KB retrieval
_SKIP_TYPES = frozenset({"dialogue", "default"})

# Summarisation is ambiguous — depends on whether KB has source material
_AMBIGUOUS_TYPES = frozenset({"summarization"})

# Heuristic patterns strongly indicating factual need
_FACTUAL_PATTERNS = re.compile(
    r"\b("
    r"what\s+is|who\s+is|when\s+did|where\s+is|how\s+many|how\s+much|"
    r"define|explain|according\s+to|based\s+on|verify|fact[- ]?check|"
    r"is\s+it\s+true|confirm|source|reference|policy|regulation|"
    r"specification|requirement|standard|guideline"
    r")\b",
    re.IGNORECASE,
)

# Heuristic patterns strongly indicating creative/non-factual
_CREATIVE_PATTERNS = re.compile(
    r"\b("
    r"write\s+(?:me\s+)?a|compose|create|imagine|brainstorm|"
    r"suggest|recommend|draft|rewrite|rephrase|paraphrase|"
    r"translate|summarise|tell\s+me\s+a\s+(?:joke|story)|"
    r"poem|story|essay|song|code\s+(?:me|a)|"
    r"hello|hi\b|hey\b|thanks|thank\s+you|bye|goodbye"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RoutingDecision:
    """Result of adaptive routing classification."""

    retrieve: bool
    task_type: str
    confidence: float  # 0–1, how sure the router is


class AdaptiveRouter:
    """Classify queries into retrieve-or-skip decisions.

    Combines the existing ``detect_task_type()`` heuristic with
    additional pattern matching for higher confidence.

    Parameters
    ----------
    factual_threshold : float
        Minimum confidence to trigger retrieval. Below this, retrieval
        is skipped (conservative: only retrieve when confident).
    default_retrieve : bool
        When confidence is ambiguous, default to retrieving or not.
    """

    def __init__(
        self,
        factual_threshold: float = 0.5,
        default_retrieve: bool = True,
    ) -> None:
        self._factual_threshold = factual_threshold
        self._default_retrieve = default_retrieve

    def should_retrieve(self, query: str, response: str = "") -> RoutingDecision:
        """Decide whether this query should trigger KB retrieval.

        Parameters
        ----------
        query : str
            The user's prompt or query.
        response : str
            Optional response text (helps with task type detection).

        Returns
        -------
        RoutingDecision
        """
        # Use existing task type detector
        task_type = _detect_task_type_safe(query, response)

        # Pattern-based confidence boosting
        factual_score = len(_FACTUAL_PATTERNS.findall(query))
        creative_score = len(_CREATIVE_PATTERNS.findall(query))

        # Base confidence from task type
        if task_type in _RETRIEVAL_TYPES:
            base_confidence = 0.8
            retrieve = True
        elif task_type in _SKIP_TYPES:
            base_confidence = 0.7
            retrieve = False
        else:
            # Ambiguous (summarisation, etc.)
            base_confidence = 0.5
            retrieve = self._default_retrieve

        # Adjust confidence based on pattern matches
        if factual_score > 0 and creative_score == 0:
            confidence = min(1.0, base_confidence + 0.15)
            retrieve = True
        elif creative_score > 0 and factual_score == 0:
            confidence = min(1.0, base_confidence + 0.15)
            retrieve = False
        elif factual_score > 0 and creative_score > 0:
            # Conflicting signals — use task type with lower confidence
            confidence = max(0.3, base_confidence - 0.2)
        else:
            confidence = base_confidence

        # Apply threshold
        if retrieve and confidence < self._factual_threshold:
            retrieve = self._default_retrieve

        return RoutingDecision(
            retrieve=retrieve,
            task_type=task_type,
            confidence=round(confidence, 3),
        )


def _detect_task_type_safe(query: str, response: str) -> str:
    """Import and call detect_task_type with fallback."""
    try:
        from director_ai.core.scoring._task_scoring import detect_task_type

        return detect_task_type(query, response)
    except ImportError:
        return "default"
