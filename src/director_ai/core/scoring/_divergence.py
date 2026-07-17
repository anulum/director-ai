# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Divergence Signals (logical scoring + heuristic fallbacks)
"""Logical divergence and heuristic fallbacks for the coherence scorer.

:class:`DivergenceMixin` composes the divergence surface the
:class:`~director_ai.core.scoring.scorer.CoherenceScorer` review pipeline
consumes: the factual calculators live in
:class:`~director_ai.core.scoring._divergence_factual.FactualDivergenceMixin`,
the task routing in
:class:`~director_ai.core.scoring._divergence_routing.TaskRoutedCoherenceMixin`,
and this module adds logical divergence between prompt and response plus the
keyword-heuristic fallbacks (Rust-accelerated when the ``backfire_kernel``
extra is installed) used when no NLI model is available. The mixin owns no
``__init__`` — the composing scorer initialises every attribute declared on
the class body.
"""

from __future__ import annotations

import re

from ..mandatory import mandatory_execution
from ..metrics import metrics
from ..otel import trace_nli_inference
from ._divergence_factual import FactualDivergenceMixin
from ._divergence_routing import TaskRoutedCoherenceMixin
from ._heuristics import (
    DIVERGENCE_ALIGNED as DIVERGENCE_ALIGNED,
)
from ._heuristics import (
    DIVERGENCE_CONTRADICTED as DIVERGENCE_CONTRADICTED,
)
from ._heuristics import (
    DIVERGENCE_NEUTRAL as DIVERGENCE_NEUTRAL,
)

try:
    from backfire_kernel import (
        rust_heuristic_factual_divergence,
        rust_heuristic_logical_divergence,
    )
except Exception:  # pragma: no cover - mandatory accelerator may be unavailable
    rust_heuristic_factual_divergence = None
    rust_heuristic_logical_divergence = None

__all__ = [
    "DIVERGENCE_ALIGNED",
    "DIVERGENCE_CONTRADICTED",
    "DIVERGENCE_NEUTRAL",
    "DivergenceMixin",
]


class DivergenceMixin(TaskRoutedCoherenceMixin, FactualDivergenceMixin):
    """Divergence-signal surface of :class:`CoherenceScorer`.

    Composes the factual calculators and the task-aware routing (see the
    base mixins) and adds logical divergence plus the keyword-heuristic
    fallbacks. All state is initialised by the composing scorer's
    ``__init__``; the base-mixin annotations declare that shared contract
    for static analysis without creating attributes.
    """

    # ── Logical divergence ────────────────────────────────────────────

    def calculate_logical_divergence(
        self,
        prompt: str,
        text_output: str,
        *,
        _inner_agg: str | None = None,
        _outer_agg: str | None = None,
    ) -> float:
        """Compute logical contradiction probability via NLI.

        When strict_mode is True and NLI is unavailable, returns 0.9 (reject).
        """
        if self._rust_scorer is not None:
            _, score_obj = self._rust_scorer.review(prompt, text_output)
            fallback = 1.0 - getattr(score_obj, "score", 0.5)
            return getattr(score_obj, "h_logical", fallback)

        if self._nli and self._nli.model_available:
            logic_inner = (
                _inner_agg if _inner_agg is not None else self._logic_inner_agg
            )
            logic_outer = (
                _outer_agg if _outer_agg is not None else self._logic_outer_agg
            )
            with trace_nli_inference(stage="logical") as nli_span:
                nli_span.set_attribute("nli.model_available", True)
                with metrics.timer("chunked_nli_seconds"):
                    score, _ = self._nli.score_chunked(
                        prompt,
                        text_output,
                        inner_agg=logic_inner,
                        outer_agg=logic_outer,
                        premise_ratio=self._premise_ratio,
                    )
                nli_span.set_attribute("nli.score", score)
            return score

        if self.strict_mode:
            return DIVERGENCE_CONTRADICTED

        self._record_nli_fallback_incident(
            stage="logical",
            reason="nli_unavailable_using_heuristic",
        )
        return self._heuristic_logical(text_output, prompt)

    # ── Heuristic fallbacks (no-NLI keyword scoring) ──────────────────

    @staticmethod
    def _heuristic_factual(context: str, text_output: str) -> float:
        """Word-overlap factual divergence with negation and entity checks.

        A negation polarity flip on near-identical content floors the
        divergence at the contradiction level (KIMI3-negation).
        Install [nli] for production scoring.
        """
        if rust_heuristic_factual_divergence is not None:
            with mandatory_execution(__name__, component="mandatory accelerated path"):
                return float(rust_heuristic_factual_divergence(context, text_output))
        from ._heuristics import (
            ENTITY_RE,
            NEGATION_FLIP_OVERLAP,
            NEGATION_WORDS,
            STOP_WORDS,
        )

        ctx_raw = set(re.findall(r"\w+", context.lower()))
        out_raw = set(re.findall(r"\w+", text_output.lower()))
        ctx_words = ctx_raw - STOP_WORDS
        out_words = out_raw - STOP_WORDS
        if not ctx_words or not out_words:
            return DIVERGENCE_NEUTRAL

        overlap = len(ctx_words & out_words)
        recall = overlap / len(ctx_words)
        precision = overlap / len(out_words)
        similarity = max(recall, precision)
        divergence = 1.0 - similarity

        # Negation asymmetry: check raw words (before stop-word removal)
        ctx_neg = bool(ctx_raw & NEGATION_WORDS)
        out_neg = bool(out_raw & NEGATION_WORDS)
        if ctx_neg != out_neg:
            divergence += 0.25
            # A polarity flip on grounded content is a direct
            # contradiction: when nearly all of the output's content
            # words come from the context, the negation necessarily
            # applies to that shared content. Gate on precision, not
            # recall — an output that covers the context but adds its
            # own negated material may be negating the added material.
            if precision >= NEGATION_FLIP_OVERLAP:
                divergence = max(divergence, DIVERGENCE_CONTRADICTED)

        # Novel entities in output not grounded in context → +0.15
        ctx_ents = set(ENTITY_RE.findall(context))
        out_ents = set(ENTITY_RE.findall(text_output))
        novel_ents = out_ents - ctx_ents
        if novel_ents:
            divergence += 0.15

        return max(0.0, min(1.0, divergence))

    @staticmethod
    def _heuristic_logical(text_output: str, prompt: str = "") -> float:
        """Keyword + word-overlap logical divergence (no-NLI fallback).

        Install [nli] for production-grade scoring.
        """
        if rust_heuristic_logical_divergence is not None:
            with mandatory_execution(__name__, component="mandatory accelerated path"):
                return float(rust_heuristic_logical_divergence(text_output, prompt))
        out = text_output.lower()
        if "consistent with reality" in out:
            return DIVERGENCE_ALIGNED
        if "opposite is true" in out:
            return DIVERGENCE_CONTRADICTED
        if "depends on your perspective" in out:
            return DIVERGENCE_NEUTRAL
        if not prompt:
            return DIVERGENCE_NEUTRAL

        p_words = set(re.findall(r"\w+", prompt.lower()))
        o_words = set(re.findall(r"\w+", out))
        if not p_words or not o_words:
            return DIVERGENCE_NEUTRAL
        similarity = len(p_words & o_words) / len(p_words | o_words)
        return max(0.0, min(1.0, 1.0 - similarity))
