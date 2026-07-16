# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Task-Routed Coherence (dialogue/summarisation routing)
"""Task-aware routing of divergence signals for the coherence scorer.

:class:`TaskRoutedCoherenceMixin` owns the task-detection and routing
surface of the :class:`~director_ai.core.scoring.scorer.CoherenceScorer`:
detecting the task type, resolving the aggregation profile, the dialogue
and summarisation task routes, and the composite ``_heuristic_coherence``
entry point that fans factual and logical divergence out in parallel. The
mixin owns no ``__init__`` — the composing scorer initialises every
attribute declared on the class body, and the ``TYPE_CHECKING`` stubs
document the divergence calculators and scorer services the routes call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..types import ScoringEvidence
from ._task_scoring import (
    detect_task_type,
    dialogue_factual_divergence,
    dialogue_raw_support_divergence,
    summarization_factual_divergence,
    weakest_link_claim_divergence,
)
from .heuristic_coherence import (
    HeuristicCoherenceInputs,
    HeuristicCoherenceRoute,
    combine_weighted_coherence,
    select_heuristic_coherence_route,
)
from .nli import NLIScorer

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

__all__ = ["TaskRoutedCoherenceMixin"]


class TaskRoutedCoherenceMixin:
    """Task-routing surface of :class:`CoherenceScorer`.

    Hosts task-type detection, the aggregation-profile resolver, the
    dialogue and summarisation routes and the composite coherence entry
    point. All state is initialised by the composing scorer's
    ``__init__``; the annotations below declare that shared contract for
    static analysis without creating attributes.
    """

    # Shared state initialised by the composing scorer.
    W_LOGIC: float
    W_FACT: float
    logger: logging.Logger
    ground_truth_store: Any
    _nli: NLIScorer | None
    _fact_inner_agg: str
    _fact_outer_agg: str
    _logic_inner_agg: str
    _logic_outer_agg: str
    _premise_ratio: float
    _use_prompt_as_premise: bool
    _auto_dialogue_profile: bool
    _dialogue_nli_baseline: float
    _summarization_nli_baseline: float
    _claim_coverage_enabled: bool
    _claim_support_threshold: float
    _claim_coverage_alpha: float
    _summarization_premise_chars: int
    _dialogue_scoring: str
    _dialogue_support_threshold: float
    _summarization_aggregation: str
    _summarization_support_threshold: float

    if TYPE_CHECKING:
        # Divergence calculators and services provided by the composing scorer.

        def calculate_factual_divergence_with_evidence(
            self,
            prompt: str,
            text_output: str,
            tenant_id: str = "",
            *,
            _inner_agg: str | None = None,
            _outer_agg: str | None = None,
        ) -> tuple[float, ScoringEvidence | None]: ...

        def _calculate_prompt_premise_divergence_with_evidence(
            self,
            prompt: str,
            text_output: str,
            tenant_id: str = "",
            *,
            _inner_agg: str | None = None,
            _outer_agg: str | None = None,
        ) -> tuple[float, ScoringEvidence | None]: ...

        def calculate_logical_divergence(
            self,
            prompt: str,
            text_output: str,
            *,
            _inner_agg: str | None = None,
            _outer_agg: str | None = None,
        ) -> float: ...

        def _get_parallel_pool(self) -> ThreadPoolExecutor: ...

        def _get_minicheck_scorer(self) -> NLIScorer | None: ...

    @staticmethod
    def _detect_task_type(prompt: str, response: str = "") -> str:
        """Detect task type from prompt content and length ratio."""
        return detect_task_type(prompt, response)

    def _resolve_agg_profile(self, prompt: str) -> tuple[str, str, str, str]:
        """Return (fact_inner, fact_outer, logic_inner, logic_outer) agg settings."""
        fi, fo = self._fact_inner_agg, self._fact_outer_agg
        li, lo = self._logic_inner_agg, self._logic_outer_agg

        if (
            self._auto_dialogue_profile
            and not self._use_prompt_as_premise
            and fi == "max"
            and fo == "max"
            and li == "max"
            and lo == "max"
            and detect_task_type(prompt) == "dialogue"
        ):
            return "min", "mean", "min", "mean"

        return fi, fo, li, lo

    # -- Dialogue-specific scoring -----------------------------------------

    def _dialogue_factual_divergence(
        self,
        prompt: str,
        response: str,
        tenant_id: str = "",
    ) -> tuple[float, ScoringEvidence | None]:
        """Dialogue factual divergence in the configured scoring mode.

        ``raw_support`` (default, WCS-2a) returns the raw weakest-link
        claim divergence — the review gate compares the resulting
        support against the matched-FPR dialogue operating point.
        ``baseline_squeeze`` restores the pre-WCS-2a bidirectional NLI
        with 0.80-baseline calibration.
        """
        if self._nli is None or not self._nli.model_available:
            raise RuntimeError("NLI model required for dialogue factual divergence")
        if self._dialogue_scoring == "raw_support":
            return dialogue_raw_support_divergence(self._nli, prompt, response)
        return dialogue_factual_divergence(
            self._nli,
            prompt,
            response,
            tenant_id,
            calculate_factual_with_evidence=self.calculate_factual_divergence_with_evidence,
            baseline=self._dialogue_nli_baseline,
        )

    # -- Summarization-specific scoring ------------------------------------

    def _summarization_factual_divergence(
        self,
        prompt: str,
        response: str,
        tenant_id: str = "",
    ) -> tuple[float, ScoringEvidence | None]:
        """Bidirectional NLI + claim coverage for summarisation."""
        if self._nli is None or not self._nli.model_available:
            raise RuntimeError(
                "NLI model required for summarization factual divergence"
            )
        return summarization_factual_divergence(
            self._nli,
            prompt,
            response,
            tenant_id,
            calculate_factual_with_evidence=(
                self._calculate_prompt_premise_divergence_with_evidence
            ),
            fact_inner_agg=self._fact_inner_agg,
            fact_outer_agg=self._fact_outer_agg,
            premise_ratio=self._premise_ratio,
            claim_coverage_enabled=self._claim_coverage_enabled,
            claim_support_threshold=self._claim_support_threshold,
            claim_coverage_alpha=self._claim_coverage_alpha,
            baseline=self._summarization_nli_baseline,
            get_minicheck_scorer=self._get_minicheck_scorer,
            premise_chars=self._summarization_premise_chars,
            aggregation=self._summarization_aggregation,
        )

    # -- Raw-support operating points (WCS-2a) ------------------------------

    def _raw_support_operating_point(self, prompt: str, action: str) -> float | None:
        """Support threshold when a raw-support route will score this input.

        Mirrors the :meth:`_heuristic_coherence` route selection exactly
        so the review gate and the scoring route stay consistent; returns
        ``None`` when the input takes a squeeze/blend route, where
        composite-coherence thresholds apply instead.
        """
        nli_ok = self._nli is not None and self._nli.model_available
        task = self._detect_task_type(prompt, action) if nli_ok else "default"
        route = select_heuristic_coherence_route(
            HeuristicCoherenceInputs(
                auto_dialogue_profile=self._auto_dialogue_profile,
                use_prompt_as_premise=self._use_prompt_as_premise,
                nli_available=nli_ok,
                task_type=task,
                w_logic=self.W_LOGIC,
            )
        )
        if (
            route is HeuristicCoherenceRoute.DIALOGUE
            and self._dialogue_scoring == "raw_support"
        ):
            return self._dialogue_support_threshold
        if (
            route is HeuristicCoherenceRoute.SUMMARISATION
            and self._summarization_aggregation == "weakest_link"
        ):
            return self._summarization_support_threshold
        return None

    def raw_task_support(self, prompt: str, response: str) -> tuple[str, float]:
        """Return the detected task type and raw weakest-link support.

        Calibration surface for
        :mod:`director_ai.core.calibration.operating_points`: scores
        through the same premise the raw-support routes use (dialogue —
        the whole conversation context; summarisation — the configured
        premise-chars budget of the source), regardless of the currently
        configured scoring mode, so a deployment can collect supports and
        pick its operating point BEFORE switching modes.
        """
        if self._nli is None or not self._nli.model_available:
            raise RuntimeError("NLI model required for raw support calibration")
        task = self._detect_task_type(prompt, response)
        premise = prompt
        if task == "summarization" and self._summarization_premise_chars > 0:
            premise = prompt[: self._summarization_premise_chars]
        divergence, _divs, _claims = weakest_link_claim_divergence(
            self._nli, premise, response
        )
        return task, 1.0 - divergence

    # ── Composite heuristic coherence ─────────────────────────────────

    def _heuristic_coherence(
        self,
        prompt: str,
        action: str,
        tenant_id: str = "",
    ) -> tuple[float, float, float, ScoringEvidence | None]:
        """Compute coherence components.

        Returns (h_logical, h_factual, coherence, evidence).
        H_logical and H_factual run in parallel – vector retrieval overlaps
        with the logical NLI forward pass.

        For dialogue prompts (auto-detected), uses bidirectional NLI with
        baseline calibration instead of standard forward-only scoring.
        Logical divergence is skipped for dialogue (entailment is meaningless).
        """
        # Eager-load NLI in the main thread to avoid PyTorch 2.6 dispatch
        # corruption when from_pretrained runs inside a ThreadPoolExecutor
        # worker after a CUDA model was already loaded.
        if self._nli is not None and hasattr(self._nli, "_ensure_model"):
            self._nli._ensure_model()

        # Task-aware aggregation profile
        fact_ia, fact_oa, logic_ia, logic_oa = self._resolve_agg_profile(prompt)

        _nli_available = self._nli is not None and self._nli.model_available
        _task_type = (
            self._detect_task_type(prompt, action) if _nli_available else "default"
        )

        route = select_heuristic_coherence_route(
            HeuristicCoherenceInputs(
                auto_dialogue_profile=self._auto_dialogue_profile,
                use_prompt_as_premise=self._use_prompt_as_premise,
                nli_available=_nli_available,
                task_type=_task_type,
                w_logic=self.W_LOGIC,
            )
        )

        if route is HeuristicCoherenceRoute.DIALOGUE:
            h_logic = 0.0
            h_fact, evidence = self._dialogue_factual_divergence(
                prompt,
                action,
                tenant_id,
            )
        elif route is HeuristicCoherenceRoute.SUMMARISATION:
            h_logic = 0.0
            h_fact, evidence = self._summarization_factual_divergence(
                prompt,
                action,
                tenant_id,
            )
        elif route is HeuristicCoherenceRoute.FACTUAL_ONLY:
            h_logic = 0.0
            h_fact, evidence = self.calculate_factual_divergence_with_evidence(
                prompt,
                action,
                tenant_id,
                _inner_agg=fact_ia,
                _outer_agg=fact_oa,
            )
        elif self.ground_truth_store is not None and not self._use_prompt_as_premise:
            # Grounded path: the factual retrieval yields the context that is
            # ALSO the correct premise for the logical NLI. A bare interrogative
            # prompt is a degenerate NLI premise — a true declarative answer does
            # not entail the question that prompted it, so premise=prompt inflates
            # h_logical for EVERY true answer and false-halts true inputs
            # (KIMI2-K, GPU-reproduced 2026-07-16). Scoring the logical signal
            # against the retrieved context keeps false claims caught (the context
            # contradicts them) while letting true claims through (the context
            # entails them). Serialised because the premise is only known after
            # retrieval; the ungrounded branch below keeps the parallel overlap.
            h_fact, evidence = self.calculate_factual_divergence_with_evidence(
                prompt,
                action,
                tenant_id,
                _inner_agg=fact_ia,
                _outer_agg=fact_oa,
            )
            logical_premise = (
                evidence.nli_premise
                if evidence is not None and evidence.nli_premise
                else prompt
            )
            h_logic = self.calculate_logical_divergence(
                logical_premise,
                action,
                _inner_agg=logic_ia,
                _outer_agg=logic_oa,
            )
        else:
            pool = self._get_parallel_pool()
            future_logic = pool.submit(
                self.calculate_logical_divergence,
                prompt,
                action,
                _inner_agg=logic_ia,
                _outer_agg=logic_oa,
            )
            future_fact = pool.submit(
                self.calculate_factual_divergence_with_evidence,
                prompt,
                action,
                tenant_id,
                _inner_agg=fact_ia,
                _outer_agg=fact_oa,
            )
            try:
                h_logic = future_logic.result()
            except Exception:
                if not future_fact.cancel():
                    try:
                        future_fact.result()
                    except Exception:
                        self.logger.debug(
                            "Suppressed factual-divergence future exception "
                            "after logical-divergence failure.",
                            exc_info=True,
                        )
                raise
            h_fact, evidence = future_fact.result()
        raw_support_route = (
            route is HeuristicCoherenceRoute.DIALOGUE
            and self._dialogue_scoring == "raw_support"
        ) or (
            route is HeuristicCoherenceRoute.SUMMARISATION
            and self._summarization_aggregation == "weakest_link"
        )
        coherence = combine_weighted_coherence(
            h_logic=h_logic,
            h_factual=h_fact,
            w_logic=self.W_LOGIC,
            w_fact=self.W_FACT,
            nli_available=_nli_available,
            evidence_present=evidence is not None,
            dialogue_route=route is HeuristicCoherenceRoute.DIALOGUE,
            raw_support_route=raw_support_route,
        )

        return h_logic, h_fact, coherence, evidence
