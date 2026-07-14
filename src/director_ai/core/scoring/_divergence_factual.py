# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Factual Divergence (grounding-store/prompt-premise scoring)
"""Factual divergence calculators for the coherence scorer.

:class:`FactualDivergenceMixin` owns the factual-divergence surface of the
:class:`~director_ai.core.scoring.scorer.CoherenceScorer`: scoring a response
against the ground-truth store (with and without evidence capture), the
prompt-as-premise summarisation path, adaptive retrieval routing, calibrated
abstention and RAG claim decomposition. The mixin owns no ``__init__`` — the
composing scorer initialises every attribute declared on the class body, and
the ``TYPE_CHECKING`` stubs document the scorer-provided services the mixin
calls (judge escalation, task detection, heuristic fallback, incident
recording).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..mandatory import mandatory_execution
from ..metrics import metrics
from ..otel import trace_nli_inference, trace_retrieval
from ..types import EvidenceChunk, ScoringEvidence
from ._heuristics import DIVERGENCE_CONTRADICTED, DIVERGENCE_NEUTRAL
from .nli import NLIScorer

__all__ = ["FactualDivergenceMixin"]


class FactualDivergenceMixin:
    """Factual-divergence surface of :class:`CoherenceScorer`.

    Hosts the ground-truth-store and prompt-as-premise factual calculators,
    including the evidence-returning variants the review pipeline consumes.
    All state is initialised by the composing scorer's ``__init__``; the
    annotations below declare that shared contract for static analysis
    without creating attributes.
    """

    # Shared state initialised by the composing scorer.
    strict_mode: bool
    ground_truth_store: Any
    logger: logging.Logger
    _nli: NLIScorer | None
    _rust_scorer: Any
    _fact_inner_agg: str
    _fact_outer_agg: str
    _premise_ratio: float
    _qa_premise_ratio: float
    _fact_retrieval_top_k: int
    _use_prompt_as_premise: bool
    _claim_support_threshold: float
    _claim_coverage_alpha: float
    _rag_claim_decomposition: bool
    _retrieval_abstention_threshold: float
    _chunk_overlap_ratio: float
    _confidence_weighted_agg: bool
    _adaptive_router: Any | None

    if TYPE_CHECKING:
        # Services provided by the composing scorer.

        def _should_escalate(
            self, nli_score: float, task_type: str = "default"
        ) -> bool: ...

        def _llm_judge_check(
            self, prompt: str, response: str, nli_score: float
        ) -> float: ...

        def _record_nli_fallback_incident(self, *, stage: str, reason: str) -> None: ...

        @staticmethod
        def _detect_task_type(prompt: str, response: str = "") -> str: ...

        @staticmethod
        def _heuristic_factual(context: str, text_output: str) -> float: ...

    @staticmethod
    def _has_grounding_query(prompt: str) -> bool:
        """Return whether ``prompt`` can be sent to a grounding store."""
        return bool(prompt.strip())

    def _calculate_prompt_premise_divergence_with_evidence(
        self,
        prompt: str,
        text_output: str,
        tenant_id: str = "",
        *,
        _inner_agg: str | None = None,
        _outer_agg: str | None = None,
    ) -> tuple[float, ScoringEvidence | None]:
        """Score summarisation directly against the source prompt."""
        del tenant_id
        if not (self._nli and self._nli.model_available):
            return DIVERGENCE_NEUTRAL, None

        fact_inner = _inner_agg if _inner_agg is not None else self._fact_inner_agg
        fact_outer = _outer_agg if _outer_agg is not None else self._fact_outer_agg
        task_type = self._detect_task_type(prompt, text_output)

        self._nli.reset_token_counter()
        with metrics.timer("chunked_nli_seconds"):
            if self._confidence_weighted_agg:
                nli_score, chunk_scores = self._nli.score_chunked_confidence_weighted(
                    prompt,
                    text_output,
                    inner_agg=fact_inner,
                    premise_ratio=self._premise_ratio,
                    overlap_ratio=self._chunk_overlap_ratio,
                )
                prem_count = 1
                hyp_count = len(chunk_scores)
            else:
                nli_score, chunk_scores, prem_count, hyp_count = (
                    self._nli._score_chunked_with_counts(
                        prompt,
                        text_output,
                        inner_agg=fact_inner,
                        outer_agg=fact_outer,
                        premise_ratio=self._premise_ratio,
                        overlap_ratio=self._chunk_overlap_ratio,
                    )
                )
        if self._should_escalate(nli_score, task_type=task_type):
            nli_score = self._llm_judge_check(prompt, text_output, nli_score)

        evidence = ScoringEvidence(
            chunks=[EvidenceChunk(text=prompt[:500], distance=0.0, source="prompt")],
            nli_premise=prompt,
            nli_hypothesis=text_output,
            nli_score=nli_score,
            chunk_scores=chunk_scores,
            premise_chunk_count=prem_count,
            hypothesis_chunk_count=hyp_count,
            token_count=self._nli.last_token_count,
            estimated_cost_usd=self._nli.last_estimated_cost,
        )
        return nli_score, evidence

    # ── Factual divergence ────────────────────────────────────────────

    def calculate_factual_divergence(
        self,
        prompt: str,
        text_output: str,
        tenant_id: str = "",
        *,
        _inner_agg: str | None = None,
        _outer_agg: str | None = None,
    ) -> float:
        """Check output against the Ground Truth Store.

        Returns 0.0 (aligned) to 1.0 (hallucinated).
        When strict_mode is True and NLI is unavailable, returns 0.9 (reject).
        """
        if self._rust_scorer is not None:
            _, score_obj = self._rust_scorer.review(prompt, text_output)
            fallback = 1.0 - getattr(score_obj, "score", 0.5)
            return getattr(score_obj, "h_factual", fallback)

        fact_inner = _inner_agg if _inner_agg is not None else self._fact_inner_agg
        fact_outer = _outer_agg if _outer_agg is not None else self._fact_outer_agg

        # Resolve effective premise_ratio: QA tasks get higher ratio
        effective_premise_ratio = self._premise_ratio
        task_type = self._detect_task_type(prompt, text_output)
        if task_type == "qa" and self._qa_premise_ratio > self._premise_ratio:
            effective_premise_ratio = self._qa_premise_ratio

        # Summarization mode: score prompt (source document) directly as premise.
        # Bypasses vector store retrieval which loses context and degrades scores.
        if self._use_prompt_as_premise and self._nli and self._nli.model_available:
            with trace_nli_inference(stage="factual_prompt") as nli_span:
                nli_span.set_attribute("nli.model_available", True)
                nli_span.set_attribute(
                    "nli.confidence_weighted", self._confidence_weighted_agg
                )
                with metrics.timer("chunked_nli_seconds"):
                    if self._confidence_weighted_agg:
                        score, _ = self._nli.score_chunked_confidence_weighted(
                            prompt,
                            text_output,
                            inner_agg=fact_inner,
                            premise_ratio=effective_premise_ratio,
                            overlap_ratio=self._chunk_overlap_ratio,
                        )
                    else:
                        score, _ = self._nli.score_chunked(
                            prompt,
                            text_output,
                            inner_agg=fact_inner,
                            outer_agg=fact_outer,
                            premise_ratio=effective_premise_ratio,
                            overlap_ratio=self._chunk_overlap_ratio,
                        )
                nli_span.set_attribute("nli.score", score)
            if self._should_escalate(score, task_type=task_type):
                score = self._llm_judge_check(prompt, text_output, score)
            return score

        if not self.ground_truth_store:
            return DIVERGENCE_NEUTRAL

        if not self._has_grounding_query(prompt):
            return DIVERGENCE_NEUTRAL

        # Adaptive retrieval routing: skip KB lookup for non-factual queries.
        if self._adaptive_router is not None:
            decision = self._adaptive_router.should_retrieve(prompt, text_output)
            if not decision.retrieve:
                self.logger.debug(
                    "Adaptive router skipped retrieval (type=%s, conf=%.2f)",
                    decision.task_type,
                    decision.confidence,
                )
                return DIVERGENCE_NEUTRAL

        with (
            trace_retrieval(
                top_k=self._fact_retrieval_top_k,
                tenant_scoped=bool(tenant_id),
            ) as retrieval_span,
            metrics.timer("factual_retrieval_seconds"),
        ):
            context = self.ground_truth_store.retrieve_context(
                prompt,
                top_k=self._fact_retrieval_top_k,
                tenant_id=tenant_id,
            )
            retrieval_span.set_attribute("retrieval.has_context", bool(context))
        if not context:
            return DIVERGENCE_NEUTRAL

        # Calibrated abstention: if retrieval returns low-quality results,
        # report insufficient context rather than a misleading score.
        if self._retrieval_abstention_threshold > 0:
            from ..retrieval.vector_store import VectorGroundTruthStore

            if isinstance(self.ground_truth_store, VectorGroundTruthStore):
                chunks = self.ground_truth_store.retrieve_context_with_chunks(
                    prompt,
                    top_k=self._fact_retrieval_top_k,
                    tenant_id=tenant_id,
                )
                if chunks:
                    best_dist = min(c.distance for c in chunks)
                    if best_dist > (1.0 - self._retrieval_abstention_threshold):
                        return DIVERGENCE_NEUTRAL

        if self._nli and self._nli.model_available:
            with trace_nli_inference(stage="factual") as nli_span:
                nli_span.set_attribute("nli.model_available", True)
                nli_span.set_attribute(
                    "nli.confidence_weighted", self._confidence_weighted_agg
                )
                with metrics.timer("chunked_nli_seconds"):
                    if self._confidence_weighted_agg:
                        score, _ = self._nli.score_chunked_confidence_weighted(
                            context,
                            text_output,
                            inner_agg=fact_inner,
                            premise_ratio=effective_premise_ratio,
                            overlap_ratio=self._chunk_overlap_ratio,
                        )
                    else:
                        score, _ = self._nli.score_chunked(
                            context,
                            text_output,
                            inner_agg=fact_inner,
                            outer_agg=fact_outer,
                            premise_ratio=effective_premise_ratio,
                            overlap_ratio=self._chunk_overlap_ratio,
                        )
                nli_span.set_attribute("nli.score", score)

            # RAG claim decomposition: score each response sentence
            # against the context independently, compute coverage.
            if (
                self._rag_claim_decomposition
                and self._nli.model_available
                and len(text_output) > 100
            ):
                claims = self._nli._split_sentences(text_output)
                if len(claims) > 1:
                    supported = 0
                    for claim in claims:
                        claim_div, _ = self._nli.score_chunked(
                            context,
                            claim,
                            inner_agg="min",
                            outer_agg="max",
                            premise_ratio=effective_premise_ratio,
                        )
                        if claim_div < self._claim_support_threshold:
                            supported += 1
                    coverage = supported / len(claims)
                    alpha = self._claim_coverage_alpha
                    score = alpha * (1.0 - coverage) + (1.0 - alpha) * score
        elif self.strict_mode:
            score = DIVERGENCE_CONTRADICTED
        else:
            score = self._heuristic_factual(context, text_output)

        if self._should_escalate(score, task_type=task_type):
            score = self._llm_judge_check(prompt, text_output, score)
        return score

    def calculate_factual_divergence_with_evidence(
        self,
        prompt: str,
        text_output: str,
        tenant_id: str = "",
        *,
        _inner_agg: str | None = None,
        _outer_agg: str | None = None,
    ) -> tuple[float, ScoringEvidence | None]:
        """Like calculate_factual_divergence but also returns evidence."""
        fact_inner = _inner_agg if _inner_agg is not None else self._fact_inner_agg
        fact_outer = _outer_agg if _outer_agg is not None else self._fact_outer_agg

        effective_premise_ratio = self._premise_ratio
        task_type = self._detect_task_type(prompt, text_output)
        if task_type == "qa" and self._qa_premise_ratio > self._premise_ratio:
            effective_premise_ratio = self._qa_premise_ratio

        # Summarization mode: score prompt directly, skip store retrieval.
        if self._use_prompt_as_premise and self._nli and self._nli.model_available:
            self._nli.reset_token_counter()
            with trace_nli_inference(stage="factual_prompt_evidence") as nli_span:
                nli_span.set_attribute("nli.model_available", True)
                nli_span.set_attribute(
                    "nli.confidence_weighted", self._confidence_weighted_agg
                )
                with metrics.timer("chunked_nli_seconds"):
                    if self._confidence_weighted_agg:
                        nli_score, chunk_scores_list = (
                            self._nli.score_chunked_confidence_weighted(
                                prompt,
                                text_output,
                                inner_agg=fact_inner,
                                premise_ratio=effective_premise_ratio,
                                overlap_ratio=self._chunk_overlap_ratio,
                            )
                        )
                        chunk_scores = chunk_scores_list
                        prem_count = 1
                        hyp_count = len(chunk_scores_list)
                    else:
                        nli_score, chunk_scores, prem_count, hyp_count = (
                            self._nli._score_chunked_with_counts(
                                prompt,
                                text_output,
                                inner_agg=fact_inner,
                                outer_agg=fact_outer,
                                premise_ratio=effective_premise_ratio,
                                overlap_ratio=self._chunk_overlap_ratio,
                            )
                        )
                nli_span.set_attribute("nli.score", nli_score)
                nli_span.set_attribute("nli.token_count", self._nli.last_token_count)
            if self._should_escalate(nli_score, task_type=task_type):
                nli_score = self._llm_judge_check(prompt, text_output, nli_score)
            evidence = ScoringEvidence(
                chunks=[
                    EvidenceChunk(text=prompt[:500], distance=0.0, source="prompt"),
                ],
                nli_premise=prompt,
                nli_hypothesis=text_output,
                nli_score=nli_score,
                chunk_scores=chunk_scores,
                premise_chunk_count=prem_count,
                hypothesis_chunk_count=hyp_count,
                token_count=self._nli.last_token_count,
                estimated_cost_usd=self._nli.last_estimated_cost,
            )
            return nli_score, evidence

        if not self.ground_truth_store:
            return DIVERGENCE_NEUTRAL, None

        if not self._has_grounding_query(prompt):
            return DIVERGENCE_NEUTRAL, None

        with (
            trace_retrieval(
                top_k=self._fact_retrieval_top_k,
                tenant_scoped=bool(tenant_id),
            ) as retrieval_span,
            metrics.timer("factual_retrieval_seconds"),
        ):
            chunks: list[EvidenceChunk] = []
            context: str | None = None
            from ..retrieval.vector_store import VectorGroundTruthStore

            if isinstance(self.ground_truth_store, VectorGroundTruthStore):
                chunks = self.ground_truth_store.retrieve_context_with_chunks(
                    prompt,
                    top_k=self._fact_retrieval_top_k,
                    tenant_id=tenant_id,
                )
                if chunks:
                    context = "; ".join(c.text for c in chunks)
            else:
                context = self.ground_truth_store.retrieve_context(
                    prompt,
                    top_k=self._fact_retrieval_top_k,
                    tenant_id=tenant_id,
                )
                if context:
                    chunks = [
                        EvidenceChunk(text=context, distance=0.0, source="keyword"),
                    ]
            retrieval_span.set_attribute("retrieval.has_context", bool(context))
            retrieval_span.set_attribute("retrieval.result_count", len(chunks))

        if not context:
            return DIVERGENCE_NEUTRAL, None

        retrieved_chunk_scores: list[float] | None = None
        prem_count = 1
        hyp_count = 1
        tok_count = 0
        if self._nli and self._nli.model_available:
            self._nli.reset_token_counter()
            with trace_nli_inference(stage="factual_evidence") as nli_span:
                nli_span.set_attribute("nli.model_available", True)
                nli_span.set_attribute(
                    "nli.confidence_weighted", self._confidence_weighted_agg
                )
                with metrics.timer("chunked_nli_seconds"):
                    if self._confidence_weighted_agg:
                        nli_score, chunk_scores_list = (
                            self._nli.score_chunked_confidence_weighted(
                                context,
                                text_output,
                                inner_agg=fact_inner,
                                premise_ratio=effective_premise_ratio,
                                overlap_ratio=self._chunk_overlap_ratio,
                            )
                        )
                        retrieved_chunk_scores = chunk_scores_list
                        hyp_count = len(chunk_scores_list)
                    else:
                        (
                            nli_score,
                            retrieved_chunk_scores,
                            prem_count,
                            hyp_count,
                        ) = self._nli._score_chunked_with_counts(
                            context,
                            text_output,
                            inner_agg=fact_inner,
                            outer_agg=fact_outer,
                            premise_ratio=effective_premise_ratio,
                            overlap_ratio=self._chunk_overlap_ratio,
                        )
            tok_count = self._nli.last_token_count
            nli_span.set_attribute("nli.score", nli_score)
            nli_span.set_attribute("nli.token_count", tok_count)
        elif self.strict_mode:
            nli_score = DIVERGENCE_CONTRADICTED
        else:
            self._record_nli_fallback_incident(
                stage="factual",
                reason="nli_unavailable_using_heuristic",
            )
            nli_score = self._heuristic_factual(context, text_output)

        if self._should_escalate(nli_score, task_type=task_type):
            nli_score = self._llm_judge_check(prompt, text_output, nli_score)

        # Sentence-level attribution: map each response sentence to its
        # best-matching source sentence with divergence score.
        attributions = None
        claim_coverage = None
        per_claim_divs = None
        claims_list = None
        if (
            self._rag_claim_decomposition
            and self._nli
            and self._nli.model_available
            and context
            and len(text_output) > 100
        ):
            with mandatory_execution(
                __name__, component="mandatory claim attribution path"
            ):
                claim_coverage, per_claim_divs, claims_list, attributions = (
                    self._nli.score_claim_coverage_with_attribution(
                        context,
                        text_output,
                        support_threshold=self._claim_support_threshold,
                    )
                )

        evidence = ScoringEvidence(
            chunks=chunks,
            nli_premise=context,
            nli_hypothesis=text_output,
            nli_score=nli_score,
            chunk_scores=retrieved_chunk_scores,
            premise_chunk_count=prem_count,
            hypothesis_chunk_count=hyp_count,
            claim_coverage=claim_coverage,
            per_claim_divergences=per_claim_divs,
            claims=claims_list,
            attributions=attributions,
            token_count=tok_count or None,
            estimated_cost_usd=(
                tok_count * self._nli._cost_per_token
                if tok_count and self._nli
                else None
            ),
        )
        return nli_score, evidence
