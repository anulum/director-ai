# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Coherence Scorer (Weighted NLI Divergence)

from __future__ import annotations

import asyncio
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ...enterprise.redactor import PIIRedactor
from ..cache import ScoreCache
from ..metrics import metrics
from ..otel import trace_review
from ..types import CoherenceScore, EvidenceChunk, ScoringEvidence
from ._llm_judge import LLMJudge
from ._task_scoring import (
    _DIALOGUE_TURN_RE,  # noqa: F401 — re-export for backward compat
    detect_task_type,
    dialogue_factual_divergence,
    summarization_factual_divergence,
)
from .nli import NLIScorer, nli_available

__all__ = ["CoherenceScorer"]

# Heuristic divergence defaults (used when NLI model unavailable)
DIVERGENCE_NEUTRAL = 0.5  # no signal → agnostic
DIVERGENCE_ALIGNED = 0.1  # keyword heuristic: "consistent with reality"
DIVERGENCE_CONTRADICTED = 0.9  # keyword heuristic: "opposite is true"


class CoherenceScorer:
    """Weighted NLI divergence scorer for AI output verification.

    Computes a composite coherence score from two NLI-based signals:
    - **Logical divergence** (H_logical): NLI contradiction probability
      between prompt and response.
    - **Factual divergence** (H_factual): NLI contradiction probability
      between retrieved context and response.

    Final score: ``coherence = 1 - (0.6 * H_logical + 0.4 * H_factual)``.
    When coherence falls below ``threshold``, the output is rejected.

    Parameters
    ----------
    threshold : float – minimum coherence to approve (default 0.5).
    soft_limit : float | None – scores between threshold and soft_limit
        trigger a warning. Default: threshold + 0.1.
    w_logic : float – weight for logical divergence (default 0.6).
    w_fact : float – weight for factual divergence (default 0.4).
        Must satisfy w_logic + w_fact = 1.0.
    strict_mode : bool – when True, disables heuristic fallbacks entirely.
        If NLI model is unavailable and strict_mode is True, divergence
        returns 0.9 (reject) and sets ``strict_mode_rejected=True``.
    history_window : int – rolling history size.
    use_nli : bool | None – True forces NLI, False disables it,
        None (default) auto-detects based on installed packages.
    ground_truth_store : GroundTruthStore | None – fact store for RAG.
    nli_model : str | None – HuggingFace model ID or local path for NLI.
    cache_size : int – LRU score cache max entries (0 to disable).
    cache_ttl : float – cache entry TTL in seconds.
    nli_quantize_8bit : bool – load NLI model with 8-bit quantization.
    nli_device : str | None – torch device for NLI model.
    nli_torch_dtype : str | None – torch dtype ("float16", "bfloat16").
    llm_judge_enabled : bool – escalate to LLM when NLI margin is low.
    llm_judge_confidence_threshold : float – softmax margin below which
        to escalate (default 0.3).
    llm_judge_provider : str – "openai" or "anthropic".
    privacy_mode : bool – redact PII (emails, phones, SSN-like patterns)
        before sending text to external LLM judge.

    """

    W_LOGIC = 0.6
    W_FACT = 0.4
    _minicheck_nli: NLIScorer | None

    def __init__(
        self,
        threshold=0.5,
        history_window=5,
        use_nli=None,
        ground_truth_store=None,
        nli_model=None,
        soft_limit=None,
        w_logic=None,
        w_fact=None,
        strict_mode=False,
        cache_size=0,
        cache_ttl=300.0,
        nli_quantize_8bit=False,
        nli_device=None,
        nli_torch_dtype=None,
        llm_judge_enabled=False,
        llm_judge_confidence_threshold=0.3,
        llm_judge_provider="",
        llm_judge_model="",
        scorer_backend="deberta",
        onnx_path=None,
        nli_devices=None,
        onnx_batch_size=16,
        onnx_flush_timeout_ms=10.0,
        privacy_mode=False,
        cache=None,
        nli_max_length=512,
        nli_revision=None,
    ):
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        self.threshold = threshold
        self.soft_limit = (
            soft_limit if soft_limit is not None else min(threshold + 0.1, 1.0)
        )
        if not (0.0 <= self.soft_limit <= 1.0):
            raise ValueError(f"soft_limit must be in [0, 1], got {self.soft_limit}")
        if self.soft_limit < threshold:
            raise ValueError(
                f"soft_limit ({self.soft_limit}) must be >= threshold ({threshold})",
            )

        self.strict_mode = strict_mode
        self.scorer_backend = scorer_backend
        self.onnx_path = onnx_path
        self._onnx_batch_size = onnx_batch_size
        self._onnx_flush_timeout_ms = onnx_flush_timeout_ms

        if scorer_backend == "hybrid" and not llm_judge_provider:
            raise ValueError("hybrid backend requires llm_judge_provider")

        if w_logic is not None or w_fact is not None:
            self.W_LOGIC = w_logic if w_logic is not None else 0.6
            self.W_FACT = w_fact if w_fact is not None else 0.4
            if not (0.0 <= self.W_LOGIC <= 1.0):
                raise ValueError(f"w_logic must be in [0, 1], got {self.W_LOGIC}")
            if not (0.0 <= self.W_FACT <= 1.0):
                raise ValueError(f"w_fact must be in [0, 1], got {self.W_FACT}")
            if abs(self.W_LOGIC + self.W_FACT - 1.0) > 1e-9:
                raise ValueError(
                    f"w_logic + w_fact must equal 1.0, got {self.W_LOGIC + self.W_FACT}",
                )
        self.history = []
        self.window = history_window
        self.ground_truth_store = ground_truth_store
        self.logger = logging.getLogger("DirectorAI")
        self._history_lock = threading.Lock()

        if cache is not None:
            self.cache = cache
        elif cache_size > 0:
            self.cache = ScoreCache(max_size=cache_size, ttl_seconds=cache_ttl)
        else:
            self.cache = None

        if use_nli is None:
            self.use_nli = nli_available()
        else:
            self.use_nli = use_nli

        # Rust/backfire backend: delegate to backfire_kernel FFI
        if scorer_backend in ("rust", "backfire"):
            try:
                from backfire_kernel import BackfireConfig, RustCoherenceScorer

                self._rust_scorer = RustCoherenceScorer(
                    config=BackfireConfig(coherence_threshold=threshold),
                    knowledge_callback=(
                        ground_truth_store.retrieve_context
                        if ground_truth_store
                        else None
                    ),
                )
                self._nli = None  # type: ignore[assignment]
                self.use_nli = False
            except (ImportError, AttributeError, OSError):
                self._rust_scorer = None
                if strict_mode:
                    raise
        else:
            self._rust_scorer = None

        nli_backend = "deberta" if scorer_backend == "hybrid" else scorer_backend
        if nli_backend == "lite":
            self._nli = NLIScorer(use_model=False, backend="lite")
        elif self.use_nli and nli_devices and len(nli_devices) > 1:
            from .sharded_nli import ShardedNLIScorer

            self._nli = ShardedNLIScorer(  # type: ignore[assignment]
                devices=nli_devices,
                use_model=True,
                model_name=nli_model,
                backend=nli_backend,
                quantize_8bit=nli_quantize_8bit,
                torch_dtype=nli_torch_dtype,
                onnx_path=onnx_path,
                onnx_batch_size=onnx_batch_size,
                onnx_flush_timeout_ms=onnx_flush_timeout_ms,
            )
        elif self.use_nli:
            self._nli = NLIScorer(
                use_model=self.use_nli,
                model_name=nli_model,
                backend=nli_backend,
                quantize_8bit=nli_quantize_8bit,
                device=nli_device,
                torch_dtype=nli_torch_dtype,
                onnx_path=onnx_path,
                onnx_batch_size=onnx_batch_size,
                onnx_flush_timeout_ms=onnx_flush_timeout_ms,
                max_length=nli_max_length,
                revision=nli_revision,
            )
        else:
            self._nli = None  # type: ignore[assignment]
        self._privacy_mode = privacy_mode
        self._redactor = PIIRedactor(enabled=privacy_mode)
        self._parallel_pool = ThreadPoolExecutor(max_workers=2)
        self._fact_inner_agg = "max"
        self._fact_outer_agg = "max"
        self._logic_inner_agg = "max"
        self._logic_outer_agg = "max"
        self._premise_ratio = 0.4
        self._fact_retrieval_top_k = 3
        self._use_prompt_as_premise = False
        self._auto_dialogue_profile = True  # auto-detect dialogue, apply bidir NLI
        self._dialogue_nli_baseline = 0.80
        self._summarization_nli_baseline = 0.20  # HaluEval 200: 25.5%→10.5% FPR
        self._claim_coverage_enabled = True
        self._claim_support_threshold = 0.6  # HaluEval 200: 10.5%→2.0% FPR
        self._rag_claim_decomposition = True  # per-sentence scoring for RAG path
        self._retrieval_abstention_threshold = 0.0
        self._claim_coverage_alpha = 0.4
        self._adaptive_threshold_enabled = False
        self._task_type_thresholds: dict[str, float] = {}
        self._chunk_overlap_ratio = 0.5
        self._qa_premise_ratio = 0.7
        self._confidence_weighted_agg = False
        self._meta_classifier_path = ""
        self._meta_classifier = None
        self._adaptive_router = None  # set via enable_adaptive_retrieval()

        # LLM-as-judge subsystem (composed — see _llm_judge.py)
        self._judge = LLMJudge(
            provider=llm_judge_provider
            if (llm_judge_enabled or scorer_backend == "hybrid")
            else "",
            model=llm_judge_model,
            confidence_threshold=llm_judge_confidence_threshold,
            device=nli_device,
            privacy_mode=privacy_mode,
        )
        # Backward-compat aliases used by tests
        self._llm_judge_enabled = self._judge.enabled
        self._llm_judge_provider = llm_judge_provider
        self._llm_judge_threshold = llm_judge_confidence_threshold

        # Injection detection: set via enable_injection_detection()
        self._injection_detector = None

    # -- Backward-compat proxies for judge internals (used by tests) ----

    @property
    def _local_judge_model(self):
        return self._judge._local_judge_model

    @_local_judge_model.setter
    def _local_judge_model(self, value):
        self._judge._local_judge_model = value

    @property
    def _local_judge_tokenizer(self):
        return self._judge._local_judge_tokenizer

    @_local_judge_tokenizer.setter
    def _local_judge_tokenizer(self, value):
        self._judge._local_judge_tokenizer = value

    @property
    def _local_judge_device(self):
        return self._judge._local_judge_device

    @_local_judge_device.setter
    def _local_judge_device(self, value):
        self._judge._local_judge_device = value

    @property
    def _judge_cache(self):
        return self._judge._judge_cache

    @property
    def _JUDGE_CACHE_MAX(self):  # noqa: N802
        return self._judge._JUDGE_CACHE_MAX

    @property
    def _JUDGE_RETRY_MAX(self):  # noqa: N802
        return self._judge._JUDGE_RETRY_MAX

    @property
    def _llm_judge_model(self):
        return self._judge.model

    @property
    def _task_judge_thresholds(self):
        return self._judge.task_judge_thresholds

    def _local_judge_check(self, prompt: str, response: str, nli_score: float) -> float:
        """Backward-compat proxy for LLMJudge._local_judge_check."""
        return self._judge._local_judge_check(prompt, response, nli_score)

    @staticmethod
    def _parse_judge_reply(reply: str) -> tuple[bool, float]:
        """Backward-compat proxy for LLMJudge._parse_judge_reply."""
        return LLMJudge._parse_judge_reply(reply)

    @staticmethod
    def _minicheck_claim_coverage(
        mc_scorer,
        source: str,
        summary: str,
    ) -> tuple[float, list[float], list[str]]:
        """Backward-compat proxy for minicheck_claim_coverage."""
        from ._task_scoring import minicheck_claim_coverage

        return minicheck_claim_coverage(mc_scorer, source, summary)

    def close(self) -> None:
        """Shut down internal thread pool."""
        self._parallel_pool.shutdown(wait=False)

    def __del__(self) -> None:
        pool = getattr(self, "_parallel_pool", None)
        if pool is not None:
            pool.shutdown(wait=False)

    _BUNDLED_CLASSIFIER = "models/dataset_type_classifier.pkl"

    def _get_meta_classifier(self):
        """Lazy-load trained meta-classifier from pickle."""
        if self._meta_classifier is not None:
            return self._meta_classifier

        path = self._meta_classifier_path
        if not path and self._adaptive_threshold_enabled:
            bundled = Path(__file__).parent.parent / self._BUNDLED_CLASSIFIER
            if bundled.exists():
                path = str(bundled)

        if not path:
            return None
        try:
            from .meta_classifier import DatasetTypeClassifier

            self._meta_classifier = DatasetTypeClassifier(path)
            return self._meta_classifier
        except (ImportError, FileNotFoundError, Exception):
            self.logger.debug("Meta-classifier unavailable at %s", path)
            self._meta_classifier_path = ""
            return None

    def _should_escalate(self, nli_score: float, task_type: str = "default") -> bool:
        """Delegate to LLMJudge.should_escalate()."""
        return self._judge.should_escalate(nli_score, task_type)

    def _llm_judge_check(self, prompt: str, response: str, nli_score: float) -> float:
        """Delegate to LLMJudge.check()."""
        return self._judge.check(
            prompt,
            response,
            nli_score,
            redactor=self._redactor,
        )

    # -- Task-aware scoring (delegated to _task_scoring.py) -----------

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
        """Bidirectional NLI scoring with baseline calibration for dialogue."""
        if self._nli is None or not self._nli.model_available:
            raise RuntimeError("NLI model required for dialogue factual divergence")
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
            calculate_factual_with_evidence=self.calculate_factual_divergence_with_evidence,
            fact_inner_agg=self._fact_inner_agg,
            fact_outer_agg=self._fact_outer_agg,
            premise_ratio=self._premise_ratio,
            claim_coverage_enabled=self._claim_coverage_enabled,
            claim_support_threshold=self._claim_support_threshold,
            claim_coverage_alpha=self._claim_coverage_alpha,
            baseline=self._summarization_nli_baseline,
            get_minicheck_scorer=self._get_minicheck_scorer,
        )

    def _get_minicheck_scorer(self) -> NLIScorer | None:
        """Lazily create a MiniCheck NLI scorer for summarisation routing."""
        if hasattr(self, "_minicheck_nli"):
            return self._minicheck_nli

        try:
            mc = NLIScorer(use_model=True, backend="minicheck")
            if mc._ensure_minicheck():
                self._minicheck_nli = mc
                self.logger.info("MiniCheck auto-routing enabled for summarisation")
                return mc
        except Exception as exc:
            self.logger.debug("MiniCheck auto-routing unavailable: %s", exc)

        self._minicheck_nli = None
        return None

    # ── Injection detection ──────────────────────────────────────────

    def enable_injection_detection(
        self,
        injection_threshold: float = 0.7,
        drift_threshold: float = 0.6,
        injection_claim_threshold: float = 0.75,
        baseline_divergence: float = 0.4,
        stage1_weight: float = 0.3,
    ) -> None:
        """Enable output-side injection detection on every review() call."""
        from ..safety.injection import InjectionDetector

        sanitizer = None
        try:
            from ..safety.sanitizer import InputSanitizer

            sanitizer = InputSanitizer()
        except Exception:
            self.logger.debug("InputSanitizer unavailable for injection detection")

        self._injection_detector = InjectionDetector(
            nli_scorer=self._nli,
            sanitizer=sanitizer,
            injection_threshold=injection_threshold,
            drift_threshold=drift_threshold,
            injection_claim_threshold=injection_claim_threshold,
            baseline_divergence=baseline_divergence,
            stage1_weight=stage1_weight,
        )
        self.logger.info(
            "Injection detection enabled (threshold=%.2f)", injection_threshold
        )

    def _get_injection_detector(self):
        """Return the InjectionDetector if enabled, else None."""
        return self._injection_detector

    def enable_adaptive_retrieval(
        self,
        threshold: float = 0.5,
        default_retrieve: bool = True,
    ) -> None:
        """Enable adaptive retrieval routing.

        When enabled, non-factual queries (creative, conversational)
        skip KB retrieval entirely, saving latency and avoiding false
        KB matches on queries that do not need grounding.
        """
        from ..retrieval.adaptive_router import AdaptiveRouter

        self._adaptive_router = AdaptiveRouter(
            factual_threshold=threshold,
            default_retrieve=default_retrieve,
        )
        self.logger.info("Adaptive retrieval enabled (threshold=%.2f)", threshold)

    # ── Factual divergence ────────────────────────────────────────────

    def calculate_factual_divergence(
        self,
        prompt,
        text_output,
        tenant_id: str = "",
        *,
        _inner_agg=None,
        _outer_agg=None,
    ):
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
            if self._should_escalate(score, task_type=task_type):
                score = self._llm_judge_check(prompt, text_output, score)
            return score

        if not self.ground_truth_store:
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

        with metrics.timer("factual_retrieval_seconds"):
            context = self.ground_truth_store.retrieve_context(
                prompt,
                top_k=self._fact_retrieval_top_k,
                tenant_id=tenant_id,
            )
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
        prompt,
        text_output,
        tenant_id: str = "",
        *,
        _inner_agg=None,
        _outer_agg=None,
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

        with metrics.timer("factual_retrieval_seconds"):
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

        if not context:
            return DIVERGENCE_NEUTRAL, None

        chunk_scores = None  # type: ignore[assignment]
        prem_count = 1
        hyp_count = 1
        tok_count = 0
        if self._nli and self._nli.model_available:
            self._nli.reset_token_counter()
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
                    chunk_scores = chunk_scores_list
                    hyp_count = len(chunk_scores_list)
                else:
                    nli_score, chunk_scores, prem_count, hyp_count = (
                        self._nli._score_chunked_with_counts(
                            context,
                            text_output,
                            inner_agg=fact_inner,
                            outer_agg=fact_outer,
                            premise_ratio=effective_premise_ratio,
                            overlap_ratio=self._chunk_overlap_ratio,
                        )
                    )
            tok_count = self._nli.last_token_count
        elif self.strict_mode:
            nli_score = DIVERGENCE_CONTRADICTED
        else:
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
            import contextlib

            with contextlib.suppress(ValueError, RuntimeError):
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
            chunk_scores=chunk_scores,
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

    @staticmethod
    def _heuristic_factual(context, text_output):
        """Word-overlap factual divergence with negation and entity checks.

        Install [nli] for production scoring.
        """
        from ._heuristics import ENTITY_RE, NEGATION_WORDS, STOP_WORDS

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

        # Novel entities in output not grounded in context â†’ +0.15
        ctx_ents = set(ENTITY_RE.findall(context))
        out_ents = set(ENTITY_RE.findall(text_output))
        novel_ents = out_ents - ctx_ents
        if novel_ents:
            divergence += 0.15

        return max(0.0, min(1.0, divergence))

    # â"€â"€ Logical divergence â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    def calculate_logical_divergence(
        self,
        prompt,
        text_output,
        *,
        _inner_agg=None,
        _outer_agg=None,
    ):
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
            with metrics.timer("chunked_nli_seconds"):
                score, _ = self._nli.score_chunked(
                    prompt,
                    text_output,
                    inner_agg=logic_inner,
                    outer_agg=logic_outer,
                    premise_ratio=self._premise_ratio,
                )
            return score

        if self.strict_mode:
            return DIVERGENCE_CONTRADICTED

        return self._heuristic_logical(text_output, prompt)

    @staticmethod
    def _heuristic_logical(text_output, prompt=""):
        """Keyword + word-overlap logical divergence (no-NLI fallback).

        Install [nli] for production-grade scoring.
        """
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

    # â"€â"€ Shared helpers â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    def _heuristic_coherence(self, prompt, action, tenant_id: str = ""):
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

        # -- Dialogue path: bidirectional NLI + baseline calibration ----
        # Logical entailment is meaningless for dialogue (a question
        # doesn’t entail its answer).  Standard NLI gives ~0.92 divergence
        # for correct responses.  The dialogue path uses min(fwd, rev) with
        # baseline calibration to bring FPR from 97.5% -> 4.5% at t=0.50.
        _is_dialogue = (
            self._auto_dialogue_profile
            and not self._use_prompt_as_premise
            and _nli_available
            and _task_type == "dialogue"
        )

        # -- Summarization path: bidirectional NLI + claim coverage -----
        # Abstractive rephrasing causes forward NLI to over-reject.  The
        # reverse direction (summary->document) catches paraphrases.
        # Auto-routes when task is detected as summarization, OR when
        # explicitly configured via _use_prompt_as_premise + W_LOGIC=0.
        _is_summarization = _nli_available and (
            (self._use_prompt_as_premise and self.W_LOGIC < 1e-9)
            or (_task_type == "summarization" and self._auto_dialogue_profile)
        )

        if _is_dialogue:
            h_logic = 0.0
            h_fact, evidence = self._dialogue_factual_divergence(
                prompt,
                action,
                tenant_id,
            )
        elif _is_summarization:
            h_logic = 0.0
            h_fact, evidence = self._summarization_factual_divergence(
                prompt,
                action,
                tenant_id,
            )
        # Short-circuit: skip logical divergence when W_LOGIC is zero.
        elif self.W_LOGIC < 1e-9:
            h_logic = 0.0
            h_fact, evidence = self.calculate_factual_divergence_with_evidence(
                prompt,
                action,
                tenant_id,
                _inner_agg=fact_ia,
                _outer_agg=fact_oa,
            )
        else:
            future_logic = self._parallel_pool.submit(
                self.calculate_logical_divergence,
                prompt,
                action,
                _inner_agg=logic_ia,
                _outer_agg=logic_oa,
            )
            future_fact = self._parallel_pool.submit(
                self.calculate_factual_divergence_with_evidence,
                prompt,
                action,
                tenant_id,
                _inner_agg=fact_ia,
                _outer_agg=fact_oa,
            )
            h_logic = future_logic.result()
            h_fact, evidence = future_fact.result()
        total_divergence = self.W_LOGIC * h_logic + self.W_FACT * h_fact
        coherence = 1.0 - total_divergence

        # Without KB context, h_fact is DIVERGENCE_NEUTRAL (0.5) and scores
        # compress to [lo, hi] (e.g. [0.2, 0.8] for default weights).
        # Rescale to [0, 1] so thresholds are meaningful.  With KB context
        # the factual component carries real signal and no rescaling is needed.
        nli_available = self._nli is not None and self._nli.model_available
        fact_is_neutral = abs(h_fact - DIVERGENCE_NEUTRAL) < 1e-9
        if nli_available and fact_is_neutral and evidence is None and not _is_dialogue:
            # Theoretical range without KB: score ∈ [1-W_L-W_F*0.5, 1-W_F*0.5]
            # Default W_L=0.6, W_F=0.4 → [0.2, 0.8].  Map to [0, 1].
            lo = 1.0 - self.W_LOGIC - self.W_FACT * DIVERGENCE_NEUTRAL
            hi = 1.0 - self.W_FACT * DIVERGENCE_NEUTRAL
            span = hi - lo
            if span > 1e-9:
                coherence = max(0.0, min(1.0, (coherence - lo) / span))

        return h_logic, h_fact, coherence, evidence

    def _finalise_review(
        self,
        coherence,
        h_logic,
        h_fact,
        action,
        evidence=None,
        threshold_override=None,
        detected_task_type=None,
        escalated_to_judge=None,
    ) -> tuple[bool, CoherenceScore]:
        """Build CoherenceScore, gate on threshold, update history.

        Returns (approved, CoherenceScore).
        """
        t = threshold_override if threshold_override is not None else self.threshold
        approved = coherence >= t
        warning = False

        if not approved:
            self.logger.critical(
                "COHERENCE FAILURE. Score: %.4f < Threshold: %s",
                coherence,
                t,
            )
        else:
            if coherence < self.soft_limit:
                warning = True
            with self._history_lock:
                self.history.append(action)
                if len(self.history) > self.window:
                    self.history.pop(0)

        strict_rejected = self.strict_mode and not (
            self._nli and self._nli.model_available
        )

        from .meta_confidence import compute_meta_confidence

        vc, _mc, sa = compute_meta_confidence(
            score=coherence,
            threshold=t,
            h_logical=h_logic,
            h_factual=h_fact,
        )

        # Retrieval confidence from evidence chunks
        retrieval_conf = None
        if evidence is not None and evidence.chunks:
            best = min((c.distance for c in evidence.chunks), default=1.0)
            retrieval_conf = max(0.0, 1.0 - best)

        score = CoherenceScore(
            score=coherence,
            approved=approved,
            h_logical=h_logic,
            h_factual=h_fact,
            evidence=evidence,
            warning=warning,
            strict_mode_rejected=strict_rejected,
            verdict_confidence=vc,
            signal_agreement=sa,
            detected_task_type=detected_task_type,
            escalated_to_judge=escalated_to_judge,
            retrieval_confidence=retrieval_conf,
        )
        return approved, score

    # â"€â"€ Composite scoring â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    def compute_divergence(self, prompt, action):
        """Compute composite divergence (lower is better).

        Weighted sum: ``W_LOGIC * H_logical + W_FACT * H_factual``.
        """
        h_logic = self.calculate_logical_divergence(prompt, action)
        h_fact = self.calculate_factual_divergence(prompt, action)
        total = (self.W_LOGIC * h_logic) + (self.W_FACT * h_fact)
        self.logger.debug(
            "Divergence: Logic=%.2f, Fact=%.2f -> Total=%.2f",
            h_logic,
            h_fact,
            total,
        )
        return total

    def review(
        self,
        prompt: str,
        action: str,
        session=None,
        tenant_id: str = "",
    ) -> tuple[bool, CoherenceScore]:
        """Score an action and decide whether to approve it.

        Parameters
        ----------
        session : ConversationSession | None – when provided, cross-turn
            divergence is blended into the logical score and the turn is
            recorded after scoring.

        """
        with trace_review() as span:
            # Rust fast-path: delegate full review to backfire_kernel
            if self._rust_scorer is not None:
                approved_r, score_obj = self._rust_scorer.review(prompt, action)
                h_l = getattr(score_obj, "h_logical", 0.0)
                h_f = getattr(score_obj, "h_factual", 0.0)
                fallback = 1.0 - (self.W_LOGIC * h_l + self.W_FACT * h_f)
                coh = getattr(score_obj, "score", fallback)
                result = self._finalise_review(coh, h_l, h_f, action)
                span.set_attribute("coherence.score", result[1].score)
                span.set_attribute("coherence.approved", result[0])
                span.set_attribute("coherence.backend", "rust")
                return result

            cache_scope = ""
            if session is not None and len(session) > 0:
                cache_scope = session.context_text

            if self.cache:
                cached = self.cache.get(
                    prompt,
                    action,
                    tenant_id=tenant_id,
                    scope=cache_scope,
                )
                if cached is not None:
                    result = self._finalise_review(
                        cached.score,
                        cached.h_logical,
                        cached.h_factual,
                        action,
                    )
                    span.set_attribute("coherence.score", cached.score)
                    span.set_attribute("coherence.approved", result[0])
                    span.set_attribute("coherence.cached", True)
                    return result
            h_logic, h_fact, coherence, evidence = self._heuristic_coherence(
                prompt,
                action,
                tenant_id=tenant_id,
            )

            cross_turn = None
            _skip_cross_turn = (
                self._auto_dialogue_profile
                and not self._use_prompt_as_premise
                and self._nli is not None
                and self._nli.model_available
                and self._detect_task_type(prompt, action) == "dialogue"
            )
            if session is not None and len(session) > 0 and not _skip_cross_turn:
                ctx = session.context_text
                if ctx.strip() and self._nli:
                    cross_turn = self._nli.score(ctx, action)
                    h_logic = 0.7 * h_logic + 0.3 * cross_turn
                    total_divergence = self.W_LOGIC * h_logic + self.W_FACT * h_fact
                    coherence = 1.0 - total_divergence
                    # Re-apply no-KB calibration after cross-turn blend
                    nli_ok = self._nli is not None and self._nli.model_available
                    if (
                        nli_ok
                        and abs(h_fact - DIVERGENCE_NEUTRAL) < 1e-9
                        and evidence is None
                    ):
                        cal_lo = 1.0 - self.W_LOGIC - self.W_FACT * DIVERGENCE_NEUTRAL
                        cal_hi = 1.0 - self.W_FACT * DIVERGENCE_NEUTRAL
                        cal_span = cal_hi - cal_lo
                        if cal_span > 1e-9:
                            coherence = max(
                                0.0, min(1.0, (coherence - cal_lo) / cal_span)
                            )

            if self.cache:
                self.cache.put(
                    prompt,
                    action,
                    coherence,
                    h_logic,
                    h_fact,
                    tenant_id=tenant_id,
                    scope=cache_scope,
                )

            # Always detect task type for explainability
            task_type = self._detect_task_type(prompt, action)

            # Adaptive threshold: select per-task-type threshold
            effective_threshold = self.threshold
            if self._adaptive_threshold_enabled and self._task_type_thresholds:
                effective_threshold = self._task_type_thresholds.get(
                    task_type,
                    self.threshold,
                )

            # Meta-classifier: dataset-type mode predicts which sub-dataset
            # the input resembles, then applies the optimal NLI threshold
            # for that dataset. Falls back to per-task-type if uncertain.
            meta_clf = self._get_meta_classifier()
            if meta_clf is not None:
                nli_threshold, meta_conf = meta_clf.predict_threshold(
                    prompt,
                    action,
                )
                if nli_threshold is not None:
                    # NLI-scale to coherence-scale
                    effective_threshold = self.W_FACT + self.W_LOGIC * nli_threshold

            result = self._finalise_review(
                coherence,
                h_logic,
                h_fact,
                action,
                evidence,
                threshold_override=effective_threshold,
                detected_task_type=task_type,
            )
            if cross_turn is not None:
                result[1].cross_turn_divergence = cross_turn
            if session is not None:
                if self._nli and self._nli.model_available:
                    try:
                        report = session.update_contradictions(
                            action, lambda p, h: self._nli.score(p, h)
                        )
                        result[1].contradiction_index = report.contradiction_index
                    except Exception:
                        self.logger.warning(
                            "Contradiction tracking failed", exc_info=True
                        )
                session.add_turn(prompt, action, result[1].score)
            # Injection detection (when enabled)
            inj_detector = self._get_injection_detector()
            if inj_detector is not None:
                try:
                    inj = inj_detector.detect(intent=prompt, response=action)
                    result[1].injection_risk = inj.injection_risk
                except Exception:
                    self.logger.warning("Injection detection failed", exc_info=True)

            span.set_attribute("coherence.score", result[1].score)
            span.set_attribute("coherence.approved", result[0])
            span.set_attribute("coherence.cached", False)
            span.set_attribute("coherence.h_logical", h_logic)
            span.set_attribute("coherence.h_factual", h_fact)
            span.set_attribute("coherence.warning", result[1].warning)
            if result[1].injection_risk is not None:
                span.set_attribute("coherence.injection_risk", result[1].injection_risk)
            return result

    # â"€â"€ Batch API (coalesced NLI) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    def review_batch(
        self,
        items: list[tuple[str, str]],
        tenant_id: str = "",
    ) -> list[tuple[bool, CoherenceScore]]:
        """Batch-review a list of (prompt, response) pairs.

        When NLI is available, batches logical and factual divergence
        through ``NLIScorer.score_batch()`` (2 GPU forward passes total
        instead of 2*N).  Falls back to sequential ``review()`` for
        items that need special handling (dialogue, summarization, rust
        backend, or when NLI is unavailable).
        """
        if not items:
            return []
        nli_ok = (
            self._nli is not None
            and self._nli.model_available
            and self._rust_scorer is None
            and not self._use_prompt_as_premise
        )
        if not nli_ok or len(items) < 2:
            return [self.review(p, a, tenant_id=tenant_id) for p, a in items]

        # Partition: batchable (standard path) vs fallback (dialogue etc.)
        batch_idx: list[int] = []
        fallback_idx: list[int] = []
        for i, (prompt, _action) in enumerate(items):
            if self._auto_dialogue_profile and self._detect_task_type(
                prompt, _action
            ) in ("dialogue", "summarization"):
                fallback_idx.append(i)
            else:
                batch_idx.append(i)

        results: list[tuple[bool, CoherenceScore] | None] = [None] * len(items)

        # Sequential fallback for dialogue/special items
        for i in fallback_idx:
            results[i] = self.review(items[i][0], items[i][1], tenant_id=tenant_id)

        if not batch_idx:
            return [r for r in results if r is not None]

        # Coalesced NLI: batch logical pairs
        assert self._nli is not None  # guarded by nli_ok check above
        logic_pairs = [(items[i][0], items[i][1]) for i in batch_idx]
        h_logics = self._nli.score_batch(logic_pairs)

        # Factual: retrieve KB context per item, batch NLI where possible
        h_facts: list[float] = []
        evidences: list[ScoringEvidence | None] = []
        fact_pairs: list[tuple[str, str]] = []
        fact_pair_map: list[int] = []  # maps fact_pairs index → batch position
        for pos, i in enumerate(batch_idx):
            prompt = items[i][0]
            if self.ground_truth_store:
                ctx = self.ground_truth_store.retrieve_context(
                    prompt,
                    tenant_id=tenant_id,
                )
            else:
                ctx = None
            if ctx and self._nli:
                fact_pairs.append((ctx, items[i][1]))
                fact_pair_map.append(pos)
                h_facts.append(0.0)  # placeholder
                evidences.append(None)
            else:
                h_facts.append(DIVERGENCE_NEUTRAL)
                evidences.append(None)

        if fact_pairs:
            fact_scores = self._nli.score_batch(fact_pairs)
            for j, fs in enumerate(fact_scores):
                h_facts[fact_pair_map[j]] = fs

        # Assemble coherence scores
        nli_available = True
        for pos, i in enumerate(batch_idx):
            h_logic = h_logics[pos]
            h_fact = h_facts[pos]
            evidence = evidences[pos]
            coherence = 1.0 - (self.W_LOGIC * h_logic + self.W_FACT * h_fact)

            # No-KB calibration
            fact_is_neutral = abs(h_fact - DIVERGENCE_NEUTRAL) < 1e-9
            if nli_available and fact_is_neutral and evidence is None:
                lo = 1.0 - self.W_LOGIC - self.W_FACT * DIVERGENCE_NEUTRAL
                hi = 1.0 - self.W_FACT * DIVERGENCE_NEUTRAL
                span = hi - lo
                if span > 1e-9:
                    coherence = max(0.0, min(1.0, (coherence - lo) / span))

            # Match review() finalisation: task-type, adaptive threshold,
            # meta-classifier — ensures batch/single parity.
            prompt = items[i][0]
            task_type = self._detect_task_type(prompt, items[i][1])
            effective_threshold = self.threshold
            if self._adaptive_threshold_enabled and self._task_type_thresholds:
                effective_threshold = self._task_type_thresholds.get(
                    task_type, self.threshold
                )
            meta_clf = self._get_meta_classifier()
            if meta_clf is not None:
                nli_threshold, _meta_conf = meta_clf.predict_threshold(
                    prompt, items[i][1]
                )
                if nli_threshold is not None:
                    effective_threshold = self.W_FACT + self.W_LOGIC * nli_threshold

            results[i] = self._finalise_review(
                coherence,
                h_logic,
                h_fact,
                items[i][1],
                evidence,
                threshold_override=effective_threshold,
                detected_task_type=task_type,
            )

        return [r for r in results if r is not None]

    # â"€â"€ Async API â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    async def areview(
        self,
        prompt: str,
        action: str,
        session=None,
        tenant_id: str = "",
    ) -> tuple[bool, CoherenceScore]:
        """Async version of review() – offloads NLI inference to a thread pool."""
        import functools

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(
                self.review,
                prompt,
                action,
                session=session,
                tenant_id=tenant_id,
            ),
        )
