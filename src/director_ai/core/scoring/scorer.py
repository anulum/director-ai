# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Coherence Scorer (Weighted NLI Divergence)
"""Composite scoring pipeline for response-level hallucination checks."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

from ..cache import ScoreCache
from ..metrics import metrics
from ..redactor import PIIRedactor
from ._divergence import (
    DIVERGENCE_ALIGNED as DIVERGENCE_ALIGNED,
)
from ._divergence import (
    DIVERGENCE_CONTRADICTED as DIVERGENCE_CONTRADICTED,
)
from ._divergence import (
    DIVERGENCE_NEUTRAL as DIVERGENCE_NEUTRAL,
)
from ._llm_judge import LLMJudge
from ._review_pipeline import ReviewPipelineMixin
from ._task_scoring import _DIALOGUE_TURN_RE
from .nli import NLIScorer, nli_available
from .reasoning_scorer import ReasoningScorer
from .scorer_config import ScorerConfig

__all__ = ["CoherenceScorer", "_DIALOGUE_TURN_RE"]

# scorer_backend values handled natively here: NLIScorer checkpoints (deberta,
# minicheck, onnx), the built-in heuristic surfaces (lite, rules), hybrid
# (→deberta + judge), and the rust FFI. Any other value — "embed", "nli-lite",
# or a third-party plugin — is resolved through the scorer-backend registry by
# _build_registry_nli. "rules" stays heuristic-only (its calibrated profile
# behaviour); only genuinely model-backed registered backends are routed.
_NATIVE_NLI_BACKENDS = frozenset(
    {"deberta", "minicheck", "onnx", "lite", "rules", "hybrid", "rust", "backfire"}
)


class CoherenceScorer(ReviewPipelineMixin):
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
    require_model_backed_nli : bool – when True, fail closed unless a
        model-backed NLI backend is available (DeBERTa/ONNX/MiniCheck/Rust).
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
        threshold: float = 0.5,
        history_window: int = 5,
        use_nli: bool | None = None,
        ground_truth_store: Any | None = None,
        nli_model: str | None = None,
        soft_limit: float | None = None,
        w_logic: float | None = None,
        w_fact: float | None = None,
        strict_mode: bool = False,
        require_model_backed_nli: bool = False,
        cache_size: int = 0,
        cache_ttl: float = 300.0,
        nli_quantize_8bit: bool = False,
        nli_device: str | None = None,
        nli_torch_dtype: str | None = None,
        llm_judge_enabled: bool = False,
        llm_judge_confidence_threshold: float = 0.3,
        llm_judge_provider: str = "",
        llm_judge_model: str = "",
        llm_judge_model_revision: str | None = None,
        llm_judge_rubric: bool = False,
        llm_judge_ensemble: int = 1,
        scorer_backend: str = "deberta",
        onnx_path: str | None = None,
        nli_devices: list[str] | None = None,
        onnx_batch_size: int = 16,
        onnx_flush_timeout_ms: float = 10.0,
        privacy_mode: bool = False,
        cache: ScoreCache | None = None,
        nli_max_length: int = 512,
        nli_revision: str | None = None,
        reasoning_enabled: bool = False,
        reasoning_provider: str = "",
        reasoning_model: str = "",
        reasoning_model_revision: str | None = None,
        reasoning_escalation_margin: float = 0.15,
        minicheck_variant: str = "deberta-v3-large",
    ) -> None:
        """Initialise backend, cache, threshold, and escalation state.

        The constructor preserves the historical keyword surface while routing
        each backend option into one review pipeline. It validates score bounds,
        soft-limit ordering, and divergence weights before creating model or
        cache state.
        """
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
        self.require_model_backed_nli = require_model_backed_nli
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
        self.history: list[str] = []
        self.window = history_window
        self.ground_truth_store = ground_truth_store
        self.logger = logging.getLogger("DirectorAI")
        self._history_lock = threading.Lock()

        self.cache: ScoreCache | None
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

        # _nli is declared up-front so mypy does not narrow to the
        # first branch's concrete type. ShardedNLIScorer is not a
        # subclass but duck-types the full NLIScorer surface the
        # callers reach into — the cast at the sharded assignment
        # below documents that contract.
        self._nli: NLIScorer | None = None

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
        elif scorer_backend not in _NATIVE_NLI_BACKENDS:
            # Registered non-native backend (embed, nli-lite, or a third-party
            # backend registered via register_backend): route it through the
            # scorer-backend registry as a custom NLI surface so the configured
            # backend actually scores, instead of silently using the heuristic.
            self._nli = self._build_registry_nli(scorer_backend, nli_device)
        elif self.use_nli and nli_devices and len(nli_devices) > 1:
            from .sharded_nli import ShardedNLIScorer

            # Duck-typed sharded variant — exposes the same surface
            # as NLIScorer for the call sites that need it.
            self._nli = cast(
                NLIScorer,
                ShardedNLIScorer(
                    devices=nli_devices,
                    use_model=True,
                    model_name=nli_model,
                    backend=nli_backend,
                    quantize_8bit=nli_quantize_8bit,
                    torch_dtype=nli_torch_dtype,
                    onnx_path=onnx_path,
                    onnx_batch_size=onnx_batch_size,
                    onnx_flush_timeout_ms=onnx_flush_timeout_ms,
                ),
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
                minicheck_variant=minicheck_variant,
            )
        self._minicheck_variant = minicheck_variant
        self._privacy_mode = privacy_mode
        self._redactor = PIIRedactor(enabled=privacy_mode)
        self._parallel_pool: ThreadPoolExecutor | None = None
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
        self._verified_scorer_enabled = False
        self._verified_scorer_atomic = True
        self._verified_scorer_evidence_top_k = 3
        self._verified_scorer_low_confidence_margin = 0.10
        self._verified_scorer_min_coverage = 0.50
        self._verified_scorer_task_types = {"rag", "summarization"}
        self._adaptive_threshold_enabled = False
        self._task_type_thresholds: dict[str, float] = {}
        self._conformal_predictor = None
        self._self_consistency_scorer = None
        self._self_consistency_weight = 0.25
        self._chunk_overlap_ratio = 0.5
        self._qa_premise_ratio = 0.7
        self._confidence_weighted_agg = False
        self._meta_classifier_path = ""
        self._meta_classifier: Any | None = None
        self._meta_classifier_lock = threading.Lock()
        self._adaptive_router: Any | None = None  # set via enable_adaptive_retrieval()
        self._adaptive_threshold_fail_closed = False
        self._dry_run = False  # when True, log but never reject
        self._cost_analyser: Any | None = (
            None  # set by config.build_scorer() when cost_tracking_enabled
        )

        # LLM-as-judge subsystem (composed — see _llm_judge.py)
        self._judge = LLMJudge(
            provider=llm_judge_provider
            if (llm_judge_enabled or scorer_backend == "hybrid")
            else "",
            model=llm_judge_model,
            model_revision=llm_judge_model_revision,
            confidence_threshold=llm_judge_confidence_threshold,
            device=nli_device,
            privacy_mode=privacy_mode,
            rubric=llm_judge_rubric,
            ensemble_n=llm_judge_ensemble,
        )
        # Backward-compat aliases used by tests
        self._llm_judge_enabled = self._judge.enabled
        self._llm_judge_provider = llm_judge_provider
        self._llm_judge_threshold = llm_judge_confidence_threshold

        # Tier-6 reasoning escalation (composed — see reasoning_scorer.py).
        # Disabled by default; fires only on borderline scores when enabled.
        self._reasoning = ReasoningScorer(
            provider=reasoning_provider if reasoning_enabled else "",
            model=reasoning_model,
            model_revision=reasoning_model_revision,
            escalation_margin=reasoning_escalation_margin,
            device=nli_device,
            privacy_mode=privacy_mode,
        )

        # Injection detection: set via enable_injection_detection()
        self._injection_lock = threading.Lock()
        self._injection_detector: Any | None = None
        self._injection_fail_closed = False
        self._nli_fallback_lock = threading.Lock()
        self._nli_fallback_incident_stages: set[str] = set()

    def _build_registry_nli(self, name: str, device: str | None) -> NLIScorer | None:
        """Wrap a registry-resolved ``ScorerBackend`` as the NLI scoring surface.

        Resolves ``name`` through the scorer-backend registry (``embed``,
        ``nli-lite``, or any backend registered via ``register_backend``) and
        injects the instance into an ``NLIScorer`` as a custom backend. Returns
        ``None`` — leaving the scorer on the heuristic path — when the backend
        is unavailable (e.g. its optional dependency is not installed), unless
        ``strict_mode`` is set, in which case the failure propagates.

        Parameters
        ----------
        name : str
            The configured ``scorer_backend`` value to resolve.
        device : str or None
            Device hint forwarded to the backend (``None`` lets the backend pick
            its own default, e.g. CPU for the embedding scorer).

        Returns
        -------
        NLIScorer or None
            An NLIScorer delegating to the resolved backend, or ``None`` on
            graceful fallback.
        """
        from .backends import get_backend

        try:
            backend = get_backend(name)(**({"device": device} if device else {}))
        except (KeyError, ImportError, OSError, AttributeError, ValueError) as exc:
            if self.strict_mode:
                raise
            self.logger.warning(
                "scorer_backend %r unavailable (%s); falling back to heuristic",
                name,
                exc,
            )
            return None
        return NLIScorer(backend=backend, use_model=True)

    @classmethod
    def from_config(
        cls,
        config: ScorerConfig,
        *,
        ground_truth_store: Any = None,
        cache: Any = None,
    ) -> CoherenceScorer:
        """Build a scorer from a grouped :class:`ScorerConfig`.

        Value settings come from *config*; the runtime dependencies
        (``ground_truth_store`` and ``cache``) are injected separately so the
        config stays serialisable. Equivalent to the per-argument constructor.
        """
        return cls(
            ground_truth_store=ground_truth_store,
            cache=cache,
            **config.to_kwargs(),
        )

    # -- Backward-compat proxies for judge internals (used by tests) ----

    @property
    def _local_judge_model(self) -> Any:
        return self._judge._local_judge_model

    @_local_judge_model.setter
    def _local_judge_model(self, value: Any) -> None:
        self._judge._local_judge_model = value

    @property
    def _local_judge_tokenizer(self) -> Any:
        return self._judge._local_judge_tokenizer

    @_local_judge_tokenizer.setter
    def _local_judge_tokenizer(self, value: Any) -> None:
        self._judge._local_judge_tokenizer = value

    @property
    def _local_judge_device(self) -> Any:
        return self._judge._local_judge_device

    @_local_judge_device.setter
    def _local_judge_device(self, value: Any) -> None:
        self._judge._local_judge_device = value

    @property
    def _judge_cache(self) -> Any:
        return self._judge._judge_cache

    # Names mirror the class-constant style on :class:`LLMJudge`
    # so tests can reach them via ``scorer._JUDGE_CACHE_MAX``
    # without knowing the internal judge object.
    _JUDGE_CACHE_MAX = property(lambda self: self._judge._JUDGE_CACHE_MAX)
    _JUDGE_RETRY_MAX = property(lambda self: self._judge._JUDGE_RETRY_MAX)

    @property
    def _llm_judge_model(self) -> str:
        return self._judge.model

    @property
    def _task_judge_thresholds(self) -> dict[str, float]:
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
        mc_scorer: Any,
        source: str,
        summary: str,
    ) -> tuple[float, list[float], list[str]]:
        """Backward-compat proxy for minicheck_claim_coverage."""
        from ._task_scoring import minicheck_claim_coverage

        return minicheck_claim_coverage(mc_scorer, source, summary)

    def close(self) -> None:
        """Shut down internal thread pool."""
        if self._parallel_pool is not None:
            self._parallel_pool.shutdown(wait=False)
            self._parallel_pool = None

    def __del__(self) -> None:
        """Release the lazily-created parallel scoring pool during teardown."""
        pool = getattr(self, "_parallel_pool", None)
        if pool is not None:
            pool.shutdown(wait=False)

    def _get_parallel_pool(self) -> ThreadPoolExecutor:
        """Create the review parallelism pool only when parallel scoring is used."""
        if self._parallel_pool is None:
            self._parallel_pool = ThreadPoolExecutor(max_workers=2)
        return self._parallel_pool

    _BUNDLED_CLASSIFIER = "models/dataset_type_classifier.json"

    def _get_meta_classifier(self) -> Any | None:
        """Lazy-load trained meta-classifier from pickle."""
        if self._meta_classifier is not None:
            return self._meta_classifier

        with self._meta_classifier_lock:
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
            except (ImportError, FileNotFoundError, ValueError) as exc:
                metrics.inc_labeled(
                    "adaptive_threshold_classifier_load_failures_total",
                    labels={"reason": type(exc).__name__},
                )
                if self._adaptive_threshold_fail_closed:
                    raise RuntimeError(
                        f"Adaptive threshold classifier unavailable at {path}: {exc}",
                    ) from exc
                self.logger.warning("Meta-classifier unavailable at %s: %s", path, exc)
                self._meta_classifier_path = ""
                return None
            except Exception as exc:
                metrics.inc_labeled(
                    "adaptive_threshold_classifier_load_failures_total",
                    labels={"reason": type(exc).__name__},
                )
                if self._adaptive_threshold_fail_closed:
                    raise RuntimeError(
                        f"Adaptive threshold classifier unavailable at {path}: {exc}",
                    ) from exc
                self.logger.warning(
                    "Meta-classifier load failed unexpectedly at %s (%s): %s",
                    path,
                    type(exc).__name__,
                    exc,
                )
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

    # -- Summarization routing (used by DivergenceMixin) ---------------

    def _get_minicheck_scorer(self) -> NLIScorer | None:
        """Lazily create a MiniCheck NLI scorer for summarisation routing."""
        if hasattr(self, "_minicheck_nli"):
            return self._minicheck_nli

        try:
            mc = NLIScorer(
                use_model=True,
                backend="minicheck",
                minicheck_variant=getattr(
                    self, "_minicheck_variant", "deberta-v3-large"
                ),
            )
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
        require_model_backed_nli: bool = False,
        fail_closed_on_error: bool = False,
    ) -> None:
        """Enable output-side injection detection on every review() call."""
        from ..safety.injection import InjectionDetector

        sanitizer = None
        try:
            from ..safety.sanitizer import InputSanitizer

            sanitizer = InputSanitizer()
        except Exception:
            self.logger.debug("InputSanitizer unavailable for injection detection")

        try:
            detector = InjectionDetector(
                nli_scorer=self._nli,
                sanitizer=sanitizer,
                injection_threshold=injection_threshold,
                drift_threshold=drift_threshold,
                injection_claim_threshold=injection_claim_threshold,
                baseline_divergence=baseline_divergence,
                stage1_weight=stage1_weight,
                require_model_backed_nli=require_model_backed_nli,
            )
        except RuntimeError as exc:
            metrics.inc_labeled(
                "injection_detector_init_failures_total",
                labels={"reason": type(exc).__name__},
            )
            with self._injection_lock:
                self._injection_detector = None
                self._injection_fail_closed = False
            raise
        with self._injection_lock:
            self._injection_detector = detector
            self._injection_fail_closed = fail_closed_on_error
        self.logger.info(
            "Injection detection enabled (threshold=%.2f)", injection_threshold
        )

    def _get_injection_detector(self) -> Any | None:
        """Return the InjectionDetector if enabled, else None."""
        with self._injection_lock:
            return self._injection_detector

    def _get_injection_runtime_state(self) -> tuple[Any | None, bool]:
        """Return (detector, fail_closed) snapshot atomically."""
        with self._injection_lock:
            return self._injection_detector, self._injection_fail_closed

    def _has_model_backed_nli(self) -> bool:
        """Return True when scoring has a model-backed contradiction path."""
        if self._rust_scorer is not None:
            return True
        if self._nli is None or not self._nli.model_available:
            return False
        return getattr(self._nli, "backend", "") != "lite"

    def _enforce_model_backed_nli_requirement(self) -> None:
        """Fail closed when model-backed NLI is required but unavailable."""
        if not self.require_model_backed_nli:
            return
        if self._has_model_backed_nli():
            return
        metrics.inc_labeled(
            "nli_fallback_incidents_total",
            labels={
                "stage": "coherence",
                "reason": "required_model_backed_nli_unavailable",
            },
        )
        raise RuntimeError(
            "CoherenceScorer requires model-backed NLI, but only heuristic/lite scoring is available",
        )

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

    def _record_nli_fallback_incident(self, *, stage: str, reason: str) -> None:
        """Emit a once-per-stage incident when model-backed NLI is unavailable."""
        with self._nli_fallback_lock:
            if stage in self._nli_fallback_incident_stages:
                return
            self._nli_fallback_incident_stages.add(stage)
        metrics.inc_labeled(
            "nli_fallback_incidents_total",
            labels={"stage": stage, "reason": reason},
        )
        self.logger.error(
            "NLI fallback incident: stage=%s reason=%s strict_mode=%s backend=%s",
            stage,
            reason,
            self.strict_mode,
            self.scorer_backend,
        )
