# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” Coherence Scorer (Weighted NLI Divergence)

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ...enterprise.redactor import PIIRedactor
from ..cache import ScoreCache
from ..metrics import metrics
from ..otel import trace_review
from ..types import CoherenceScore, EvidenceChunk, ScoringEvidence
from .nli import NLIScorer, nli_available

__all__ = ["CoherenceScorer"]

# Heuristic divergence defaults (used when NLI model unavailable)
DIVERGENCE_NEUTRAL = 0.5  # no signal â†’ agnostic
DIVERGENCE_ALIGNED = 0.1  # keyword heuristic: "consistent with reality"
DIVERGENCE_CONTRADICTED = 0.9  # keyword heuristic: "opposite is true"

# LLM-as-judge blending constants
LLM_JUDGE_AGREE_DIVERGENCE = 0.2
LLM_JUDGE_DISAGREE_DIVERGENCE = 0.8
LLM_JUDGE_NLI_WEIGHT = 0.7
LLM_JUDGE_LLM_WEIGHT = 0.3  # = 1 - NLI_WEIGHT

# Dialogue detection: â‰Ą2 speaker-turn markers â†’ dialogue task.
# Uses (?:^|\s) to match speakers at line start OR after whitespace
# (HaluEval puts all turns on one line: "[Human]: text [Assistant]: text").
_DIALOGUE_TURN_RE = re.compile(
    r"(?:^|\s)(?:"
    r"(?:User|Human|Customer|Student|Interviewer|Speaker"
    r"|Assistant|AI|Bot|Agent|Interviewee|System)"
    r"[\s\d]*:"
    r"|\[(?:User|Human|Assistant|AI|System)\]"
    r")",
    re.IGNORECASE,
)


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
    threshold : float â€” minimum coherence to approve (default 0.5).
    soft_limit : float | None â€” scores between threshold and soft_limit
        trigger a warning. Default: threshold + 0.1.
    w_logic : float â€” weight for logical divergence (default 0.6).
    w_fact : float â€” weight for factual divergence (default 0.4).
        Must satisfy w_logic + w_fact = 1.0.
    strict_mode : bool â€” when True, disables heuristic fallbacks entirely.
        If NLI model is unavailable and strict_mode is True, divergence
        returns 0.9 (reject) and sets ``strict_mode_rejected=True``.
    history_window : int â€” rolling history size.
    use_nli : bool | None â€” True forces NLI, False disables it,
        None (default) auto-detects based on installed packages.
    ground_truth_store : GroundTruthStore | None â€” fact store for RAG.
    nli_model : str | None â€” HuggingFace model ID or local path for NLI.
    cache_size : int â€” LRU score cache max entries (0 to disable).
    cache_ttl : float â€” cache entry TTL in seconds.
    nli_quantize_8bit : bool â€” load NLI model with 8-bit quantization.
    nli_device : str | None â€” torch device for NLI model.
    nli_torch_dtype : str | None â€” torch dtype ("float16", "bfloat16").
    llm_judge_enabled : bool â€” escalate to LLM when NLI margin is low.
    llm_judge_confidence_threshold : float â€” softmax margin below which
        to escalate (default 0.3).
    llm_judge_provider : str â€” "openai" or "anthropic".
    privacy_mode : bool â€” redact PII (emails, phones, SSN-like patterns)
        before sending text to external LLM judge.

    """

    W_LOGIC = 0.6
    W_FACT = 0.4

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
            )
        else:
            self._nli = None  # type: ignore[assignment]
        self._llm_judge_enabled = llm_judge_enabled or scorer_backend == "hybrid"
        self._llm_judge_threshold = llm_judge_confidence_threshold
        self._llm_judge_provider = llm_judge_provider
        self._llm_judge_model = llm_judge_model
        self._judge_cache: dict[int, float] = {}
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
        self._dialogue_nli_baseline = (
            0.80  # expected NLI divergence for correct dialogue
        )
        self._summarization_nli_baseline = 0.20  # HaluEval 200: 25.5%â†’10.5% FPR
        self._claim_coverage_enabled = True
        self._claim_support_threshold = 0.6  # HaluEval 200: 10.5%â†’2.0% FPR
        self._rag_claim_decomposition = True  # per-sentence scoring for RAG path
        self._retrieval_abstention_threshold = (
            0.0  # 0 = disabled; >0 = min retrieval score
        )
        self._claim_coverage_alpha = (
            0.4  # blend: alpha * (1-coverage) + (1-alpha) * layer_a
        )
        self._adaptive_threshold_enabled = False
        self._task_type_thresholds: dict[str, float] = {}
        self._chunk_overlap_ratio = 0.5
        self._qa_premise_ratio = 0.7
        self._confidence_weighted_agg = False
        self._meta_classifier_path = ""
        self._meta_classifier = None
        # Phase 6B: per-task-type judge escalation thresholds
        # Wider borderline zone â†’ more aggressive judge routing
        self._task_judge_thresholds: dict[str, float] = {
            "dialogue": 0.35,
            "summarization": 0.25,
            "qa": 0.30,
            "fact_check": 0.20,
            "default": self._llm_judge_threshold,
        }

        # Local DeBERTa-base judge model (replaces LLM API calls)
        self._local_judge_model = None
        self._local_judge_tokenizer = None
        self._local_judge_device = "cpu"
        if llm_judge_provider == "local" and llm_judge_model:
            self._init_local_judge(llm_judge_model, nli_device)

    def close(self) -> None:
        """Shut down internal thread pool."""
        self._parallel_pool.shutdown(wait=False)

    def __del__(self) -> None:
        pool = getattr(self, "_parallel_pool", None)
        if pool is not None:
            pool.shutdown(wait=False)

    def _init_local_judge(
        self,
        model_path: str,
        device: str | None = None,
    ):  # pragma: no cover
        """Load local DeBERTa-base judge model for borderline escalation."""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._local_judge_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._local_judge_model = (
                AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=False,
                )
            )
            self._local_judge_device = device or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._local_judge_model.to(self._local_judge_device)
            self._local_judge_model.eval()
            self.logger.info(
                "Local judge loaded: %s on %s",
                model_path,
                self._local_judge_device,
            )
        except Exception as exc:
            self.logger.warning("Failed to load local judge model: %s", exc)
            self._local_judge_model = None
            self._local_judge_tokenizer = None

    def _local_judge_check(self, prompt: str, response: str, nli_score: float) -> float:
        """Local DeBERTa-base binary judge (replaces LLM API call).

        Returns adjusted divergence via the same 70/30 blending as the LLM path.
        Falls back to raw nli_score if model unavailable.
        """
        if self._local_judge_model is None or self._local_judge_tokenizer is None:
            return nli_score
        return self._local_judge_infer(prompt, response, nli_score)

    def _local_judge_infer(  # pragma: no cover â€” requires torch, tested locally
        self,
        prompt: str,
        response: str,
        nli_score: float,
    ) -> float:
        """Run local judge forward pass and blend with NLI score."""
        import torch

        cache_key = hash((prompt[:500], response[:500]))
        cached = self._judge_cache.get(cache_key)
        if cached is not None:
            metrics.inc("llm_judge_cache_hits")
            return cached
        metrics.inc("llm_judge_cache_misses")

        t0 = time.monotonic()
        metrics.inc("llm_judge_escalations")

        judge_input = (
            f"NLI divergence: {nli_score:.2f}\n"
            f"Context: {prompt[:400]}\n"
            f"Response: {response[:400]}"
        )
        tokenizer = self._local_judge_tokenizer
        model = self._local_judge_model
        assert tokenizer is not None and model is not None
        inputs = tokenizer(
            judge_input,
            return_tensors="pt",
            max_length=384,
            truncation=True,
        )
        inputs = {k: v.to(self._local_judge_device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        judge_agrees = probs[0] > 0.5  # class 0 = approve
        judge_conf = float(max(probs))  # confidence = max class probability
        llm_divergence = (
            LLM_JUDGE_AGREE_DIVERGENCE
            if judge_agrees
            else LLM_JUDGE_DISAGREE_DIVERGENCE
        )
        llm_w = LLM_JUDGE_LLM_WEIGHT * judge_conf
        nli_w = 1.0 - llm_w
        adjusted = max(0.0, min(1.0, nli_w * nli_score + llm_w * llm_divergence))

        if len(self._judge_cache) >= self._JUDGE_CACHE_MAX:
            self._judge_cache.pop(next(iter(self._judge_cache)))
        self._judge_cache[cache_key] = adjusted

        metrics.observe("llm_judge_seconds", time.monotonic() - t0)
        return adjusted

    # â”€â”€ LLM-as-judge escalation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    _DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-haiku-4-5-20251001",
    }
    _JUDGE_CACHE_MAX = 256
    _JUDGE_RETRY_MAX = 3
    _JUDGE_RETRY_BACKOFF = (0.5, 1.0)

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
        """True when the judge (LLM or local) should be consulted."""
        if not self._llm_judge_enabled or not self._llm_judge_provider:
            return False
        if self._llm_judge_provider == "local" and self._local_judge_model is None:
            return False
        threshold = self._task_judge_thresholds.get(
            task_type,
            self._llm_judge_threshold,
        )
        return bool(abs(nli_score - 0.5) < threshold)

    def _llm_judge_check(self, prompt: str, response: str, nli_score: float) -> float:
        """Escalate to judge when NLI confidence is low.

        Routes to local DeBERTa judge or external LLM API depending on
        llm_judge_provider. Returns adjusted divergence score.
        """
        if self._llm_judge_provider == "local":
            return self._local_judge_check(prompt, response, nli_score)

        import os

        cache_key = hash((prompt[:500], response[:500]))
        cached = self._judge_cache.get(cache_key)
        if cached is not None:
            metrics.inc("llm_judge_cache_hits")
            return cached
        metrics.inc("llm_judge_cache_misses")

        t0 = time.monotonic()
        metrics.inc("llm_judge_escalations")

        model = (
            os.environ.get("DIRECTOR_LLM_JUDGE_MODEL")
            or getattr(self, "_llm_judge_model", "")
            or self._DEFAULT_MODELS.get(self._llm_judge_provider, "")
        )

        p_text = prompt[:500]
        r_text = response[:500]
        if self._privacy_mode:
            p_text = self._redactor(p_text)
            r_text = self._redactor(r_text)

        judge_prompt = (
            f"Given the prompt: {p_text}\n"
            f"Response: {r_text}\n"
            f"NLI divergence score: {nli_score:.3f}\n"
            "Is this response factually correct? "
            'Reply with JSON: {"verdict": "YES" or "NO", "confidence": 0-100}'
        )
        try:
            reply = self._call_llm_judge(model, judge_prompt, nli_score)
            if reply is None:
                return nli_score

            llm_agrees, judge_conf = self._parse_judge_reply(reply)
            llm_divergence = (
                LLM_JUDGE_AGREE_DIVERGENCE
                if llm_agrees
                else LLM_JUDGE_DISAGREE_DIVERGENCE
            )
            # Scale LLM influence by judge confidence: at conf=1.0 full
            # weight (0.3), at conf=0.5 half weight (0.15).
            llm_w = LLM_JUDGE_LLM_WEIGHT * judge_conf
            nli_w = 1.0 - llm_w
            adjusted = max(
                0.0,
                min(1.0, nli_w * nli_score + llm_w * llm_divergence),
            )
            if len(self._judge_cache) >= self._JUDGE_CACHE_MAX:
                self._judge_cache.pop(next(iter(self._judge_cache)))
            self._judge_cache[cache_key] = adjusted
            return adjusted
        finally:
            metrics.observe("llm_judge_seconds", time.monotonic() - t0)

    def _call_llm_judge(
        self,
        model: str,
        judge_prompt: str,
        fallback: float,
    ) -> str | None:
        """Call LLM provider with retry on transient errors.

        Returns reply text, or None on permanent/exhausted failure.
        """
        last_exc: Exception | None = None
        for attempt in range(self._JUDGE_RETRY_MAX):
            try:
                if self._llm_judge_provider == "openai":
                    import openai

                    client = openai.OpenAI()
                    result = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": judge_prompt}],
                        max_tokens=50,
                        response_format={"type": "json_object"},
                    )
                    return result.choices[0].message.content or ""
                if self._llm_judge_provider == "anthropic":
                    import anthropic

                    client = anthropic.Anthropic()  # type: ignore[assignment]
                    result = client.messages.create(  # type: ignore[attr-defined]
                        model=model,
                        max_tokens=50,
                        messages=[{"role": "user", "content": judge_prompt}],
                    )
                    return result.content[0].text if result.content else ""
                return None
            except ImportError as exc:
                self.logger.warning("LLM judge import failed: %s", exc)
                return None
            except Exception as exc:
                last_exc = exc
                if attempt < len(self._JUDGE_RETRY_BACKOFF):
                    time.sleep(self._JUDGE_RETRY_BACKOFF[attempt])

        self.logger.warning(
            "LLM judge failed after %d attempts: %s",
            self._JUDGE_RETRY_MAX,
            last_exc,
        )
        return None

    @staticmethod
    def _parse_judge_reply(reply: str) -> tuple[bool, float]:
        """Parse verdict and confidence from LLM judge JSON.

        Returns (agrees: bool, confidence: 0.0-1.0).
        Falls back to string matching with 0.5 confidence.
        """
        import json as _json

        try:
            data = _json.loads(reply)
            agrees = str(data.get("verdict", "")).upper() == "YES"
            raw_conf = float(data.get("confidence", 50))
            conf = max(0.0, min(1.0, raw_conf / 100.0))
            return agrees, conf
        except (ValueError, TypeError, AttributeError):
            return "YES" in reply.upper(), 0.5

    # â”€â”€ Task-aware scoring profiles â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @staticmethod
    def _detect_task_type(prompt: str) -> str:
        """Detect task type from prompt content.

        Returns one of: ``"dialogue"``, ``"summarization"``, ``"rag"``,
        ``"fact_check"``, ``"qa"``, or ``"default"``.
        """
        matches = _DIALOGUE_TURN_RE.findall(prompt)
        if len(matches) >= 2:
            return "dialogue"

        lower = prompt.lower()
        if any(
            kw in lower
            for kw in ("summarize", "summary", "summarise", "tldr", "abstract")
        ):
            return "summarization"
        if any(
            kw in lower
            for kw in (
                "based on the context",
                "based on the following",
                "given the document",
                "given the passage",
                "retrieved",
                "source document",
                "reference text",
            )
        ):
            return "rag"
        if any(
            kw in lower
            for kw in ("verify", "fact-check", "is it true", "claim", "support")
        ):
            return "fact_check"
        if "?" in prompt or any(
            kw in lower
            for kw in ("answer the question", "based on the", "according to")
        ):
            return "qa"
        return "default"

    def _resolve_agg_profile(self, prompt: str) -> tuple[str, str, str, str]:
        """Return (fact_inner, fact_outer, logic_inner, logic_outer) agg settings.

        For dialogue prompts the aggregation override is less important than
        the bidirectional-NLI + baseline-calibration path handled by
        ``_heuristic_coherence``.  This method still returns min-mean for
        dialogue so that callers who bypass ``_heuristic_coherence`` get a
        reasonable default.
        """
        fi, fo = self._fact_inner_agg, self._fact_outer_agg
        li, lo = self._logic_inner_agg, self._logic_outer_agg

        if (
            self._auto_dialogue_profile
            and not self._use_prompt_as_premise
            and fi == "max"
            and fo == "max"
            and li == "max"
            and lo == "max"
            and self._detect_task_type(prompt) == "dialogue"
        ):
            return "min", "mean", "min", "mean"

        return fi, fo, li, lo

    # â”€â”€ Dialogue-specific scoring â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _dialogue_factual_divergence(
        self,
        prompt: str,
        response: str,
        tenant_id: str = "",
    ) -> tuple[float, ScoringEvidence | None]:
        """Bidirectional NLI scoring with baseline calibration for dialogue.

        The standard NLI approach (FactCG "supported/not-supported") gives
        ~0.92 divergence for *correct* dialogue responses because new
        information isn't "supported" by the conversation context.  This
        method:

        1. Scores both directions:
           - forward ``score(context, response)``
           - reverse ``score(response, context)``
        2. Takes the **minimum** (most lenient direction).
        3. Applies **baseline calibration**:
           ``adjusted = max(0, (raw - baseline) / (1 - baseline))``
           to shift out expected dialogue divergence (default baseline=0.80).

        Benchmark (HaluEval dialogue, n=200, L4 GPU):
        - Forward-only default:  FPR=97.5%
        - Bidir + bl=0.80 t=0.50: FPR=4.5%
        - Bidir + bl=0.85 t=0.50: FPR=4.5%
        """
        if self._nli is None or not self._nli.model_available:
            raise RuntimeError("NLI model required for dialogue factual divergence")

        # Forward pass: full evidence path
        h_fact_fwd, evidence = self.calculate_factual_divergence_with_evidence(
            prompt,
            response,
            tenant_id,
            _inner_agg="min",
            _outer_agg="mean",
        )

        # Reverse pass: does the response support the context?
        # Uses score_chunked for chunking support on long texts.
        h_fact_rev, _ = self._nli.score_chunked(
            response,
            prompt,
            inner_agg="min",
            outer_agg="mean",
            premise_ratio=0.4,
        )

        # Bidirectional minimum (most lenient direction)
        raw_div = min(h_fact_fwd, h_fact_rev)

        # Baseline calibration: shift expected dialogue divergence to 0
        baseline = self._dialogue_nli_baseline
        denom = 1.0 - baseline
        adjusted = max(0.0, (raw_div - baseline) / denom) if denom > 1e-9 else raw_div

        return adjusted, evidence

    # â”€â”€ Summarization-specific scoring â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _summarization_factual_divergence(
        self,
        prompt: str,
        response: str,
        tenant_id: str = "",
    ) -> tuple[float, ScoringEvidence | None]:
        """Bidirectional NLI + claim coverage for summarization.

        Layer A: bidirectional NLI with baseline calibration.
        Layer C: decompose summary into claims, score each against source,
        compute coverage = supported_claims / total_claims.

        Final divergence = alpha * (1 - coverage) + (1 - alpha) * layer_a.

        Benchmark (HaluEval summarization, n=200, L4 GPU):
        - Forward-only (Phase 3):  FPR=25.5%
        - Bidir + bl=0.20 (Layer A): FPR=10.5%
        - Layer A + C (alpha=0.4, st=0.6): FPR=2.0%
        """
        if self._nli is None or not self._nli.model_available:
            raise RuntimeError(
                "NLI model required for summarization factual divergence"
            )

        # Layer A: bidirectional NLI
        h_fact_fwd, evidence = self.calculate_factual_divergence_with_evidence(
            prompt,
            response,
            tenant_id,
            _inner_agg=self._fact_inner_agg,
            _outer_agg=self._fact_outer_agg,
        )

        h_fact_rev, _ = self._nli.score_chunked(
            response,
            prompt,
            inner_agg="min",
            outer_agg="mean",
            premise_ratio=self._premise_ratio,
        )

        raw_div = min(h_fact_fwd, h_fact_rev)

        baseline = self._summarization_nli_baseline
        if baseline > 0.0:
            layer_a = max(0.0, (raw_div - baseline) / (1.0 - baseline))
        else:
            layer_a = raw_div

        # Layer C: claim decomposition + coverage scoring with attribution
        if self._claim_coverage_enabled:
            coverage, per_claim_divs, claims, attributions = (
                self._nli.score_claim_coverage_with_attribution(
                    prompt,
                    response,
                    support_threshold=self._claim_support_threshold,
                )
            )
            alpha = self._claim_coverage_alpha
            adjusted = alpha * (1.0 - coverage) + (1.0 - alpha) * layer_a

            if evidence is not None:
                evidence.claim_coverage = coverage
                evidence.per_claim_divergences = per_claim_divs
                evidence.claims = claims
                evidence.attributions = attributions
        else:
            adjusted = layer_a

        return adjusted, evidence

    # â”€â”€ Factual divergence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
        task_type = self._detect_task_type(prompt)
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
            if self._should_escalate(score):
                score = self._llm_judge_check(prompt, text_output, score)
            return score

        if not self.ground_truth_store:
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

        if self._should_escalate(score):
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
        task_type = self._detect_task_type(prompt)
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
            if self._should_escalate(nli_score):
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

        if self._should_escalate(nli_score):
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

    # â”€â”€ Logical divergence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

    # â”€â”€ Shared helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _heuristic_coherence(self, prompt, action, tenant_id: str = ""):
        """Compute coherence components.

        Returns (h_logical, h_factual, coherence, evidence).
        H_logical and H_factual run in parallel â€” vector retrieval overlaps
        with the logical NLI forward pass.

        For dialogue prompts (auto-detected), uses bidirectional NLI with
        baseline calibration instead of standard forward-only scoring.
        Logical divergence is skipped for dialogue (entailment is meaningless).
        """
        # Eager-load NLI in the main thread to avoid PyTorch 2.6 dispatch
        # corruption when from_pretrained runs inside a ThreadPoolExecutor
        # worker after a CUDA model was already loaded.
        if self._nli is not None:
            self._nli._ensure_model()

        # Task-aware aggregation profile
        fact_ia, fact_oa, logic_ia, logic_oa = self._resolve_agg_profile(prompt)

        # â”€â”€ Dialogue path: bidirectional NLI + baseline calibration â”€â”€â”€â”€
        # Logical entailment is meaningless for dialogue (a question
        # doesn't entail its answer).  Standard NLI gives ~0.92 divergence
        # for correct responses.  The dialogue path uses min(fwd, rev) with
        # baseline calibration to bring FPR from 97.5% â†’ 4.5% at t=0.50.
        _is_dialogue = (
            self._auto_dialogue_profile
            and not self._use_prompt_as_premise
            and self._nli is not None
            and self._nli.model_available
            and self._detect_task_type(prompt) == "dialogue"
        )

        if _is_dialogue:
            h_logic = 0.0
            h_fact, evidence = self._dialogue_factual_divergence(
                prompt,
                action,
                tenant_id,
            )
        # â”€â”€ Summarization path: bidirectional NLI when prompt-as-premise â”€â”€
        # Abstractive rephrasing causes forward NLI to over-reject.  The
        # reverse direction (summaryâ†’document) catches paraphrases.
        elif (
            self._use_prompt_as_premise
            and self._nli is not None
            and self._nli.model_available
            and self.W_LOGIC < 1e-9
        ):
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
        # compress to [0.25, 0.55].  Rescale to [0, 1] so thresholds are
        # meaningful.  With KB context the factual component carries real
        # signal and no rescaling is needed.
        nli_available = self._nli is not None and self._nli.model_available
        fact_is_neutral = abs(h_fact - DIVERGENCE_NEUTRAL) < 1e-9
        if nli_available and fact_is_neutral and evidence is None:
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
                self.threshold,
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
        score = CoherenceScore(
            score=coherence,
            approved=approved,
            h_logical=h_logic,
            h_factual=h_fact,
            evidence=evidence,
            warning=warning,
            strict_mode_rejected=strict_rejected,
        )
        return approved, score

    # â”€â”€ Composite scoring â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
        session : ConversationSession | None â€” when provided, cross-turn
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
                and self._detect_task_type(prompt) == "dialogue"
            )
            if session is not None and len(session) > 0 and not _skip_cross_turn:
                ctx = session.context_text
                if ctx.strip() and self._nli:
                    cross_turn = self._nli.score(ctx, action)
                    h_logic = 0.7 * h_logic + 0.3 * cross_turn
                    total_divergence = self.W_LOGIC * h_logic + self.W_FACT * h_fact
                    coherence = 1.0 - total_divergence

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

            # Adaptive threshold: select per-task-type threshold
            effective_threshold = self.threshold
            if self._adaptive_threshold_enabled and self._task_type_thresholds:
                task_type = self._detect_task_type(prompt)
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
                    # NLI-scale â†’ coherence-scale: c = 0.4 + 0.6 * nli_t
                    effective_threshold = 0.4 + 0.6 * nli_threshold

            result = self._finalise_review(
                coherence,
                h_logic,
                h_fact,
                action,
                evidence,
                threshold_override=effective_threshold,
            )
            if cross_turn is not None:
                result[1].cross_turn_divergence = cross_turn
            if session is not None:
                session.add_turn(prompt, action, result[1].score)
            span.set_attribute("coherence.score", result[1].score)
            span.set_attribute("coherence.approved", result[0])
            span.set_attribute("coherence.cached", False)
            span.set_attribute("coherence.h_logical", h_logic)
            span.set_attribute("coherence.h_factual", h_fact)
            span.set_attribute("coherence.warning", result[1].warning)
            return result

    # â”€â”€ Batch API (coalesced NLI) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def review_batch(
        self,
        items: list[tuple[str, str]],
        tenant_id: str = "",
    ) -> list[tuple[bool, CoherenceScore]]:
        """Batch-review with coalesced NLI inference.

        Collects all H_logical pairs into one score_batch() call, runs
        vector retrieval in parallel, then coalesces all H_factual pairs
        into a second score_batch() call. 2 GPU kernel calls total
        regardless of batch size, vs 2*N for per-item review().
        """
        if not items:
            return []
        return [
            self.review(prompt, response, tenant_id=tenant_id)
            for prompt, response in items
        ]

    # â”€â”€ Async API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def areview(
        self,
        prompt: str,
        action: str,
        session=None,
        tenant_id: str = "",
    ) -> tuple[bool, CoherenceScore]:
        """Async version of review() â€” offloads NLI inference to a thread pool."""
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
