# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Coherence Scorer (Dual-Entropy Oversight)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time

from ..enterprise.redactor import PIIRedactor
from .cache import ScoreCache
from .metrics import metrics
from .nli import NLIScorer, nli_available
from .otel import trace_review
from .types import CoherenceScore, EvidenceChunk, ScoringEvidence

__all__ = ["CoherenceScorer"]

# Heuristic divergence defaults (used when NLI model unavailable)
DIVERGENCE_NEUTRAL = 0.5  # no signal → agnostic
DIVERGENCE_ALIGNED = 0.1  # keyword heuristic: "consistent with reality"
DIVERGENCE_CONTRADICTED = 0.9  # keyword heuristic: "opposite is true"

# LLM-as-judge blending constants
LLM_JUDGE_AGREE_DIVERGENCE = 0.2
LLM_JUDGE_DISAGREE_DIVERGENCE = 0.8
LLM_JUDGE_NLI_WEIGHT = 0.7
LLM_JUDGE_LLM_WEIGHT = 0.3  # = 1 - NLI_WEIGHT


class CoherenceScorer:
    """
    Dual-entropy coherence scorer for AI output verification.

    Computes a composite coherence score from two independent signals:
    - **Logical divergence** (H_logical): NLI contradiction probability.
    - **Factual divergence** (H_factual): Ground-truth deviation via RAG.

    The coherence score is ``1 - (W_LOGIC * H_logical + W_FACT * H_factual)``.
    When the score falls below ``threshold``, the output is rejected.

    Parameters
    ----------
    threshold : float — minimum coherence to approve (default 0.5).
    soft_limit : float | None — scores between threshold and soft_limit
        trigger a warning. Default: threshold + 0.1.
    w_logic : float — weight for logical divergence (default 0.6).
    w_fact : float — weight for factual divergence (default 0.4).
        Must satisfy w_logic + w_fact = 1.0.
    strict_mode : bool — when True, disables heuristic fallbacks entirely.
        If NLI model is unavailable and strict_mode is True, divergence
        returns 0.9 (reject) and sets ``strict_mode_rejected=True``.
    history_window : int — rolling history size.
    use_nli : bool | None — True forces NLI, False disables it,
        None (default) auto-detects based on installed packages.
    ground_truth_store : GroundTruthStore | None — fact store for RAG.
    nli_model : str | None — HuggingFace model ID or local path for NLI.
    cache_size : int — LRU score cache max entries (0 to disable).
    cache_ttl : float — cache entry TTL in seconds.
    nli_quantize_8bit : bool — load NLI model with 8-bit quantization.
    nli_device : str | None — torch device for NLI model.
    nli_torch_dtype : str | None — torch dtype ("float16", "bfloat16").
    llm_judge_enabled : bool — escalate to LLM when NLI margin is low.
    llm_judge_confidence_threshold : float — softmax margin below which
        to escalate (default 0.3).
    llm_judge_provider : str — "openai" or "anthropic".
    privacy_mode : bool — redact PII (emails, phones, SSN-like patterns)
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
                f"soft_limit ({self.soft_limit}) must be >= threshold ({threshold})"
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
                    f"w_logic + w_fact must equal 1.0, got {self.W_LOGIC + self.W_FACT}"
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
            )
        else:
            self._nli = None  # type: ignore[assignment]
        self._llm_judge_enabled = llm_judge_enabled or scorer_backend == "hybrid"
        self._llm_judge_threshold = llm_judge_confidence_threshold
        self._llm_judge_provider = llm_judge_provider
        self._llm_judge_model = llm_judge_model
        self._privacy_mode = privacy_mode
        self._redactor = PIIRedactor(enabled=privacy_mode)

    @staticmethod
    def _redact_pii(text: str) -> str:
        """Redact PII from text (static convenience method)."""
        return PIIRedactor(enabled=True)(text)

    # ── LLM-as-judge escalation ───────────────────────────────────────

    _DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-haiku-4-5-20251001",
    }

    def _llm_judge_check(self, prompt: str, response: str, nli_score: float) -> float:
        """Escalate to LLM-as-judge when NLI confidence is low.

        Returns adjusted divergence score. Falls back to nli_score on error.
        When privacy_mode is enabled, PII is redacted before sending.
        """
        import os

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
                    reply = result.choices[0].message.content or ""
                elif self._llm_judge_provider == "anthropic":
                    import anthropic

                    client = anthropic.Anthropic()  # type: ignore[assignment]
                    result = client.messages.create(  # type: ignore[attr-defined]
                        model=model,
                        max_tokens=50,
                        messages=[{"role": "user", "content": judge_prompt}],
                    )
                    reply = result.content[0].text if result.content else ""
                else:
                    return nli_score
            except ImportError as exc:
                self.logger.warning("LLM judge import failed: %s", exc)
                return nli_score
            except Exception as exc:
                self.logger.warning("LLM judge API call failed: %s", exc)
                return nli_score

            llm_agrees = self._parse_judge_reply(reply)
            llm_divergence = (
                LLM_JUDGE_AGREE_DIVERGENCE
                if llm_agrees
                else LLM_JUDGE_DISAGREE_DIVERGENCE
            )
            adjusted = (
                LLM_JUDGE_NLI_WEIGHT * nli_score + LLM_JUDGE_LLM_WEIGHT * llm_divergence
            )
            return max(0.0, min(1.0, adjusted))
        finally:
            metrics.observe("llm_judge_seconds", time.monotonic() - t0)

    @staticmethod
    def _parse_judge_reply(reply: str) -> bool:
        """Parse structured JSON from LLM judge, fall back to string matching."""
        import json as _json

        try:
            data = _json.loads(reply)
            return str(data.get("verdict", "")).upper() == "YES"
        except (ValueError, TypeError, AttributeError):
            return "YES" in reply.upper()

    # ── Factual divergence ────────────────────────────────────────────

    def calculate_factual_divergence(
        self, prompt, text_output, tenant_id: str = ""
    ):
        """Check output against the Ground Truth Store.

        Returns 0.0 (aligned) to 1.0 (hallucinated).
        When strict_mode is True and NLI is unavailable, returns 0.9 (reject).
        """
        if self._rust_scorer is not None:
            _, score_obj = self._rust_scorer.review(prompt, text_output)
            fallback = 1.0 - getattr(score_obj, "score", 0.5)
            return getattr(score_obj, "h_factual", fallback)

        if not self.ground_truth_store:
            return DIVERGENCE_NEUTRAL

        with metrics.timer("factual_retrieval_seconds"):
            context = self.ground_truth_store.retrieve_context(
                prompt, tenant_id=tenant_id
            )
        if not context:
            return DIVERGENCE_NEUTRAL

        if self._nli and self._nli.model_available:
            with metrics.timer("chunked_nli_seconds"):
                score, _ = self._nli.score_chunked(context, text_output)
            return score

        if self.strict_mode:
            return DIVERGENCE_CONTRADICTED

        return self._heuristic_factual(context, text_output)

    def calculate_factual_divergence_with_evidence(
        self, prompt, text_output, tenant_id: str = ""
    ) -> tuple[float, ScoringEvidence | None]:
        """Like calculate_factual_divergence but also returns evidence."""
        if not self.ground_truth_store:
            return DIVERGENCE_NEUTRAL, None

        with metrics.timer("factual_retrieval_seconds"):
            chunks: list[EvidenceChunk] = []
            context: str | None = None
            from .vector_store import VectorGroundTruthStore

            if isinstance(self.ground_truth_store, VectorGroundTruthStore):
                chunks = self.ground_truth_store.retrieve_context_with_chunks(
                    prompt, tenant_id=tenant_id
                )
                if chunks:
                    context = "; ".join(c.text for c in chunks)
            else:
                context = self.ground_truth_store.retrieve_context(
                    prompt, tenant_id=tenant_id
                )
                if context:
                    chunks = [
                        EvidenceChunk(text=context, distance=0.0, source="keyword")
                    ]

        if not context:
            return DIVERGENCE_NEUTRAL, None

        chunk_scores = None
        prem_count = 1
        hyp_count = 1
        if self._nli and self._nli.model_available:
            with metrics.timer("chunked_nli_seconds"):
                nli_score, chunk_scores, prem_count, hyp_count = (
                    self._nli._score_chunked_with_counts(context, text_output)
                )
        elif self.strict_mode:
            nli_score = DIVERGENCE_CONTRADICTED
        else:
            nli_score = self._heuristic_factual(context, text_output)

        should_escalate = (
            self._llm_judge_enabled
            and self._llm_judge_provider
            and (
                self.scorer_backend == "hybrid"
                or abs(nli_score - 0.5) < self._llm_judge_threshold
            )
        )
        if should_escalate:
            nli_score = self._llm_judge_check(prompt, text_output, nli_score)

        evidence = ScoringEvidence(
            chunks=chunks,
            nli_premise=context,
            nli_hypothesis=text_output,
            nli_score=nli_score,
            chunk_scores=chunk_scores,
            premise_chunk_count=prem_count,
            hypothesis_chunk_count=hyp_count,
        )
        return nli_score, evidence

    @staticmethod
    def _heuristic_factual(context, text_output):
        """Word-overlap factual divergence with negation and entity checks.

        Install [nli] for production scoring.
        """
        from ._heuristics import ENTITY_RE, NEGATION_WORDS, STOP_WORDS

        ctx_words = set(re.findall(r"\w+", context.lower())) - STOP_WORDS
        out_words = set(re.findall(r"\w+", text_output.lower())) - STOP_WORDS
        if not ctx_words or not out_words:
            return DIVERGENCE_NEUTRAL

        overlap = len(ctx_words & out_words)
        recall = overlap / len(ctx_words)
        precision = overlap / len(out_words)
        similarity = max(recall, precision)
        divergence = 1.0 - similarity

        # Negation asymmetry: one side negates, other doesn't → +0.25
        ctx_neg = bool(ctx_words & NEGATION_WORDS)
        out_neg = bool(out_words & NEGATION_WORDS)
        if ctx_neg != out_neg:
            divergence += 0.25

        # Novel entities in output not grounded in context → +0.15
        ctx_ents = set(ENTITY_RE.findall(context))
        out_ents = set(ENTITY_RE.findall(text_output))
        novel_ents = out_ents - ctx_ents
        if novel_ents:
            divergence += 0.15

        return max(0.0, min(1.0, divergence))

    # ── Logical divergence ────────────────────────────────────────────

    def calculate_logical_divergence(self, prompt, text_output):
        """Compute logical contradiction probability via NLI.

        When strict_mode is True and NLI is unavailable, returns 0.9 (reject).
        """
        if self._rust_scorer is not None:
            _, score_obj = self._rust_scorer.review(prompt, text_output)
            fallback = 1.0 - getattr(score_obj, "score", 0.5)
            return getattr(score_obj, "h_logical", fallback)

        if self._nli and self._nli.model_available:
            with metrics.timer("chunked_nli_seconds"):
                score, _ = self._nli.score_chunked(prompt, text_output)
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

        from ._heuristics import STOP_WORDS

        p_words = set(re.findall(r"\w+", prompt.lower())) - STOP_WORDS
        o_words = set(re.findall(r"\w+", out)) - STOP_WORDS
        if not p_words or not o_words:
            return DIVERGENCE_NEUTRAL
        similarity = len(p_words & o_words) / len(p_words | o_words)
        return max(0.0, min(1.0, 1.0 - similarity))

    # ── Shared helpers ────────────────────────────────────────────────

    def _heuristic_coherence(self, prompt, action, tenant_id: str = ""):
        """Compute coherence components.

        Returns (h_logical, h_factual, coherence, evidence).
        """
        h_logic = self.calculate_logical_divergence(prompt, action)
        h_fact, evidence = self.calculate_factual_divergence_with_evidence(
            prompt, action, tenant_id=tenant_id
        )
        total_divergence = self.W_LOGIC * h_logic + self.W_FACT * h_fact
        coherence = 1.0 - total_divergence
        return h_logic, h_fact, coherence, evidence

    def _finalise_review(
        self, coherence, h_logic, h_fact, action, evidence=None
    ) -> tuple[bool, CoherenceScore]:
        """Build CoherenceScore, gate on threshold, update history.

        Returns (approved, CoherenceScore).
        """
        approved = coherence >= self.threshold
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

    # ── Composite scoring ─────────────────────────────────────────────

    def compute_divergence(self, prompt, action):
        """
        Compute composite divergence (lower is better).

        Weighted sum: ``W_LOGIC * H_logical + W_FACT * H_factual``.
        """
        h_logic = self.calculate_logical_divergence(prompt, action)
        h_fact = self.calculate_factual_divergence(prompt, action)
        total = (self.W_LOGIC * h_logic) + (self.W_FACT * h_fact)
        self.logger.debug(
            f"Divergence: Logic={h_logic:.2f}, Fact={h_fact:.2f} -> Total={total:.2f}"
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
        session : ConversationSession | None — when provided, cross-turn
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

            if self.cache:
                cached = self.cache.get(prompt, action)
                if cached is not None:
                    result = self._finalise_review(
                        cached.score, cached.h_logical, cached.h_factual, action
                    )
                    span.set_attribute("coherence.score", cached.score)
                    span.set_attribute("coherence.approved", result[0])
                    span.set_attribute("coherence.cached", True)
                    return result
            h_logic, h_fact, coherence, evidence = self._heuristic_coherence(
                prompt, action, tenant_id=tenant_id
            )

            cross_turn = None
            if session is not None and len(session) > 0:
                ctx = session.context_text
                if ctx.strip() and self._nli:
                    cross_turn = self._nli.score(ctx, action)
                    h_logic = 0.7 * h_logic + 0.3 * cross_turn
                    total_divergence = self.W_LOGIC * h_logic + self.W_FACT * h_fact
                    coherence = 1.0 - total_divergence

            if self.cache:
                self.cache.put(prompt, action, coherence, h_logic, h_fact)
            result = self._finalise_review(coherence, h_logic, h_fact, action, evidence)
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

    # ── Async API ──────────────────────────────────────────────────────

    async def areview(
        self,
        prompt: str,
        action: str,
        session=None,
        tenant_id: str = "",
    ) -> tuple[bool, CoherenceScore]:
        """Async version of review() — offloads NLI inference to a thread pool."""
        import functools

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(
                self.review, prompt, action, session=session, tenant_id=tenant_id
            ),
        )

    # ── Backward-compatible aliases ───────────────────────────────────

    def calculate_factual_entropy(self, prompt, text_output):
        """Deprecated: use ``calculate_factual_divergence``."""
        import warnings

        warnings.warn(
            "calculate_factual_entropy is deprecated, use calculate_factual_divergence",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.calculate_factual_divergence(prompt, text_output)

    def calculate_logical_entropy(self, prompt, text_output):
        """Deprecated: use ``calculate_logical_divergence``."""
        import warnings

        warnings.warn(
            "calculate_logical_entropy is deprecated, use calculate_logical_divergence",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.calculate_logical_divergence(prompt, text_output)

    def simulate_future_state(self, prompt, action):
        """Deprecated: use ``compute_divergence``."""
        import warnings

        warnings.warn(
            "simulate_future_state is deprecated, use compute_divergence",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.compute_divergence(prompt, action)

    def review_action(self, prompt, action):
        """Deprecated: use ``review``."""
        import warnings

        warnings.warn(
            "review_action is deprecated, use review",
            DeprecationWarning,
            stacklevel=2,
        )
        approved, cs = self.review(prompt, action)
        return approved, cs.score
