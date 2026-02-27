# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Coherence Scorer (Dual-Entropy Oversight)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import threading

from .cache import ScoreCache
from .nli import NLIScorer, nli_available
from .types import CoherenceScore, EvidenceChunk, ScoringEvidence


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
        If NLI model is unavailable and strict_mode is True, logical
        divergence returns 0.5 (neutral) instead of keyword heuristics.
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
    ):
        self.threshold = threshold
        self.soft_limit = soft_limit if soft_limit is not None else threshold + 0.1
        self.strict_mode = strict_mode

        if w_logic is not None or w_fact is not None:
            self.W_LOGIC = w_logic if w_logic is not None else 0.6
            self.W_FACT = w_fact if w_fact is not None else 0.4
        self.history = []
        self.window = history_window
        self.ground_truth_store = ground_truth_store
        self.logger = logging.getLogger("DirectorAI")
        self._history_lock = threading.Lock()
        self.cache = (
            ScoreCache(max_size=cache_size, ttl_seconds=cache_ttl)
            if cache_size > 0
            else None
        )

        if use_nli is None:
            self.use_nli = nli_available()
        else:
            self.use_nli = use_nli

        self._nli = (
            NLIScorer(
                use_model=self.use_nli,
                model_name=nli_model,
                quantize_8bit=nli_quantize_8bit,
                device=nli_device,
                torch_dtype=nli_torch_dtype,
            )
            if self.use_nli
            else None
        )

    # ── Factual divergence ────────────────────────────────────────────

    def calculate_factual_divergence(self, prompt, text_output):
        """Check output against the Ground Truth Store.

        Returns 0.0 (aligned) to 1.0 (hallucinated).
        When strict_mode is True and NLI is unavailable, returns 0.5.
        """
        if not self.ground_truth_store:
            return 0.5

        context = self.ground_truth_store.retrieve_context(prompt)
        if not context:
            return 0.5

        if self._nli and self._nli.model_available:
            return self._nli.score(context, text_output)

        if self.strict_mode:
            return 0.5

        return self._heuristic_factual(context, text_output)

    def calculate_factual_divergence_with_evidence(
        self, prompt, text_output
    ) -> tuple[float, ScoringEvidence | None]:
        """Like calculate_factual_divergence but also returns evidence."""
        if not self.ground_truth_store:
            return 0.5, None

        # Try structured chunks from VectorGroundTruthStore
        chunks: list[EvidenceChunk] = []
        context: str | None = None
        from .vector_store import VectorGroundTruthStore

        if isinstance(self.ground_truth_store, VectorGroundTruthStore):
            chunks = self.ground_truth_store.retrieve_context_with_chunks(prompt)
            if chunks:
                context = "; ".join(c.text for c in chunks)
        else:
            context = self.ground_truth_store.retrieve_context(prompt)
            if context:
                chunks = [EvidenceChunk(text=context, distance=0.0, source="keyword")]

        if not context:
            return 0.5, None

        if self._nli and self._nli.model_available:
            nli_score = self._nli.score(context, text_output)
        elif self.strict_mode:
            nli_score = 0.5
        else:
            nli_score = self._heuristic_factual(context, text_output)

        evidence = ScoringEvidence(
            chunks=chunks,
            nli_premise=context,
            nli_hypothesis=text_output,
            nli_score=nli_score,
        )
        return nli_score, evidence

    @staticmethod
    def _heuristic_factual(context, text_output):
        """Word-overlap factual divergence (no-NLI fallback).

        Uses bidirectional containment: max of recall (context words
        in output) and precision (output words grounded in context).
        Install [nli] for production scoring.
        """
        import re

        ctx_words = set(re.findall(r"\w+", context.lower()))
        out_words = set(re.findall(r"\w+", text_output.lower()))
        if not ctx_words or not out_words:
            return 0.5
        overlap = len(ctx_words & out_words)
        recall = overlap / len(ctx_words)
        precision = overlap / len(out_words)
        similarity = max(recall, precision)
        return max(0.0, min(1.0, 1.0 - similarity))

    # ── Logical divergence ────────────────────────────────────────────

    def calculate_logical_divergence(self, prompt, text_output):
        """Compute logical contradiction probability via NLI.

        When strict_mode is True and NLI is unavailable, returns 0.5.
        """
        if self._nli and self._nli.model_available:
            return self._nli.score(prompt, text_output)

        if self.strict_mode:
            return 0.5

        return self._heuristic_logical(text_output, prompt)

    @staticmethod
    def _heuristic_logical(text_output, prompt=""):
        """Keyword + word-overlap logical divergence (no-NLI fallback).

        Install [nli] for production-grade scoring.
        """
        out = text_output.lower()
        if "consistent with reality" in out:
            return 0.1
        if "opposite is true" in out:
            return 0.9
        if "depends on your perspective" in out:
            return 0.5
        if not prompt:
            return 0.5
        import re

        p_words = set(re.findall(r"\w+", prompt.lower()))
        o_words = set(re.findall(r"\w+", out))
        if not p_words or not o_words:
            return 0.5
        similarity = len(p_words & o_words) / len(p_words | o_words)
        return max(0.0, min(1.0, 1.0 - similarity))

    # ── Shared helpers ────────────────────────────────────────────────

    def _heuristic_coherence(self, prompt, action):
        """Compute coherence components.

        Returns (h_logical, h_factual, coherence, evidence).
        """
        h_logic = self.calculate_logical_divergence(prompt, action)
        h_fact, evidence = self.calculate_factual_divergence_with_evidence(
            prompt, action
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

        score = CoherenceScore(
            score=coherence,
            approved=approved,
            h_logical=h_logic,
            h_factual=h_fact,
            evidence=evidence,
            warning=warning,
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

    def review(self, prompt: str, action: str) -> tuple[bool, CoherenceScore]:
        """Score an action and decide whether to approve it."""
        if self.cache:
            cached = self.cache.get(prompt, action)
            if cached is not None:
                return self._finalise_review(
                    cached.score, cached.h_logical, cached.h_factual, action
                )
        h_logic, h_fact, coherence, evidence = self._heuristic_coherence(prompt, action)
        if self.cache:
            self.cache.put(prompt, action, coherence, h_logic, h_fact)
        return self._finalise_review(coherence, h_logic, h_fact, action, evidence)

    # ── Async API ──────────────────────────────────────────────────────

    async def areview(self, prompt: str, action: str) -> tuple[bool, CoherenceScore]:
        """Async version of review() — offloads NLI inference to a thread pool."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.review, prompt, action)

    # ── Backward-compatible aliases ───────────────────────────────────

    def calculate_factual_entropy(self, prompt, text_output):
        """Alias for ``calculate_factual_divergence`` (backward compat)."""
        return self.calculate_factual_divergence(prompt, text_output)

    def calculate_logical_entropy(self, prompt, text_output):
        """Alias for ``calculate_logical_divergence`` (backward compat)."""
        return self.calculate_logical_divergence(prompt, text_output)

    def simulate_future_state(self, prompt, action):
        """Alias for ``compute_divergence`` (backward compat)."""
        return self.compute_divergence(prompt, action)

    def review_action(self, prompt, action):
        """Alias for ``review`` returning (approved, score_float) (backward compat)."""
        approved, cs = self.review(prompt, action)
        return approved, cs.score
