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
    history_window : int — rolling history size.
    use_nli : bool | None — True forces NLI, False disables it,
        None (default) auto-detects based on installed packages.
    ground_truth_store : GroundTruthStore | None — fact store for RAG.
    nli_model : str | None — HuggingFace model ID or local path for NLI.
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
    ):
        self.threshold = threshold
        self.soft_limit = soft_limit if soft_limit is not None else threshold + 0.1
        self.history = []
        self.window = history_window
        self.ground_truth_store = ground_truth_store
        self.logger = logging.getLogger("DirectorAI")
        self._history_lock = threading.Lock()

        if use_nli is None:
            self.use_nli = nli_available()
        else:
            self.use_nli = use_nli

        self._nli = (
            NLIScorer(use_model=self.use_nli, model_name=nli_model)
            if self.use_nli
            else None
        )

    # ── Factual divergence ────────────────────────────────────────────

    def calculate_factual_divergence(self, prompt, text_output):
        """
        Check output against the Ground Truth Store.

        When NLI is available, uses NLI to compare retrieved context
        against the output. Otherwise falls back to keyword heuristics.

        Returns:
            0.0  — perfect alignment with ground truth
            1.0  — total hallucination / contradiction
        """
        if not self.ground_truth_store:
            return 0.5

        context = self.ground_truth_store.retrieve_context(prompt)
        if not context:
            return 0.5

        if self._nli and self._nli.model_available:
            return self._nli.score(context, text_output)

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
        """
        Compute logical contradiction probability.

        When NLI is available, uses DeBERTa. Otherwise falls back to
        deterministic heuristics for testing.
        """
        if self._nli and self._nli.model_available:
            return self._nli.score(prompt, text_output)

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
        h_logic, h_fact, coherence, evidence = self._heuristic_coherence(prompt, action)
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
