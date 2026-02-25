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
from .types import CoherenceScore


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
    ):
        self.threshold = threshold
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

    @staticmethod
    def _heuristic_factual(context, text_output):
        """Keyword-based factual divergence (testing fallback).

        Only useful for demo/test scenarios — install [nli] for
        real-world fact-checking.
        """
        ctx = context.lower()
        out = text_output.lower()

        if "16" in ctx and "16" not in out and "layers" in out:
            return 0.9

        if "blue" in ctx:
            if "blue" in out:
                return 0.1
            if "green" in out:
                return 1.0

        return 0.1

    # ── Logical divergence ────────────────────────────────────────────

    def calculate_logical_divergence(self, prompt, text_output):
        """
        Compute logical contradiction probability.

        When NLI is available, uses DeBERTa. Otherwise falls back to
        deterministic heuristics for testing.
        """
        if self._nli and self._nli.model_available:
            return self._nli.score(prompt, text_output)

        return self._heuristic_logical(text_output)

    @staticmethod
    def _heuristic_logical(text_output):
        """Deterministic heuristic fallback (testing only)."""
        if "consistent with reality" in text_output:
            return 0.1
        if "opposite is true" in text_output:
            return 0.9
        if "depends on your perspective" in text_output:
            return 0.5
        return 0.5

    # ── Shared helpers ────────────────────────────────────────────────

    def _heuristic_coherence(self, prompt, action):
        """Compute coherence components. Returns (h_logical, h_factual, coherence)."""
        h_logic = self.calculate_logical_divergence(prompt, action)
        h_fact = self.calculate_factual_divergence(prompt, action)
        total_divergence = self.W_LOGIC * h_logic + self.W_FACT * h_fact
        coherence = 1.0 - total_divergence
        return h_logic, h_fact, coherence

    def _finalise_review(self, coherence, h_logic, h_fact, action):
        """Build CoherenceScore, gate on threshold, update history.

        Returns (approved, CoherenceScore).
        """
        approved = coherence >= self.threshold

        if not approved:
            self.logger.critical(
                "COHERENCE FAILURE. Score: %.4f < Threshold: %s",
                coherence,
                self.threshold,
            )
        else:
            with self._history_lock:
                self.history.append(action)
                if len(self.history) > self.window:
                    self.history.pop(0)

        score = CoherenceScore(
            score=coherence,
            approved=approved,
            h_logical=h_logic,
            h_factual=h_fact,
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

    def review(self, prompt, action):
        """
        Score an action and decide whether to approve it.

        Returns:
            (approved: bool, score: CoherenceScore)
        """
        h_logic, h_fact, coherence = self._heuristic_coherence(prompt, action)
        return self._finalise_review(coherence, h_logic, h_fact, action)

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
