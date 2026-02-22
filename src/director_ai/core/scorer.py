# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Coherence Scorer (Dual-Entropy Oversight)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import logging
import math
import warnings

from .types import CoherenceScore


class CoherenceScorer:
    """
    Dual-entropy coherence scorer for AI output verification.

    Computes a composite coherence score from two independent signals:
    - **Logical divergence** (H_logical): NLI-based contradiction probability.
    - **Factual divergence** (H_factual): Ground-truth deviation via RAG retrieval.

    The coherence score is ``1 - (w_logic * H_logical + w_fact * H_factual)``.
    When the score falls below ``threshold``, the output is rejected.
    """

    def __init__(
        self, threshold=0.5, history_window=5, use_nli=False, ground_truth_store=None
    ):
        self.threshold = threshold
        self.history = []
        self.window = history_window
        self.ground_truth_store = ground_truth_store
        self.logger = logging.getLogger("DirectorAI")
        self.logger.setLevel(logging.INFO)

        self.use_nli = use_nli
        if self.use_nli:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.logger.info(
                f"Coherence Scorer initialized with NLI model: {model_name}"
            )

    # ── Factual divergence ────────────────────────────────────────────

    def calculate_factual_divergence(self, prompt, text_output):
        """
        Check output against the Ground Truth Store.

        Returns:
            0.0  — perfect alignment with ground truth
            1.0  — total hallucination / contradiction
        """
        if not self.ground_truth_store:
            return 0.5  # Neutral when no store is configured

        context = self.ground_truth_store.retrieve_context(prompt)
        if not context:
            return 0.5  # Neutral when no relevant facts found

        # Prototype heuristic checks (NLI replaces these in production)
        if "16" in context and "16" not in text_output and "layers" in text_output:
            return 0.9  # Factual hallucination

        if "sky color" in context:
            if "blue" in text_output:
                return 0.1  # Consistent
            if "green" in text_output:
                return 1.0  # Contradiction

        return 0.1  # Default: consistent if no contradiction detected

    # ── Logical divergence ────────────────────────────────────────────

    def calculate_logical_divergence(self, prompt, text_output):
        """
        Compute logical contradiction probability.

        When ``use_nli=True``, uses a DeBERTa NLI model.
        Otherwise falls back to deterministic heuristics for testing.
        """
        if self.use_nli:
            import torch

            input_text = f"{prompt} [SEP] {text_output}"
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True)

            with torch.no_grad():
                logits = self.model(**inputs).logits

            # DeBERTa-mnli classes: 0=entailment, 1=neutral, 2=contradiction
            probs = torch.softmax(logits, dim=1).numpy()[0]
            contradiction_prob = probs[2]
            neutral_prob = probs[1]

            h = float(contradiction_prob * 1.0) + float(neutral_prob * 0.5)
            # Guard against NaN/Inf from malformed logits
            if not math.isfinite(h):
                self.logger.warning("NLI returned non-finite score, defaulting to 0.5")
                return 0.5
            return max(0.0, min(1.0, h))

        # Deterministic mock for testing
        if "consistent with reality" in text_output:
            return 0.1
        elif "opposite is true" in text_output:
            return 0.9
        elif "depends on your perspective" in text_output:
            return 0.5

        # Deterministic default for unknown text (avoids flaky tests)
        return 0.5

    # ── Weight constants ─────────────────────────────────────────────

    W_LOGIC: float = 0.6
    W_FACT: float = 0.4

    # ── Shared scoring helpers ─────────────────────────────────────

    def _heuristic_coherence(self, prompt: str, action: str) -> tuple[float, float, float]:
        """Compute clamped divergences and heuristic coherence.

        Returns (h_logic, h_fact, coherence) with all values in [0, 1].
        """
        h_logic = max(0.0, min(1.0, self.calculate_logical_divergence(prompt, action)))
        h_fact = max(0.0, min(1.0, self.calculate_factual_divergence(prompt, action)))
        coherence = max(0.0, min(1.0, 1.0 - (self.W_LOGIC * h_logic + self.W_FACT * h_fact)))
        return h_logic, h_fact, coherence

    def _finalise_review(
        self, coherence: float, h_logic: float, h_fact: float, action: str
    ) -> tuple[bool, CoherenceScore]:
        """Approve/reject, log, update history, and build CoherenceScore."""
        approved = bool(coherence >= self.threshold)

        if not approved:
            self.logger.critical(
                "COHERENCE FAILURE. Score: %.4f < Threshold: %s",
                coherence,
                self.threshold,
            )
        else:
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

        Weighted sum: ``0.6 * H_logical + 0.4 * H_factual``.
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
        """Deprecated: use ``calculate_factual_divergence``."""
        warnings.warn(
            "calculate_factual_entropy() is deprecated, "
            "use calculate_factual_divergence()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.calculate_factual_divergence(prompt, text_output)

    def calculate_logical_entropy(self, prompt, text_output):
        """Deprecated: use ``calculate_logical_divergence``."""
        warnings.warn(
            "calculate_logical_entropy() is deprecated, "
            "use calculate_logical_divergence()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.calculate_logical_divergence(prompt, text_output)

    def simulate_future_state(self, prompt, action):
        """Deprecated: use ``compute_divergence``."""
        warnings.warn(
            "simulate_future_state() is deprecated, use compute_divergence()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.compute_divergence(prompt, action)

    def review_action(self, prompt, action):
        """Deprecated: use ``review``."""
        warnings.warn(
            "review_action() is deprecated, use review()",
            DeprecationWarning,
            stacklevel=2,
        )
        approved, cs = self.review(prompt, action)
        return approved, cs.score
