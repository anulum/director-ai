# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Coherence Scorer (Dual-Entropy Oversight)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import logging
import math
import threading
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
        self.history: list[str] = []
        self._history_lock = threading.Lock()
        self.window = history_window
        self.ground_truth_store = ground_truth_store
        self.logger = logging.getLogger("DirectorAI")

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

        Scoring cascade:
        1. NLI — contradiction probability via DeBERTa (requires ``use_nli=True``)
        2. Semantic gate — off-topic detection via sentence-transformers cosine sim
        3. String match — value presence check (handles on-topic factual accuracy)

        Returns:
            0.0  — perfect alignment with ground truth
            1.0  — total hallucination / contradiction
        """
        if not self.ground_truth_store:
            return 0.5

        context = self.ground_truth_store.retrieve_context(prompt)
        if not context:
            return 0.5

        if self.use_nli:
            from .nli import NLIScorer

            nli = NLIScorer(use_model=True)
            if nli.model_available:
                return nli.score(context, text_output)

        semantic = self._semantic_divergence(context, text_output)
        if semantic is not None:
            return semantic

        return self._fact_extraction_divergence(context, text_output)

    @staticmethod
    def _semantic_divergence(context: str, text_output: str) -> float | None:
        """Cosine similarity gate via sentence-transformers embeddings.

        Detects off-topic responses (low similarity → high divergence).
        Returns ``None`` for on-topic responses so the caller can fall
        through to string-match for fine-grained value checking, since
        cosine similarity alone cannot distinguish factually correct
        from factually incorrect when sentences are structurally similar.

        Returns ``None`` when sentence-transformers is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return None

        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb_ctx, emb_out = model.encode([context, text_output])

        dot = float(emb_ctx @ emb_out)
        norm_ctx = float((emb_ctx @ emb_ctx) ** 0.5)
        norm_out = float((emb_out @ emb_out) ** 0.5)
        denom = norm_ctx * norm_out
        if denom < 1e-12:
            return 0.5
        cosine_sim = dot / denom

        # Low similarity → output is clearly off-topic
        if cosine_sim < 0.3:
            return 0.9
        # On-topic → fall through to string-match for value accuracy
        return None

    @staticmethod
    def _fact_extraction_divergence(context: str, text_output: str) -> float:
        """String-match fallback: check if ground-truth values appear in output.

        Parses "key is value" entries from context. Low divergence when
        values match, high when they don't.
        """
        output_lower = text_output.lower()
        entries = [e.strip() for e in context.split(";") if e.strip()]
        if not entries:
            return 0.5

        scores: list[float] = []
        for entry in entries:
            parts = entry.split(" is ", 1)
            if len(parts) != 2:
                continue
            key, value = parts[0].strip(), parts[1].strip()
            key_words = set(key.lower().split())
            if not any(w in output_lower for w in key_words):
                continue
            if value.lower() in output_lower:
                scores.append(0.1)
            else:
                scores.append(0.9)

        if not scores:
            return 0.5
        return sum(scores) / len(scores)

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

    def _heuristic_coherence(
        self, prompt: str, action: str
    ) -> tuple[float, float, float]:
        """Compute clamped divergences and heuristic coherence.

        Returns (h_logic, h_fact, coherence) with all values in [0, 1].
        """
        h_logic = max(0.0, min(1.0, self.calculate_logical_divergence(prompt, action)))
        h_fact = max(0.0, min(1.0, self.calculate_factual_divergence(prompt, action)))
        coherence = max(
            0.0, min(1.0, 1.0 - (self.W_LOGIC * h_logic + self.W_FACT * h_fact))
        )
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
