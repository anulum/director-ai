# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Coherence Agent (Main Orchestrator)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import logging
import warnings

from .actor import LLMGenerator, MockGenerator
from .kernel import SafetyKernel
from .scorer import CoherenceScorer
from .streaming import StreamingKernel, StreamSession
from .types import ReviewResult
from .vector_store import InMemoryBackend, VectorGroundTruthStore

#: Maximum prompt length (characters) accepted before rejection.
MAX_PROMPT_LENGTH = 100_000

#: Maximum candidates to evaluate per request.
MAX_CANDIDATES = 50


class CoherenceAgent:
    """
    Integrated coherence-verification agent.

    Orchestrates:
    - **Generator**: Candidate response generation (mock or real LLM).
    - **Scorer**: Dual-entropy coherence oversight.
    - **Ground Truth Store**: RAG-based fact retrieval.
    - **Safety Kernel**: Software safety gate.

    The pipeline is a recursive feedback loop: each candidate output is
    scored *before* emission, and only the highest-coherence candidate
    that passes the threshold is forwarded through the safety kernel.
    """

    def __init__(self, llm_api_url=None, ground_truth_store=None):
        self.logger = logging.getLogger("CoherenceAgent")
        self.generator: MockGenerator | LLMGenerator

        if llm_api_url:
            self.generator = LLMGenerator(llm_api_url)
            self.logger.info("Connected to LLM at %s", llm_api_url)
        else:
            self.generator = MockGenerator()
            self.logger.info("Using Mock Generator (Simulation Mode)")

        self.store = ground_truth_store or VectorGroundTruthStore(
            InMemoryBackend(), auto_index=False
        )
        self.scorer = CoherenceScorer(threshold=0.6, ground_truth_store=self.store)
        self.kernel = SafetyKernel()

    def process(self, prompt: str) -> "ReviewResult":
        """
        Process a prompt end-to-end and return the verified output.

        Parameters:
            prompt: Non-empty query string.

        Returns:
            A ``ReviewResult`` with the final output, coherence score,
            halt status, and number of candidates evaluated.

        Raises:
            ValueError: If *prompt* is empty or not a string.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if len(prompt) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"prompt exceeds maximum length ({len(prompt)} > {MAX_PROMPT_LENGTH})"
            )

        # Log-safe: truncate and strip control characters
        safe_prompt = prompt[:200].replace("\n", " ").replace("\r", "")
        self.logger.info(
            "Received Prompt: '%s%s'", safe_prompt, "..." if len(prompt) > 200 else ""
        )

        # 1. Generate candidates (feed-forward)
        candidates = self.generator.generate_candidates(prompt)

        # Cap candidate count to prevent resource exhaustion
        if len(candidates) > MAX_CANDIDATES:
            self.logger.warning(
                "Generator returned %d candidates, capping to %d",
                len(candidates),
                MAX_CANDIDATES,
            )
            candidates = candidates[:MAX_CANDIDATES]

        best_response = None
        best_score = None
        best_coherence = -1.0

        # 2. Recursive oversight — score each candidate
        for i, cand in enumerate(candidates):
            if not isinstance(cand, dict) or "text" not in cand:
                self.logger.warning(
                    "Skipping malformed candidate %d: %s", i, type(cand)
                )
                continue
            text = cand["text"]
            if not isinstance(text, str):
                self.logger.warning("Skipping candidate %d: text is not a string", i)
                continue
            approved, score = self.scorer.review(prompt, text)

            self.logger.info(
                f"Candidate {i} Coherence={score.score:.4f} | Approved={approved}"
            )

            if approved and score.score > best_coherence:
                best_coherence = score.score
                best_response = text
                best_score = score

        # 3. Safety kernel output streaming
        if best_response:

            def coherence_monitor(token):
                return best_coherence

            final_output = self.kernel.stream_output([best_response], coherence_monitor)
            return ReviewResult(
                output=f"[AGI Output]: {final_output}",
                coherence=best_score,
                halted=False,
                candidates_evaluated=len(candidates),
            )

        return ReviewResult(
            output=(
                "[SYSTEM HALT]: No coherent response found."
                " Self-termination to prevent divergence."
            ),
            coherence=None,
            halted=True,
            candidates_evaluated=len(candidates),
        )

    def process_streaming(self, prompt: str, provider=None) -> "StreamSession":
        """
        Process a prompt with real-time token streaming and coherence gating.

        Connects an ``LLMProvider.stream_generate()`` token stream to a
        ``StreamingKernel`` for per-token coherence oversight.  The scorer
        evaluates the *accumulated* text (not individual tokens) so that
        coherence checks are meaningful across the full response.

        Parameters:
            prompt: Non-empty query string.
            provider: An ``LLMProvider`` instance with ``stream_generate()``.
                      If *None*, falls back to the mock generator (word tokens).

        Returns:
            A ``StreamSession`` with the emitted tokens and halt status.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        streaming_kernel = StreamingKernel()

        if provider is not None:
            token_gen = provider.stream_generate(prompt)
        else:
            # Fallback: generate one candidate and yield word-level tokens
            candidates = self.generator.generate_candidates(prompt, n=1)
            text = candidates[0]["text"] if candidates else ""
            token_gen = iter(text.split())

        # Accumulate tokens so the scorer sees the full text so far
        accumulated: list[str] = []

        def coherence_callback(token):
            accumulated.append(token)
            text_so_far = " ".join(accumulated)
            _, score = self.scorer.review(prompt, text_so_far)
            return score.score

        return streaming_kernel.stream_tokens(token_gen, coherence_callback)

    # ── Backward-compatible alias ─────────────────────────────────────

    def process_query(self, prompt):
        """Deprecated: use ``process``."""
        warnings.warn(
            "process_query() is deprecated, use process()",
            DeprecationWarning,
            stacklevel=2,
        )
        result = self.process(prompt)
        return result.output
