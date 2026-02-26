# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Coherence Agent (Main Orchestrator)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import logging
import os

from .actor import LLMGenerator, MockGenerator
from .kernel import SafetyKernel
from .knowledge import GroundTruthStore
from .scorer import CoherenceScorer
from .types import ReviewResult

_PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class CoherenceAgent:
    """
    Integrated coherence-verification agent.

    Orchestrates:
    - **Generator**: Candidate response generation (mock or real LLM).
    - **Scorer**: Dual-entropy coherence oversight.
    - **Ground Truth Store**: RAG-based fact retrieval.
    - **Safety Kernel**: Hardware-level output interlock.

    Parameters
    ----------
    llm_api_url : str | None — direct URL to OpenAI-compatible endpoint.
    use_nli : bool | None — enable NLI model scoring.
    provider : str | None — "openai" or "anthropic". Reads API key from env.
        Mutually exclusive with llm_api_url.
    """

    def __init__(self, llm_api_url=None, use_nli=None, provider=None):
        self.logger = logging.getLogger("CoherenceAgent")

        if provider and llm_api_url:
            raise ValueError("provider and llm_api_url are mutually exclusive")

        if provider:
            self.generator = self._build_provider(provider)
            self.logger.info("Using %s provider", provider)
        elif llm_api_url:
            self.generator = LLMGenerator(llm_api_url)
            self.logger.info("Connected to LLM at %s", llm_api_url)
        else:
            self.generator = MockGenerator()
            self.logger.info("Using Mock Generator (Simulation Mode)")
            if use_nli is None:
                use_nli = False

        self.store = GroundTruthStore()
        self.scorer = CoherenceScorer(
            threshold=0.6, ground_truth_store=self.store, use_nli=use_nli
        )
        self.kernel = SafetyKernel()

    @staticmethod
    def _build_provider(name: str):
        from ..integrations.providers import AnthropicProvider, OpenAIProvider

        env_key = _PROVIDER_ENV_KEYS.get(name)
        if not env_key:
            raise ValueError(f"Unknown provider {name!r}; use 'openai' or 'anthropic'")
        api_key = os.environ.get(env_key, "")
        if not api_key:
            raise ValueError(f"{env_key} not set in environment")
        if name == "openai":
            return OpenAIProvider(api_key=api_key)
        return AnthropicProvider(api_key=api_key)

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

        self.logger.info(f"Received Prompt: '{prompt}'")

        # 1. Generate candidates (feed-forward)
        candidates = self.generator.generate_candidates(prompt)

        best_response = None
        best_score = None
        best_coherence = -1.0

        # 2. Recursive oversight — score each candidate
        for i, cand in enumerate(candidates):
            text = cand["text"]
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

    # ── Backward-compatible alias ─────────────────────────────────────

    def process_query(self, prompt):
        """Alias for ``process`` returning just the output string (backward compat)."""
        result = self.process(prompt)
        return result.output
