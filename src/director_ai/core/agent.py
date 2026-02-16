# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Coherence Agent (Main Orchestrator)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import logging

from .actor import LLMGenerator, MockGenerator
from .kernel import SafetyKernel
from .knowledge import GroundTruthStore
from .scorer import CoherenceScorer
from .types import ReviewResult


class CoherenceAgent:
    """
    Integrated coherence-verification agent.

    Orchestrates:
    - **Generator**: Candidate response generation (mock or real LLM).
    - **Scorer**: Dual-entropy coherence oversight.
    - **Ground Truth Store**: RAG-based fact retrieval.
    - **Safety Kernel**: Hardware-level output interlock.

    The pipeline is a recursive feedback loop: each candidate output is
    scored *before* emission, and only the highest-coherence candidate
    that passes the threshold is forwarded through the safety kernel.
    """

    def __init__(self, llm_api_url=None):
        if llm_api_url:
            self.generator = LLMGenerator(llm_api_url)
            print(f"CoherenceAgent: Connected to LLM at {llm_api_url}")
        else:
            self.generator = MockGenerator()
            print("CoherenceAgent: Using Mock Generator (Simulation Mode)")

        self.store = GroundTruthStore()
        self.scorer = CoherenceScorer(threshold=0.6, ground_truth_store=self.store)
        self.kernel = SafetyKernel()

        self.logger = logging.getLogger("CoherenceAgent")
        logging.basicConfig(level=logging.INFO)

    def process(self, prompt):
        """
        Process a prompt end-to-end and return the verified output.

        Returns:
            A ``ReviewResult`` with the final output, coherence score,
            halt status, and number of candidates evaluated.
        """
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
