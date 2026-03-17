# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Coherence Agent (Main Orchestrator)

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator

from .actor import LLMGenerator, MockGenerator
from .kernel import HaltMonitor
from .knowledge import GroundTruthStore
from .scorer import CoherenceScorer
from .streaming import StreamingKernel
from .types import HaltEvidence, ReviewResult

__all__ = ["CoherenceAgent"]

_PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class CoherenceAgent:
    """Integrated coherence-verification agent.

    Orchestrates:
    - **Generator**: Candidate response generation (mock or real LLM).
    - **Scorer**: Weighted NLI divergence scoring.
    - **Ground Truth Store**: RAG-based fact retrieval.
    - **Safety Kernel**: Output interlock.

    Parameters
    ----------
    llm_api_url : str | None — direct URL to OpenAI-compatible endpoint.
    use_nli : bool | None — enable NLI model scoring.
    provider : str | None — "openai" or "anthropic". Reads API key from env.
        Mutually exclusive with llm_api_url.

    """

    def __init__(
        self,
        llm_api_url=None,
        use_nli=None,
        provider=None,
        fallback=None,
        disclaimer_prefix="[Unverified] ",
        *,
        _scorer=None,
        _store=None,
    ):
        self.logger = logging.getLogger("CoherenceAgent")
        self.fallback = fallback
        self.disclaimer_prefix = disclaimer_prefix

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

        self.store = _store if _store is not None else GroundTruthStore()
        self.scorer = _scorer if _scorer is not None else self._build_scorer(use_nli)
        self.kernel = HaltMonitor()
        self.streaming_kernel = StreamingKernel(
            hard_limit=self.kernel.hard_limit,
            adaptive=True,
        )

    def _build_scorer(self, use_nli):
        """Construct scorer, preferring Rust backend when installed."""
        from .backends import get_backend

        try:
            get_backend("rust")
            from backfire_kernel import BackfireConfig, RustCoherenceScorer

            cfg = BackfireConfig(coherence_threshold=0.6)
            scorer = RustCoherenceScorer(
                config=cfg,
                knowledge_callback=self.store.retrieve_context,
            )
            self.logger.info("Rust CoherenceScorer active (via registry)")
            return scorer
        except (  # pragma: no cover — only when backfire_kernel absent
            KeyError,
            ImportError,
            TypeError,
            ValueError,
            RuntimeError,
            OSError,
        ) as exc:
            self.logger.debug("Rust scorer unavailable (%s) — Python fallback", exc)
        return CoherenceScorer(  # pragma: no cover
            threshold=0.6,
            ground_truth_store=self.store,
            use_nli=use_nli,
        )

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

    _ERROR_MARKERS = ("[Timeout]", "[Error]", "[ConnectionError]", "[Connection Error]")

    def process(self, prompt: str, tenant_id: str = "") -> ReviewResult:
        """Process a prompt end-to-end and return the verified output."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        self.logger.debug("Processing prompt (%d chars)", len(prompt))
        candidates = self.generator.generate_candidates(prompt)

        best, rejected, n = self._score_candidates(candidates, prompt, tenant_id)

        if best[0] is not None:
            return self._emit_approved(best, n)
        return self._handle_rejection(prompt, tenant_id, rejected, n)

    def _score_candidates(self, candidates, prompt, tenant_id):
        """Score all candidates, return (best_approved, best_rejected, count)."""
        best = (None, None, -1.0)  # (text, score, coherence)
        rejected = (None, None, -1.0)

        for i, cand in enumerate(candidates):
            text = cand["text"]
            if any(text.strip().startswith(m) for m in self._ERROR_MARKERS):
                self.logger.warning(
                    "Candidate %d is error text, skipping: %s", i, text[:60]
                )
                continue
            try:
                approved, score = self.scorer.review(prompt, text, tenant_id=tenant_id)
            except TypeError:
                approved, score = self.scorer.review(prompt, text)

            self.logger.info(
                "Candidate %d Coherence=%.4f Approved=%s",
                i,
                score.score,
                approved,
            )

            if approved and score.score > best[2]:
                best = (text, score, score.score)
            elif not approved and score.score > rejected[2]:
                rejected = (text, score, score.score)

        return best, rejected, len(candidates)

    def _emit_approved(self, best, n_candidates):
        """Build ReviewResult for the best approved candidate."""
        text, score, coherence = best

        def coherence_monitor(_token):
            return coherence

        final_output = self.kernel.stream_output([text], coherence_monitor)
        prefix = self.disclaimer_prefix if score and score.warning else ""
        return ReviewResult(
            output=f"{prefix}{final_output}",
            coherence=score,
            halted=False,
            candidates_evaluated=n_candidates,
        )

    def _handle_rejection(self, prompt, tenant_id, rejected, n_candidates):
        """Handle all-candidates-rejected: try fallback or halt."""
        rej_text, rej_score, rej_coherence = rejected

        if self.fallback == "retrieval":
            result = self._retrieval_fallback(
                prompt, tenant_id, rej_score, n_candidates
            )
            if result:
                return result

        if self.fallback == "disclaimer" and rej_text:
            return ReviewResult(
                output="Note: This response could not be fully verified. " + rej_text,
                coherence=rej_score,
                halted=False,
                candidates_evaluated=n_candidates,
                fallback_used=True,
            )

        ev_chunks = []
        nli_scores = None
        if rej_score and rej_score.evidence:
            ev_chunks = rej_score.evidence.chunks
            if rej_score.evidence.chunk_scores:
                nli_scores = rej_score.evidence.chunk_scores
        return ReviewResult(
            output="[HALT]: All candidates rejected.",
            coherence=rej_score,
            halted=True,
            candidates_evaluated=n_candidates,
            halt_evidence=HaltEvidence(
                reason="all_candidates_rejected",
                last_score=rej_coherence,
                evidence_chunks=ev_chunks,
                nli_scores=nli_scores,
                suggested_action="Rephrase the prompt or add relevant facts to the knowledge base.",
            ),
        )

    def _retrieval_fallback(self, prompt, tenant_id, rej_score, n_candidates):
        """Try RAG retrieval as fallback when all candidates rejected."""
        from .vector_store import VectorGroundTruthStore

        if isinstance(self.store, VectorGroundTruthStore):
            context = self.store.retrieve_context(prompt, tenant_id=tenant_id)
            if context and isinstance(context, list):
                context = "; ".join(c.text for c in context)
        else:
            context = self.store.retrieve_context(prompt, tenant_id=tenant_id)
        if context:
            return ReviewResult(
                output=f"Based on verified sources: {context}",
                coherence=rej_score,
                halted=False,
                candidates_evaluated=n_candidates,
                fallback_used=True,
            )
        return None

    async def aprocess(self, prompt: str, tenant_id: str = "") -> ReviewResult:
        """Async version of :meth:`process` via ``run_in_executor``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.process, prompt, tenant_id)

    async def stream(
        self,
        prompt: str,
        tenant_id: str = "",
    ) -> AsyncIterator[tuple[str, float]]:
        """Stream tokens with StreamingKernel oversight.

        Uses sliding window, trend detection, and hard/soft halt from
        ``StreamingKernel``. Yields ``(token, coherence)`` tuples.
        Halting stops future tokens but does not retract delivered ones.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        if not hasattr(self.generator, "stream_tokens"):
            result = self.process(prompt, tenant_id=tenant_id)
            for word in result.output.split():
                yield word, result.coherence.score if result.coherence else 0.0
            return

        self.streaming_kernel.reset_state()
        accumulated: list[str] = []

        def _coherence_cb(token: str) -> float:
            accumulated.append(token)
            text = " ".join(accumulated)
            try:
                _, score = self.scorer.review(prompt, text, tenant_id=tenant_id)
            except TypeError:
                _, score = self.scorer.review(prompt, text)
            return float(score.score)

        async for token in self.generator.stream_tokens(prompt):  # pragma: no branch
            score = _coherence_cb(token)
            yield token, score
            if self.streaming_kernel.check_halt(score):
                return
