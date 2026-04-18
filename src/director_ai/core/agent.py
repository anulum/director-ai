# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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
from typing import TYPE_CHECKING

from .actor import LLMGenerator, MockGenerator
from .retrieval.knowledge import GroundTruthStore
from .runtime.kernel import HaltMonitor
from .runtime.streaming import StreamingKernel
from .scoring.scorer import CoherenceScorer
from .types import HaltEvidence, ReviewResult

if TYPE_CHECKING:
    from .containment import ContainmentGuard, RealityAnchor
    from .cyber_physical import GroundingHook, GroundingVerdict, PhysicalAction
    from .zk_attestation import (
        CrossOrgPassport,
        PassportVerdict,
        PassportVerifier,
    )

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
        api_key=None,
        *,
        _scorer=None,
        _store=None,
        containment_guard: ContainmentGuard | None = None,
        containment_anchor: RealityAnchor | None = None,
        grounding_hook: GroundingHook | None = None,
        passport_verifier: PassportVerifier | None = None,
    ):
        self.logger = logging.getLogger("CoherenceAgent")
        self.fallback = fallback
        self.disclaimer_prefix = disclaimer_prefix

        if provider and llm_api_url:
            raise ValueError("provider and llm_api_url are mutually exclusive")

        if (containment_guard is None) != (containment_anchor is None):
            raise ValueError(
                "containment_guard and containment_anchor must be "
                "configured together (both or neither)"
            )

        if provider:
            self.generator = self._build_provider(provider, api_key=api_key)
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

        # Opt-in safety hooks — None means "not engaged", preserving
        # the existing end-to-end behaviour bit-for-bit when the
        # caller does not configure any of them.
        self.containment_guard = containment_guard
        self.containment_anchor = containment_anchor
        self.grounding_hook = grounding_hook
        self.passport_verifier = passport_verifier

    def _build_scorer(self, use_nli):
        """Construct scorer, preferring Rust backend when installed."""
        from .scoring.backends import get_backend

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
    def _build_provider(name: str, api_key: str | None = None):
        from ..integrations.providers import AnthropicProvider, OpenAIProvider

        env_key = _PROVIDER_ENV_KEYS.get(name)
        if not env_key:
            raise ValueError(f"Unknown provider {name!r}; use 'openai' or 'anthropic'")
        resolved_key = api_key or os.environ.get(env_key, "")
        if not resolved_key:
            raise ValueError(
                f"API key for {name!r} not supplied; pass api_key=... or set {env_key}"
            )
        if name == "openai":
            return OpenAIProvider(api_key=resolved_key)
        return AnthropicProvider(api_key=resolved_key)

    _ERROR_MARKERS = ("[Timeout]", "[Error]", "[ConnectionError]", "[Connection Error]")

    def process(self, prompt: str, tenant_id: str = "") -> ReviewResult:
        """Process a prompt end-to-end and return the verified output."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        self.logger.debug("Processing prompt (%d chars)", len(prompt))
        candidates = self.generator.generate_candidates(prompt)

        best, rejected, n = self._score_candidates(candidates, prompt, tenant_id)

        if best[0] is not None:
            result = self._emit_approved(best, n)
        else:
            result = self._handle_rejection(prompt, tenant_id, rejected, n)

        return self._apply_containment_guard(result, prompt)

    def _apply_containment_guard(
        self, result: ReviewResult, prompt: str
    ) -> ReviewResult:
        """If a containment guard is configured, scan the output text
        against the session's reality anchor. A ``"block"`` verdict
        converts the result into a halted ReviewResult whose
        ``halt_evidence`` carries the guard's findings for audit.
        """
        guard = self.containment_guard
        anchor = self.containment_anchor
        if guard is None or anchor is None:
            return result
        verdict = guard.check({"text": result.output, "prompt": prompt}, anchor)
        if verdict.decision != "block":
            return result

        reasons = "; ".join(f"{f.category}:{f.severity}" for f in verdict.findings)
        if verdict.anchor_reason:
            reasons = verdict.anchor_reason + (f"; {reasons}" if reasons else "")
        return ReviewResult(
            output="[CONTAINMENT-BLOCK]: Output suppressed by containment guard.",
            coherence=result.coherence,
            halted=True,
            candidates_evaluated=result.candidates_evaluated,
            halt_evidence=HaltEvidence(
                reason="containment_block",
                last_score=(
                    result.coherence.score if result.coherence is not None else 0.0
                ),
                evidence_chunks=[],
                nli_scores=None,
                suggested_action=("Review the containment findings: " + reasons),
            ),
            fallback_used=result.fallback_used,
        )

    def verify_physical_action(self, action: PhysicalAction) -> GroundingVerdict:
        """Screen a proposed physical action against the configured
        grounding hook. Raises :class:`RuntimeError` if no hook is
        configured — callers opt in explicitly.
        """
        if self.grounding_hook is None:
            raise RuntimeError("grounding_hook not configured on this CoherenceAgent")
        return self.grounding_hook.evaluate(action)

    def verify_passport(self, passport: CrossOrgPassport) -> PassportVerdict:
        """Run the configured passport verifier against *passport*.
        Raises :class:`RuntimeError` when no verifier is attached."""
        if self.passport_verifier is None:
            raise RuntimeError(
                "passport_verifier not configured on this CoherenceAgent"
            )
        return self.passport_verifier.verify(passport)

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

    def _emit_approved(self, best, n_candidates) -> ReviewResult:
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

    def _handle_rejection(
        self, prompt, tenant_id, rejected, n_candidates
    ) -> ReviewResult:
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

    def _retrieval_fallback(
        self, prompt, tenant_id, rej_score, n_candidates
    ) -> ReviewResult | None:
        """Try RAG retrieval as fallback when all candidates rejected."""
        from .retrieval.vector_store import VectorGroundTruthStore

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
