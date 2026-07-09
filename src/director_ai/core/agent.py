# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Coherence Agent (Main Orchestrator)

"""Coherence agent: generate candidate responses, score, and emit verified output."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from .actor import LLMGenerator, MockGenerator
from .calibration.recall_correctness import recall_outcome
from .retrieval.knowledge import GroundTruthStore
from .runtime.kernel import HaltMonitor
from .runtime.streaming import StreamingKernel
from .runtime.streaming_gate import StreamingCoherenceGate, ends_claim
from .scoring.scorer import CoherenceScorer
from .types import HaltEvidence, ReviewResult

# One scored candidate: (text, coherence score object, coherence value).
_ScoredCandidate = tuple[str | None, Any, float]

if TYPE_CHECKING:
    from .calibration.recall_correctness_client import RemanentiaCorrectnessClient
    from .containment import ContainmentGuard, RealityAnchor
    from .cyber_physical import GroundingHook, GroundingVerdict, PhysicalAction
    from .runtime.contradiction_halt import ContradictionHalt
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


def _physical_budget_exhausted(verdict: GroundingVerdict) -> bool:
    event = getattr(verdict, "safety_event", None)
    return event is not None and event.halt_reason == "physical_budget_exceeded"


class CoherenceAgent:
    """Integrated coherence-verification agent.

    Orchestrates:
    - **Generator**: Candidate response generation (test or real LLM).
    - **Scorer**: Weighted NLI divergence scoring.
    - **Ground Truth Store**: RAG-based fact retrieval.
    - **Safety Kernel**: Output interlock.

    Parameters
    ----------
    llm_api_url : str | None — direct URL to OpenAI-compatible endpoint.
    use_nli : bool | None — enable NLI model scoring.
    provider : str | None — "openai" or "anthropic". Reads API key from env.
        Mutually exclusive with llm_api_url.
    production_mode : bool — require an explicit real LLM generator.
    max_candidates : int — how many candidate responses the generator produces
        per prompt (passed as ``n`` to ``generate_candidates``); the best-scoring
        approved candidate is emitted. Sourced from
        ``DirectorConfig.max_candidates`` at the server/CLI/gRPC construction sites.

    """

    def __init__(
        self,
        llm_api_url: str | None = None,
        use_nli: bool | None = None,
        provider: str | None = None,
        fallback: str | None = None,
        disclaimer_prefix: str = "[Unverified] ",
        api_key: str | None = None,
        production_mode: bool = False,
        llm_max_tokens: int = 128,
        llm_temperature: float = 0.8,
        max_candidates: int = 3,
        *,
        _scorer: CoherenceScorer | None = None,
        _store: GroundTruthStore | None = None,
        containment_guard: ContainmentGuard | None = None,
        containment_anchor: RealityAnchor | None = None,
        grounding_hook: GroundingHook | None = None,
        passport_verifier: PassportVerifier | None = None,
        physical_action_mode: str = "warn",
        allow_physical_action_blocking: bool = False,
        contradiction_halt: ContradictionHalt | None = None,
        correctness_feedback: RemanentiaCorrectnessClient | None = None,
    ) -> None:
        self.logger = logging.getLogger("CoherenceAgent")
        self.fallback = fallback
        self.disclaimer_prefix = disclaimer_prefix
        if max_candidates < 1:
            raise ValueError(f"max_candidates must be >= 1; got {max_candidates!r}")
        self.max_candidates = max_candidates

        if provider and llm_api_url:
            raise ValueError("provider and llm_api_url are mutually exclusive")
        if production_mode and not (provider or llm_api_url):
            raise ValueError(
                "production_mode requires provider or llm_api_url; "
                "refusing to use MockGenerator"
            )

        if (containment_guard is None) != (containment_anchor is None):
            raise ValueError(
                "containment_guard and containment_anchor must be "
                "configured together (both or neither)"
            )
        if physical_action_mode not in {"warn", "block"}:
            raise ValueError("physical_action_mode must be 'warn' or 'block'")
        if physical_action_mode == "block" and not allow_physical_action_blocking:
            raise ValueError(
                "physical_action_mode='block' requires "
                "allow_physical_action_blocking=True"
            )

        if provider:
            self.generator = self._build_provider(provider, api_key=api_key)
            self.logger.info("Using %s provider", provider)
        elif llm_api_url:
            self.generator = LLMGenerator(
                llm_api_url,
                max_tokens=llm_max_tokens,
                temperature=llm_temperature,
            )
            self.logger.info("Connected to LLM at %s", llm_api_url)
        else:
            self.generator = MockGenerator()
            self.logger.info("Using MockGenerator for test/demo mode")
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
        self.physical_action_mode = physical_action_mode
        self.allow_physical_action_blocking = allow_physical_action_blocking
        # Opt-in contradiction-driven streaming halt (the working real-time
        # halt). When set, stream() halts on a claim that contradicts retrieved
        # grounding instead of using the coherence kernel.
        self.contradiction_halt = contradiction_halt
        # Opt-in REMANENTIA recall-correctness feedback. When set, process()
        # posts each verification verdict back as the recall's was_correct
        # label, closing the two-label memory loop. None preserves behaviour.
        self.correctness_feedback = correctness_feedback

    def _build_scorer(self, use_nli: bool | None) -> CoherenceScorer:
        """Construct scorer, preferring Rust backend when installed."""
        from .scoring.backends import get_backend

        try:
            get_backend("rust")
            from backfire_kernel import BackfireConfig, RustCoherenceScorer

            cfg = BackfireConfig(coherence_threshold=0.6)
            # The Rust scorer satisfies the same review() contract; the kernel
            # extension is untyped, so it is bound to the CoherenceScorer type here.
            scorer: CoherenceScorer = RustCoherenceScorer(
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
    def _build_provider(name: str, api_key: str | None = None) -> Any:
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

    @staticmethod
    def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("processing cancelled")

    def process(
        self,
        prompt: str,
        tenant_id: str = "",
        cancel_event: threading.Event | None = None,
    ) -> ReviewResult:
        """Process a prompt end-to-end and return the verified output."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        self.logger.debug("Processing prompt (%d chars)", len(prompt))
        self._raise_if_cancelled(cancel_event)
        candidates = self.generator.generate_candidates(prompt, n=self.max_candidates)
        self._raise_if_cancelled(cancel_event)

        best, rejected, n = self._score_candidates(
            candidates,
            prompt,
            tenant_id,
            cancel_event=cancel_event,
        )
        self._raise_if_cancelled(cancel_event)

        if best[0] is not None:
            result = self._emit_approved(best, n)
        else:
            result = self._handle_rejection(prompt, tenant_id, rejected, n)

        result = self._apply_containment_guard(result, prompt)
        self._report_recall_correctness(prompt, result)
        return result

    def _report_recall_correctness(self, prompt: str, result: ReviewResult) -> None:
        """Post the verification verdict to REMANENTIA as the recall's label.

        No-op unless a correctness-feedback client is configured. The verdict on
        the emitted answer (clean → correct; halted or verified-downgraded →
        incorrect) becomes ``was_correct`` for the recall that ``prompt``
        grounded, closing REMANENTIA's two-label loop. ``try_record`` swallows
        every transport and protocol failure — memory feedback must never break
        answering — and a query with no prior REMANENTIA recall is a no-op (404).
        """
        client = self.correctness_feedback
        if client is None:
            return
        try:
            client.try_record(recall_outcome(prompt, result))
        except Exception:  # noqa: BLE001 — feedback must never break answering
            self.logger.warning("recall-correctness feedback failed", exc_info=True)

    def _apply_containment_guard(
        self, result: ReviewResult, prompt: str
    ) -> ReviewResult:
        """Scan the output against the session's reality anchor if a guard is set.

        A ``"block"`` verdict
        converts the result into a halted ReviewResult whose
        ``halt_evidence`` carries the guard's findings for audit.
        """
        guard = self.containment_guard
        anchor = self.containment_anchor
        if guard is None or anchor is None:
            return result
        verdict = guard.check({"text": result.output, "prompt": prompt}, anchor)
        if verdict.decision != "block":
            if verdict.safety_event is not None:
                result.safety_events = (*result.safety_events, verdict.safety_event)
            return result

        reasons = "; ".join(f"{f.category}:{f.severity}" for f in verdict.findings)
        if verdict.anchor_reason:
            reasons = verdict.anchor_reason + (f"; {reasons}" if reasons else "")
        safety_events = result.safety_events + (
            (verdict.safety_event,) if verdict.safety_event is not None else ()
        )
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
            safety_events=safety_events,
        )

    def verify_physical_action(
        self,
        action: PhysicalAction,
        *,
        tenant_id: str = "",
    ) -> GroundingVerdict:
        """Screen a proposed physical action against the configured grounding hook.

        Raises :class:`RuntimeError` if no hook is
        configured — callers opt in explicitly.
        """
        if self.grounding_hook is None:
            raise RuntimeError("grounding_hook not configured on this CoherenceAgent")
        verdict = self.grounding_hook.evaluate(action, tenant_id=tenant_id)
        if _physical_budget_exhausted(verdict):
            return verdict
        if verdict.allowed or self.physical_action_mode == "block":
            return verdict
        return self._warn_only_grounding_verdict(verdict)

    @staticmethod
    def _warn_only_grounding_verdict(
        verdict: GroundingVerdict,
    ) -> GroundingVerdict:
        """Convert a physical block verdict into an advisory verdict."""
        from .cyber_physical import GroundingVerdict
        from .safety_event import SafetyEvent

        event = verdict.safety_event
        evidence_refs = (
            event.evidence_refs
            if event is not None
            else tuple(f"physical:{v.constraint}" for v in verdict.violations)
        )
        attributes = dict(event.attributes) if event is not None else {}
        attributes["enforcement"] = "warn_only"
        attributes["source_policy_decision"] = (
            event.policy_decision if event is not None else "block"
        )
        return GroundingVerdict(
            action=verdict.action,
            allowed=True,
            violations=verdict.violations,
            safety_event=SafetyEvent.from_policy_decision(
                hook_id="cyber_physical.grounding",
                hook_scope="cyber_physical",
                policy_decision="warn",
                halt_reason="physical_constraint_warning",
                observed_score=(event.observed_score if event is not None else 0.0),
                tenant_safe_explanation=(
                    "Physical action has constraint warnings; blocking requires "
                    "the explicit physical action blocking flag."
                ),
                evidence_refs=evidence_refs,
                attributes=attributes,
            ),
        )

    def verify_passport(self, passport: CrossOrgPassport) -> PassportVerdict:
        """Run the configured passport verifier against *passport*.

        Raises :class:`RuntimeError` when no verifier is attached.
        """
        if self.passport_verifier is None:
            raise RuntimeError(
                "passport_verifier not configured on this CoherenceAgent"
            )
        return self.passport_verifier.verify(passport)

    def _score_candidates(
        self,
        candidates: list[dict[str, Any]],
        prompt: str,
        tenant_id: str,
        cancel_event: threading.Event | None = None,
    ) -> tuple[_ScoredCandidate, _ScoredCandidate, int]:
        """Score all candidates, return (best_approved, best_rejected, count)."""
        best: _ScoredCandidate = (None, None, -1.0)  # (text, score, coherence)
        rejected: _ScoredCandidate = (None, None, -1.0)

        for i, cand in enumerate(candidates):
            self._raise_if_cancelled(cancel_event)
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

    def _emit_approved(self, best: _ScoredCandidate, n_candidates: int) -> ReviewResult:
        """Build ReviewResult for the best approved candidate."""
        text, score, coherence = best
        # An approved candidate always carries its generated text.
        assert text is not None

        def coherence_monitor(_token: str) -> float:
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
        self,
        prompt: str,
        tenant_id: str,
        rejected: _ScoredCandidate,
        n_candidates: int,
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
        self,
        prompt: str,
        tenant_id: str,
        rej_score: Any,
        n_candidates: int,
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

    async def aprocess(
        self,
        prompt: str,
        tenant_id: str = "",
        cancel_event: threading.Event | None = None,
    ) -> ReviewResult:
        """Async version of :meth:`process` via ``run_in_executor``."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.process,
            prompt,
            tenant_id,
            cancel_event,
        )

    async def stream(
        self,
        prompt: str,
        tenant_id: str = "",
    ) -> AsyncIterator[tuple[str, float]]:
        """Stream tokens with real-time halt oversight.

        Yields ``(token, coherence)`` tuples. Halting stops future tokens but
        does not retract delivered ones.

        With a configured :class:`ContradictionHalt`
        (``streaming_contradiction_halt=True``) each completed claim is scored
        by ``P(contradiction)`` against retrieved grounding facts and the
        stream halts on genuine contradiction — the calibrated real-time
        mechanism (false-halt 1.48% over 135 grounded passages at threshold
        0.2; see ``benchmarks/results/streaming_contradiction_halt_base.json``
        and the streaming guide).

        Without it, halting falls back to the coherence signal
        (``StreamingKernel`` sliding window, trend detection, hard/soft halt),
        which remains experimental: coherence scores of correct and
        hallucinated partial text overlap, so that path cannot separate them
        without a high false-halt rate (see
        ``benchmarks/streaming_false_halt_bench.py``). Coherence is re-scored
        only at claim boundaries via :class:`StreamingCoherenceGate` to avoid
        scoring half-finished sentences, but this does not resolve the signal
        overlap — enable the contradiction halt or use the response-level
        scorer for production gating.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        if not hasattr(self.generator, "stream_tokens"):
            result = self.process(prompt, tenant_id=tenant_id)
            for word in result.output.split():
                yield word, result.coherence.score if result.coherence else 0.0
            return

        contradiction_halt = getattr(self, "contradiction_halt", None)
        if contradiction_halt is not None:
            # Working real-time halt: at each completed claim, halt when the
            # claim contradicts retrieved grounding. Coherence ≈ 1 − P(contra)
            # is yielded for observability; it is held between claim boundaries.
            claim_tokens: list[str] = []
            last_contra = 0.0
            async for token in self.generator.stream_tokens(prompt):
                claim_tokens.append(token)
                if ends_claim(token):
                    decision = contradiction_halt.should_halt(
                        " ".join(claim_tokens).strip(),
                    )
                    last_contra = decision.contradiction
                    claim_tokens = []
                    yield token, 1.0 - last_contra
                    if decision.halt:
                        return
                else:
                    yield token, 1.0 - last_contra
            return

        self.streaming_kernel.reset_state()
        accumulated: list[str] = []

        def _score_text(text: str) -> float:
            try:
                _, score = self.scorer.review(prompt, text, tenant_id=tenant_id)
            except TypeError:
                _, score = self.scorer.review(prompt, text)
            return float(score.score)

        # Gate re-scoring to complete claims: NLI/RAG coherence on a half-finished
        # sentence dips below the hard limit and false-halts correct text, so the
        # model judges a finished claim and the score holds between boundaries.
        gate = StreamingCoherenceGate(_score_text)

        def _coherence_cb(token: str) -> float:
            accumulated.append(token)
            return gate.update(" ".join(accumulated))

        async for token in self.generator.stream_tokens(prompt):  # pragma: no branch
            score = _coherence_cb(token)
            yield token, score
            if self.streaming_kernel.check_halt(score):
                return
