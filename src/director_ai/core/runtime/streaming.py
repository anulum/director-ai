# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Streaming Token-by-Token Kernel Oversight

"""Streaming oversight for token-by-token coherence monitoring.

Provides ``StreamingKernel`` which extends ``HaltMonitor`` with:
- Async token processing via generator protocol
- Sliding window coherence checks
- Real-time halt with partial output recovery
- Token-level divergence tracking
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..causal_verifier import CounterfactualVerifier
from ..observability.callbacks import (
    TokenTraceCallback,
    TokenTraceEmitter,
    TokenTraceEvent,
)
from ..observability.tracing import trace_token
from ..otel import trace_streaming
from ..safety_event import SafetyEvent
from ..types import EvidenceChunk, HaltEvidence, HaltTraceAttribution
from .kernel import HaltMonitor
from .structured_recovery import (
    StructuredRecoveryConfig,
    StructuredRecoveryResult,
    StructuredRecoveryState,
)

try:
    from backfire_kernel import (
        rust_mean as _rust_mean,
    )
    from backfire_kernel import (
        rust_trend_drop as _rust_trend_drop,
    )

    _RUST_TREND = True
except ImportError:
    _RUST_TREND = False

    def _rust_mean(_values: list[float]) -> float:
        raise RuntimeError("backfire_kernel rust_mean is unavailable")

__all__ = ["StreamSession", "StreamingKernel", "TokenEvent"]

if TYPE_CHECKING:  # pragma: no cover
    from ..scoring.scorer import CoherenceScorer

logger = logging.getLogger("DirectorAI.Streaming")


def _mean(values: Sequence[float] | deque[float]) -> float:
    """Return arithmetic mean with optional Rust kernel acceleration."""
    if not values:
        return 0.0
    if _RUST_TREND:
        try:
            return float(_rust_mean(list(values)))
        except Exception:
            pass
    return sum(values) / len(values)


def _trend_drop(values: list[float] | deque) -> float:
    """Linear regression slope drop over a window of coherence scores.

    Returns the projected drop magnitude: -slope * (n - 1).
    Positive values indicate downward trend.
    """
    if _RUST_TREND:
        try:
            return float(_rust_trend_drop(list(values)))
        except Exception:
            pass
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = _mean(values)
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    slope = num / den if den > 1e-12 else 0.0
    return -slope * (n - 1)


@dataclass
class TokenEvent:
    """A single token event in the stream."""

    token: str
    index: int
    coherence: float
    timestamp: float
    halted: bool = False
    warning: bool = False
    evidence: str | None = None
    halt_evidence: HaltEvidence | None = None
    safety_event: SafetyEvent | None = None
    debug_info: dict | None = None


@dataclass
class StreamSession:
    """Tracks state of a streaming oversight session."""

    tokens: list[str] = field(default_factory=list)
    events: list[TokenEvent] = field(default_factory=list)
    coherence_history: list[float] = field(default_factory=list)
    halted: bool = False
    soft_halted: bool = False
    halt_index: int = -1
    halt_reason: str = ""
    halt_evidence: str | None = None
    halt_evidence_structured: HaltEvidence | None = None
    structured_recovery: StructuredRecoveryResult | None = None
    safety_events: list[SafetyEvent] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    warning_count: int = 0
    debug_log: list[dict] = field(default_factory=list)

    @property
    def output(self) -> str:
        """Return generated output with hard-halted tokens removed."""
        if self.soft_halted:
            return "".join(self.tokens)
        if self.halted and self.halt_index >= 0:
            return "".join(self.tokens[: self.halt_index])
        return "".join(self.tokens)

    @property
    def token_count(self) -> int:
        """Return the number of observed stream tokens."""
        return len(self.tokens)

    @property
    def avg_coherence(self) -> float:
        """Return the average coherence score across observed tokens."""
        return _mean(self.coherence_history)

    @property
    def min_coherence(self) -> float:
        """Return the minimum coherence score across observed tokens."""
        if not self.coherence_history:
            return 0.0
        return min(self.coherence_history)

    @property
    def duration_ms(self) -> float:
        """Return session duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000


class StreamingKernel(HaltMonitor):
    """Streaming token-by-token safety kernel with sliding window oversight.

    Extends ``HaltMonitor`` with token-level monitoring and a sliding
    window coherence check that can catch gradual degradation.

    Parameters
    ----------
    hard_limit : float — absolute coherence floor (halt if below).
    window_size : int — number of tokens in sliding coherence window.
    window_threshold : float — halt if sliding window average drops below this.
    trend_window : int — tokens to check for downward trend.
    trend_threshold : float — halt if coherence drops this much over trend window.

    """

    _SENTENCE_ENDS = frozenset(".!?")
    _SOFT_HALT_CAP = 50

    def __init__(
        self,
        hard_limit: float = 0.5,
        window_size: int = 10,
        window_threshold: float = 0.55,
        trend_window: int = 5,
        trend_threshold: float = 0.15,
        on_halt=None,
        soft_limit: float = 0.6,
        streaming_debug: bool = False,
        halt_mode: str = "hard",
        score_every_n: int = 1,
        adaptive: bool = False,
        max_cadence: int = 8,
    ) -> None:
        super().__init__(hard_limit=hard_limit, on_halt=on_halt)
        if halt_mode not in ("hard", "soft"):
            raise ValueError(f"halt_mode must be 'hard' or 'soft', got {halt_mode!r}")
        if score_every_n < 1:
            raise ValueError(f"score_every_n must be >= 1, got {score_every_n}")
        if max_cadence < 1:
            raise ValueError(f"max_cadence must be >= 1, got {max_cadence}")
        self.window_size = window_size
        self.window_threshold = window_threshold
        self.trend_window = trend_window
        self.trend_threshold = trend_threshold
        self.soft_limit = soft_limit
        self.streaming_debug = streaming_debug
        self.halt_mode = halt_mode
        self.score_every_n = score_every_n
        self.adaptive = adaptive
        self.max_cadence = max_cadence
        self._window: deque[float] = deque(maxlen=window_size)
        self._history: list[float] = []

    def check_halt(self, score: float) -> bool:
        """Evaluate halt conditions for a single score update.

        Maintains internal sliding window and trend history.
        Returns True if any halt condition is met.
        """
        self._window.append(score)
        self._history.append(score)
        if score < self.hard_limit:
            return True
        if len(self._window) >= self.window_size:
            avg = _mean(self._window)
            if avg < self.window_threshold:
                return True
        if len(self._history) >= self.trend_window:
            recent = list(self._history)[-self.trend_window :]
            if _trend_drop(recent) > self.trend_threshold:
                return True
        return False

    def reset_state(self) -> None:
        """Clear internal window/trend state and re-arm kernel for a new stream."""
        self._window.clear()
        self._history.clear()
        self.reactivate()

    @staticmethod
    def _suggested_action(reason: str) -> str:
        """Return tenant-safe remediation guidance for a halt reason."""
        if "hard_limit" in reason:
            return "Reduce generation temperature or add KB facts."
        if "window_avg" in reason:
            return "Context may be drifting from grounded facts."
        if "downward_trend" in reason:
            return "Response quality degrading; rephrase the prompt."
        return "Review the generated output for factual accuracy."

    @staticmethod
    def _scorer_path(scorer: object) -> str:
        """Return a stable dotted path for scorer review attribution."""
        scorer_type = type(scorer)
        return f"{scorer_type.__module__}.{scorer_type.__qualname__}.review"

    @staticmethod
    def _fact_source(chunks: Sequence[EvidenceChunk]) -> str:
        """Return a comma-separated list of unique evidence sources."""
        sources: list[str] = []
        for chunk in chunks:
            source_value = getattr(chunk, "source", "")
            if not isinstance(source_value, str):
                continue
            source = source_value.strip()
            if source and source not in sources:
                sources.append(source)
        return ",".join(sources)

    @staticmethod
    def _retrieval_path(chunks: Sequence[EvidenceChunk]) -> str:
        """Return the retrieval path used for halt trace attribution."""
        if chunks:
            return "scorer.review.evidence.chunks"
        return "scorer.review.no_chunks"

    def _trace_metrics(
        self,
        reason: str,
        event: TokenEvent,
        history: list[float],
        window: deque[float],
    ) -> tuple[float | None, float]:
        """Return threshold and margin values for a halt reason."""
        if "hard_limit" in reason:
            return self.hard_limit, max(0.0, self.hard_limit - event.coherence)
        if "window_avg" in reason:
            avg = _mean(window) if window else event.coherence
            return self.window_threshold, max(0.0, self.window_threshold - avg)
        if "downward_trend" in reason:
            recent = history[-self.trend_window :]
            drop = _trend_drop(recent)
            return self.trend_threshold, max(0.0, drop - self.trend_threshold)
        return None, 0.0

    @staticmethod
    def _set_halt_otel_attributes(span: object, evidence: HaltEvidence | None) -> None:
        """Attach halt and counterfactual attributes to an OTEL span."""
        setter = getattr(span, "set_attribute", None)
        if setter is None or evidence is None:
            return
        trace = evidence.trace_attribution
        if trace is not None:
            setter("stream.halt_cause.fact_source", trace.fact_source)
            setter("stream.halt_cause.retrieval_path", trace.retrieval_path)
            setter("stream.halt_cause.scorer_path", trace.scorer_path)
            setter("stream.halt_cause.token_offset", trace.token_offset)
            if trace.threshold is not None:
                setter("stream.halt_cause.threshold", trace.threshold)
            setter("stream.halt_cause.margin", trace.causal_contribution)
        diagnostic = evidence.counterfactual_diagnostic
        setter("stream.counterfactual.available", diagnostic is not None)
        if diagnostic is None:
            return
        setter("stream.counterfactual.observed_score", diagnostic.observed_score)
        setter("stream.counterfactual.threshold", diagnostic.threshold)
        setter("stream.counterfactual.candidate_count", len(diagnostic.candidates))
        best = diagnostic.best_change
        if best is not None:
            setter("stream.counterfactual.best_fact_source", best.fact_source)
            setter(
                "stream.counterfactual.required_score_delta",
                best.required_score_delta,
            )
            setter("stream.counterfactual.prevented_halt", best.prevented_halt)

    def stream_tokens(
        self,
        token_generator,
        coherence_callback,
        evidence_callback=None,
        scorer: CoherenceScorer | None = None,
        top_k: int = 3,
        prompt: str = "",
        trace_callbacks: list[TokenTraceCallback] | None = None,
        tenant_id: str = "",
        request_id: str = "",
        structured_recovery: StructuredRecoveryConfig | None = None,
    ) -> StreamSession:
        """Process tokens one by one with sliding window oversight.

        Parameters
        ----------
        token_generator : iterable of str — token source.
        coherence_callback : callable(str) -> float — receives the
            accumulated output so far (not the individual token) and
            returns a coherence score in [0, 1]. Called every
            ``score_every_n`` tokens; cadence adapts when ``adaptive=True``.
        evidence_callback : callable(str) -> str | None — optional, returns
            human-readable evidence snippet explaining the coherence score.
            Called only on halt events to avoid overhead on every token.
        scorer : CoherenceScorer | None — when provided, halt events
            include structured HaltEvidence with top-K contradicting chunks.
        top_k : int — number of evidence chunks to include (default 3).
        prompt : str — original user prompt, passed to scorer.review() for
            KB/RAG context in halt evidence.

        Returns
        -------
        StreamSession with full oversight trace.

        """
        session = StreamSession(start_time=time.monotonic())
        window: deque[float] = deque(maxlen=self.window_size)
        _soft_halt_pending = False
        _soft_halt_reason = ""
        _soft_halt_extra_tokens = 0
        recovery_state = (
            StructuredRecoveryState(structured_recovery)
            if structured_recovery is not None
            else None
        )

        emitter = TokenTraceEmitter(trace_callbacks)

        def _emit_trace(event: TokenEvent, halt_reason: str = "") -> None:
            """Fan out an OTEL child span and every registered callback."""
            with trace_token(
                event.index,
                token=event.token,
                tenant_id=tenant_id,
                request_id=request_id,
            ) as span:
                setter = getattr(span, "set_attribute", None)
                if setter is not None:
                    setter("token.coherence", float(event.coherence))
                    setter("token.halted", bool(event.halted))
                    if halt_reason:
                        setter("token.halt_reason", halt_reason)
                    if event.halt_evidence is not None:
                        self._set_halt_otel_attributes(span, event.halt_evidence)
                    if event.safety_event is not None:
                        setter("safety.event_id", event.safety_event.event_id)
                        setter("safety.hook_id", event.safety_event.hook_id)
                        setter(
                            "safety.policy_decision",
                            event.safety_event.policy_decision,
                        )
                    if event.warning:
                        setter("token.warning", True)
            if emitter.enabled:
                emitter.emit(
                    TokenTraceEvent(
                        index=event.index,
                        token=event.token,
                        coherence=event.coherence,
                        timestamp=event.timestamp,
                        halted=event.halted,
                        halt_reason=halt_reason,
                        tenant_id=tenant_id,
                        request_id=request_id,
                    )
                )

        def _finalize_halt(event: TokenEvent, reason: str) -> None:
            """Populate session, safety, and structured evidence for a halt."""
            event.halted = True
            session.halted = True
            if session.halt_index < 0:
                session.halt_index = event.index
            session.halt_reason = reason
            if recovery_state is not None:
                session.structured_recovery = recovery_state.finalise(
                    halted_at=session.halt_index,
                )
            if evidence_callback:
                ev = evidence_callback("".join(session.tokens))
                event.evidence = ev
                session.halt_evidence = ev
            if scorer is not None:
                accumulated = "".join(session.tokens)
                prompt_context = prompt.strip() or accumulated
                _, cs = scorer.review(prompt_context, accumulated)
                chunks: list[EvidenceChunk] = []
                nli_scores: list[float] | None = None
                if cs.evidence and cs.evidence.chunks:
                    sorted_chunks = sorted(cs.evidence.chunks, key=lambda c: c.distance)
                    chunks = sorted_chunks[:top_k]
                    if cs.evidence.chunk_scores:  # pragma: no branch
                        nli_scores = cs.evidence.chunk_scores[:top_k]
                threshold, contribution = self._trace_metrics(
                    reason,
                    event,
                    session.coherence_history,
                    window,
                )
                counterfactual = (
                    CounterfactualVerifier.explain_halt_fact_change(
                        observed_score=event.coherence,
                        threshold=threshold,
                        evidence_chunks=chunks,
                        proposed_fact=(
                            cs.evidence.nli_hypothesis
                            if cs.evidence is not None
                            else accumulated
                        ),
                    )
                    if threshold is not None
                    else None
                )
                structured = HaltEvidence(
                    reason=reason,
                    last_score=cs.score,
                    evidence_chunks=chunks,
                    nli_scores=nli_scores,
                    suggested_action=self._suggested_action(reason),
                    trace_attribution=HaltTraceAttribution(
                        fact_source=self._fact_source(chunks),
                        retrieval_path=self._retrieval_path(chunks),
                        scorer_path=self._scorer_path(scorer),
                        token_offset=(
                            session.halt_index
                            if session.halt_index >= 0
                            else event.index
                        ),
                        threshold=threshold,
                        causal_contribution=contribution,
                    ),
                    counterfactual_diagnostic=counterfactual,
                )
                event.halt_evidence = structured
                session.halt_evidence_structured = structured
                event.safety_event = SafetyEvent.from_halt_evidence(
                    structured,
                    hook_id="streaming.kernel",
                    request_id=request_id,
                    tenant_id=tenant_id,
                    attributes={"token_offset": str(event.index)},
                )
            if event.safety_event is None:
                threshold, _ = self._trace_metrics(
                    reason,
                    event,
                    session.coherence_history,
                    window,
                )
                event.safety_event = SafetyEvent.from_policy_decision(
                    hook_id="streaming.kernel",
                    hook_scope="streaming",
                    policy_decision="halt",
                    halt_reason=reason,
                    threshold=threshold,
                    observed_score=event.coherence,
                    request_id=request_id,
                    tenant_id=tenant_id,
                    tenant_safe_explanation=self._suggested_action(reason),
                    attributes={"token_offset": str(event.index)},
                )
            session.safety_events.append(event.safety_event)

        def _is_sentence_boundary(tok: str) -> bool:
            """Return whether a token ends a sentence for soft halts."""
            stripped = tok.rstrip()
            return bool(stripped) and stripped[-1] in self._SENTENCE_ENDS

        cadence = self.score_every_n
        last_score = 0.5

        for i, token in enumerate(token_generator):
            if not self.is_active:
                session.halted = True
                session.halt_index = i
                session.halt_reason = "kernel_inactive"
                break

            if i % cadence == 0:
                accumulated = "".join(session.tokens) + token
                try:
                    score = coherence_callback(accumulated)
                except (TimeoutError, OSError):
                    session.halted = True
                    session.halt_index = i
                    session.halt_reason = "callback_timeout"
                    break
                last_score = score
                if self.adaptive:
                    w_avg = _mean(window) if window else score
                    if w_avg > self.soft_limit and cadence < self.max_cadence:
                        cadence = min(cadence + 1, self.max_cadence)
                    elif score < self.soft_limit:
                        cadence = 1
            else:
                score = last_score
            now = time.monotonic()

            event = TokenEvent(
                token=token,
                index=i,
                coherence=score,
                timestamp=now,
            )

            session.tokens.append(token)
            if recovery_state is not None:
                recovery_state.update("".join(session.tokens))
            session.coherence_history.append(score)
            window.append(score)

            if self.streaming_debug:
                w_avg = _mean(window) if window else 0.0
                recent = session.coherence_history[-self.trend_window :]
                t_drop = _trend_drop(recent)
                snap = {
                    "index": i,
                    "coherence": score,
                    "window_avg": round(w_avg, 6),
                    "trend_drop": round(t_drop, 6),
                    "accumulated_tokens": len(session.tokens),
                }
                event.debug_info = snap
                session.debug_log.append(snap)

            # If soft-halt pending, check for sentence boundary or cap
            if _soft_halt_pending:
                _soft_halt_extra_tokens += 1
                session.events.append(event)
                at_cap = _soft_halt_extra_tokens >= self._SOFT_HALT_CAP
                if _is_sentence_boundary(token) or at_cap:
                    session.soft_halted = True
                    _finalize_halt(event, _soft_halt_reason)
                    _emit_trace(event, _soft_halt_reason)
                    break
                _emit_trace(event)
                continue

            halt_reason = ""
            if score < self.hard_limit:
                halt_reason = f"hard_limit ({score:.4f} < {self.hard_limit})"
            elif len(window) >= self.window_size:
                avg = _mean(window)
                if avg < self.window_threshold:
                    halt_reason = f"window_avg ({avg:.4f} < {self.window_threshold})"
            if not halt_reason and len(session.coherence_history) >= self.trend_window:
                recent = session.coherence_history[-self.trend_window :]
                drop = _trend_drop(recent)
                if drop > self.trend_threshold:
                    halt_reason = (
                        f"downward_trend ({drop:.4f} > {self.trend_threshold})"
                    )

            if halt_reason:
                if self.halt_mode == "soft" and "hard_limit" not in halt_reason:
                    _soft_halt_pending = True
                    _soft_halt_reason = halt_reason
                    session.halt_index = event.index
                    session.events.append(event)
                    if _is_sentence_boundary(token):
                        session.soft_halted = True
                        _finalize_halt(event, halt_reason)
                        _emit_trace(event, halt_reason)
                        break
                    _emit_trace(event, halt_reason)
                    continue
                # Hard halt
                _finalize_halt(event, halt_reason)
                if "hard_limit" in halt_reason:
                    self.emergency_stop()
                session.events.append(event)
                _emit_trace(event, halt_reason)
                break

            if score < self.soft_limit:
                event.warning = True
                session.warning_count += 1

            session.events.append(event)
            _emit_trace(event)

        session.end_time = time.monotonic()
        if session.halted and self.on_halt:
            self.on_halt(session)

        with trace_streaming() as span:
            span.set_attribute("stream.halted", session.halted)
            span.set_attribute("stream.soft_halted", session.soft_halted)
            span.set_attribute("stream.halt_reason", session.halt_reason)
            span.set_attribute("stream.token_count", session.token_count)
            span.set_attribute("stream.warning_count", session.warning_count)
            self._set_halt_otel_attributes(span, session.halt_evidence_structured)
            if session.coherence_history:
                span.set_attribute("stream.avg_coherence", session.avg_coherence)

        if emitter.enabled:
            emitter.end(
                tenant_id=tenant_id,
                request_id=request_id,
                summary={
                    "halted": session.halted,
                    "soft_halted": session.soft_halted,
                    "halt_reason": session.halt_reason,
                    "token_count": session.token_count,
                    "warning_count": session.warning_count,
                    "avg_coherence": (
                        session.avg_coherence if session.coherence_history else 0.0
                    ),
                },
            )
        return session

    def stream_output(self, token_generator, coherence_callback) -> str:
        """Backward-compatible: returns string output or interrupt message."""
        session = self.stream_tokens(token_generator, coherence_callback)
        if session.halted:
            return f"[KERNEL INTERRUPT: {session.halt_reason}]"
        return session.output
