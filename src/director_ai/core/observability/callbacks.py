# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — per-token trace callback protocol

"""Pluggable token-trace callbacks.

The streaming kernel calls
:meth:`TokenTraceCallback.on_token` once per emitted token and
:meth:`on_stream_end` when the session completes. The default
``LangfuseTokenCallback`` adapter posts each token as an
observation to a Langfuse trace; callers that use another vendor
implement the two-method protocol against their own client.

Callbacks should not raise — the emitter catches and logs any
exception so one broken sink cannot corrupt the stream. Callbacks
must also tolerate receiving no ``on_stream_end`` when the host
process crashes mid-stream; design them to be restartable.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("DirectorAI.Observability")

__all__ = [
    "LangfuseTokenCallback",
    "TokenTraceCallback",
    "TokenTraceEmitter",
    "TokenTraceEvent",
]


@dataclass
class TokenTraceEvent:
    """Structured record of one token's state.

    ``token`` is kept out of the default attribute set on OTEL spans
    but is passed to callbacks so a Langfuse-style sink can choose to
    record it (encrypted, hashed, or verbatim depending on policy).
    """

    index: int
    token: str
    coherence: float
    timestamp: float
    halted: bool = False
    halt_reason: str = ""
    tenant_id: str = ""
    request_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def token_hash(self) -> str:
        """SHA-256 truncated fingerprint of the token text — useful for
        callbacks that must log a stable ID without the raw text."""
        return hashlib.sha256(self.token.encode("utf-8")).hexdigest()[:16]


class TokenTraceCallback(ABC):
    """Protocol every token sink implements."""

    @abstractmethod
    def on_token(self, event: TokenTraceEvent) -> None:
        """Receive one token event. Must not raise."""
        ...  # pragma: no cover

    def on_stream_end(
        self, *, tenant_id: str, request_id: str, summary: dict[str, Any]
    ) -> None:
        """Optional — called once when the stream completes.

        The default implementation records a debug log line and
        returns. Making it ``abstractmethod`` would force every
        minimal sink that only cares about per-token events to
        override with an empty stub, and a truly empty body trips
        ruff's ``B027`` ``empty-method-without-abstract-decorator``
        rule. A non-empty but harmless body satisfies both.
        """
        logger.debug(
            "stream end tenant=%s request=%s summary_keys=%s",
            tenant_id,
            request_id,
            sorted(summary),
        )


class TokenTraceEmitter:
    """Fan-out helper that dispatches events to every registered callback.

    Storing one emitter per kernel instance keeps the streaming hot
    path branch-free: the fan-out is a method call with a short
    list, and there is no global registry to lock.
    """

    def __init__(
        self,
        callbacks: list[TokenTraceCallback] | None = None,
    ) -> None:
        self._callbacks: list[TokenTraceCallback] = list(callbacks or [])

    def register(self, callback: TokenTraceCallback) -> None:
        """Add a callback to the fan-out."""
        if not isinstance(callback, TokenTraceCallback):
            raise TypeError(f"{callback!r} must subclass TokenTraceCallback")
        self._callbacks.append(callback)

    def unregister(self, callback: TokenTraceCallback) -> None:
        """Remove a previously registered callback. Missing entries are
        silently ignored so idempotent lifecycle hooks remain safe."""
        with contextlib.suppress(ValueError):
            self._callbacks.remove(callback)

    def __len__(self) -> int:
        return len(self._callbacks)

    @property
    def enabled(self) -> bool:
        return bool(self._callbacks)

    def emit(self, event: TokenTraceEvent) -> None:
        """Dispatch ``event`` to every callback. Exceptions are
        caught and logged — the scoring stream must not be aborted
        by a broken observer."""
        for cb in self._callbacks:
            try:
                cb.on_token(event)
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    "token trace callback %s raised %s", type(cb).__name__, exc
                )

    def end(self, *, tenant_id: str, request_id: str, summary: dict[str, Any]) -> None:
        """Notify every callback that the session is over."""
        for cb in self._callbacks:
            try:
                cb.on_stream_end(
                    tenant_id=tenant_id,
                    request_id=request_id,
                    summary=summary,
                )
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning(
                    "token trace callback %s end raised %s",
                    type(cb).__name__,
                    exc,
                )


class LangfuseTokenCallback(TokenTraceCallback):
    """Adapter that records each token as a Langfuse observation.

    ``client`` is an object that exposes a ``trace()`` method
    returning a handle with ``span()``, ``score()`` and ``update()``
    — the subset of the Langfuse SDK this adapter depends on. The
    shape is replicated in :class:`tests.test_observability.FakeLangfuse`
    for testing without the real dependency.

    The adapter emits:

    * one ``span`` per token — name ``"director.token"``, with
      ``coherence``, ``halted``, ``halt_reason`` as metadata;
    * one ``score`` per token tagged ``"coherence"`` — the numeric
      value, for threshold analytics;
    * one ``update()`` on the parent trace at ``on_stream_end`` —
      final halt reason, token count, and average coherence.

    The adapter keeps a dict of per-``request_id`` trace handles so
    parallel streams do not cross-contaminate their observations.
    """

    _TRACE_NAME = "director.stream"

    def __init__(
        self,
        client: Any,
        *,
        default_tenant: str = "",
        record_token_text: bool = False,
    ) -> None:
        if client is None:
            raise ValueError("client is required")
        self._client = client
        self._default_tenant = default_tenant
        self._record_token_text = record_token_text
        self._traces: dict[str, Any] = {}

    def _trace_for(self, request_id: str, tenant_id: str) -> Any:
        key = request_id or "anonymous"
        trace = self._traces.get(key)
        if trace is None:
            trace = self._client.trace(
                name=self._TRACE_NAME,
                user_id=tenant_id or self._default_tenant,
                metadata={"request_id": request_id or ""},
            )
            self._traces[key] = trace
        return trace

    def on_token(self, event: TokenTraceEvent) -> None:
        trace = self._trace_for(event.request_id, event.tenant_id)
        payload: dict[str, Any] = {
            "index": event.index,
            "coherence": event.coherence,
            "halted": event.halted,
            "halt_reason": event.halt_reason,
        }
        if self._record_token_text:
            payload["token"] = event.token
        else:
            payload["token_hash"] = event.token_hash()
        if event.extra:
            payload["extra"] = event.extra
        trace.span(
            name="director.token",
            metadata=payload,
            start_time=event.timestamp,
        )
        trace.score(
            name="coherence",
            value=float(event.coherence),
            comment=event.halt_reason or None,
        )

    def on_stream_end(
        self, *, tenant_id: str, request_id: str, summary: dict[str, Any]
    ) -> None:
        key = request_id or "anonymous"
        trace = self._traces.pop(key, None)
        if trace is None:
            return
        try:
            trace.update(
                status_message=("halted" if summary.get("halted") else "completed"),
                metadata=summary,
            )
        except AttributeError:  # pragma: no cover — client shape drift
            logger.debug("Langfuse trace object missing update(); skipping end update")
