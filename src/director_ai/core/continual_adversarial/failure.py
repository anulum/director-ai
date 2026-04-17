# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — FailureEvent + FailureStore

"""Thread-safe bounded log of production failure events.

A :class:`FailureEvent` records a prompt that bypassed the
guardrail and the label the guardrail should have emitted. The
store evicts oldest events FIFO when its capacity is reached
and supports windowed queries by count or by elapsed seconds.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Literal

FailureLabel = Literal["unsafe", "policy_violation", "hallucination"]

_VALID_LABELS: frozenset[FailureLabel] = frozenset(
    ("unsafe", "policy_violation", "hallucination")
)


@dataclass(frozen=True)
class FailureEvent:
    """One production failure.

    ``tenant_id`` is empty for single-tenant deployments.
    ``metadata`` carries free-form tags — detector name,
    severity, route identifier — that the miner ignores but
    operators use for cross-referencing."""

    prompt: str
    label: FailureLabel
    tenant_id: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("prompt must be non-empty")
        if self.label not in _VALID_LABELS:
            raise ValueError(
                f"label must be one of {sorted(_VALID_LABELS)}; got {self.label!r}"
            )
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")


class FailureStore:
    """FIFO-bounded thread-safe event log.

    Parameters
    ----------
    capacity :
        Maximum retained events. Default 10 000.
    clock :
        Timestamp source; injection point for tests.
    """

    def __init__(
        self,
        *,
        capacity: int = 10_000,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._clock = clock or time.time
        self._lock = threading.Lock()
        self._events: deque[FailureEvent] = deque(maxlen=capacity)

    def append(self, event: FailureEvent) -> None:
        with self._lock:
            self._events.append(event)

    def record(
        self,
        *,
        prompt: str,
        label: FailureLabel,
        tenant_id: str = "",
        metadata: dict[str, str] | None = None,
    ) -> FailureEvent:
        event = FailureEvent(
            prompt=prompt,
            label=label,
            tenant_id=tenant_id,
            metadata=dict(metadata or {}),
            timestamp=float(self._clock()),
        )
        self.append(event)
        return event

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def snapshot(self) -> tuple[FailureEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def window(
        self,
        *,
        last_n: int | None = None,
        since_seconds: float | None = None,
    ) -> tuple[FailureEvent, ...]:
        if (last_n is None) == (since_seconds is None):
            raise ValueError(
                "exactly one of last_n / since_seconds must be supplied"
            )
        with self._lock:
            snapshot = tuple(self._events)
        if last_n is not None:
            if last_n <= 0:
                raise ValueError("last_n must be positive")
            return snapshot[-last_n:]
        if since_seconds is not None:
            if since_seconds <= 0:
                raise ValueError("since_seconds must be positive")
            cutoff = float(self._clock()) - since_seconds
            return tuple(e for e in snapshot if e.timestamp >= cutoff)
        return snapshot  # pragma: no cover — defensive

    def iter_labelled(self, label: FailureLabel) -> Iterator[FailureEvent]:
        if label not in _VALID_LABELS:
            raise ValueError(f"unknown label {label!r}")
        with self._lock:
            snapshot = tuple(self._events)
        for event in snapshot:
            if event.label == label:
                yield event
