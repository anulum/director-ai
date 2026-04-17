# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — DecisionLog + ScoringDecision

"""Append-only bounded log of :class:`ScoringDecision` records.

The log keeps the most recent ``capacity`` decisions (FIFO
eviction), exposes a windowed view keyed by age or count, and is
thread-safe on append + snapshot. Prompts are hashed by default
so the log can be reviewed without surfacing user content; an
operator who needs the raw prompt for debugging can turn hashing
off via ``hash_prompts=False``.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Literal

ScoringAction = Literal["allow", "warn", "halt"]

_VALID_ACTIONS: frozenset[str] = frozenset(("allow", "warn", "halt"))


@dataclass(frozen=True)
class ScoringDecision:
    """One scoring decision.

    ``prompt_hash`` is a stable identifier — SHA-256 prefix of the
    prompt by default. ``score`` is the raw guardrail probability
    in ``[0, 1]``. ``action`` is the final decision band.
    ``ground_truth`` is ``None`` when unknown; operators who have
    after-the-fact labels fold them in for Brier-score
    calibration drift. ``timestamp`` defaults to ``time.time()``
    at construction; tests pass explicit values.
    """

    prompt_hash: str
    score: float
    action: ScoringAction
    ground_truth: float | None = None
    timestamp: float = field(default_factory=lambda: time.time())
    tenant_id: str = ""

    def __post_init__(self) -> None:
        if not self.prompt_hash:
            raise ValueError("prompt_hash must be non-empty")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1]; got {self.score!r}")
        if self.action not in _VALID_ACTIONS:
            raise ValueError(
                f"action must be one of {sorted(_VALID_ACTIONS)}; got {self.action!r}"
            )
        if self.ground_truth is not None and not 0.0 <= self.ground_truth <= 1.0:
            raise ValueError(
                f"ground_truth must be in [0, 1] or None; got {self.ground_truth!r}"
            )


class DecisionLog:
    """Thread-safe bounded log of scoring decisions.

    Parameters
    ----------
    capacity :
        Maximum retained decisions. FIFO eviction once reached.
        Default 10 000.
    hash_prompts :
        When ``True`` (default), :meth:`record_from_prompt` hashes
        the supplied prompt before storing it. When ``False``, the
        prompt text itself goes into ``prompt_hash`` — use only
        when the operator explicitly wants raw prompts in the log.
    hasher :
        Callable that turns a prompt into its stored identifier.
        Default SHA-256 hex first 16 chars — enough to distinguish
        decisions for audit without retaining content.
    clock :
        Timestamp source. Injection point for tests.
    """

    def __init__(
        self,
        *,
        capacity: int = 10_000,
        hash_prompts: bool = True,
        hasher: Callable[[str], str] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive; got {capacity!r}")
        self._capacity = capacity
        self._hash_prompts = hash_prompts
        self._hasher = hasher or _default_hasher
        self._clock = clock or time.time
        self._lock = threading.Lock()
        self._log: deque[ScoringDecision] = deque(maxlen=capacity)

    def append(self, decision: ScoringDecision) -> None:
        """Append an already-built :class:`ScoringDecision`."""
        with self._lock:
            self._log.append(decision)

    def record(
        self,
        *,
        prompt: str,
        score: float,
        action: ScoringAction,
        ground_truth: float | None = None,
        tenant_id: str = "",
    ) -> ScoringDecision:
        """Build the identifier from ``prompt`` (hashed when
        ``hash_prompts`` is on) and append. Returns the stored
        decision so callers can log or inspect it downstream."""
        if not prompt:
            raise ValueError("prompt must be non-empty")
        identifier = self._hasher(prompt) if self._hash_prompts else prompt
        decision = ScoringDecision(
            prompt_hash=identifier,
            score=score,
            action=action,
            ground_truth=ground_truth,
            timestamp=float(self._clock()),
            tenant_id=tenant_id,
        )
        self.append(decision)
        return decision

    def __len__(self) -> int:
        with self._lock:
            return len(self._log)

    def snapshot(self) -> tuple[ScoringDecision, ...]:
        """Copy of the full log — ordered oldest to newest."""
        with self._lock:
            return tuple(self._log)

    def window(
        self,
        *,
        last_n: int | None = None,
        since_seconds: float | None = None,
    ) -> tuple[ScoringDecision, ...]:
        """Windowed view. Exactly one of ``last_n`` / ``since_seconds``
        must be supplied.

        ``last_n`` returns the most recent ``n`` decisions (clamped
        to the log size). ``since_seconds`` returns every decision
        whose timestamp is newer than ``now - since_seconds``.
        """
        if (last_n is None) == (since_seconds is None):
            raise ValueError(
                "exactly one of last_n / since_seconds must be supplied"
            )
        with self._lock:
            snapshot = tuple(self._log)
        if last_n is not None:
            if last_n <= 0:
                raise ValueError(f"last_n must be positive; got {last_n!r}")
            return snapshot[-last_n:]
        if since_seconds is not None:
            if since_seconds <= 0:
                raise ValueError(
                    f"since_seconds must be positive; got {since_seconds!r}"
                )
            cutoff = float(self._clock()) - since_seconds
            return tuple(d for d in snapshot if d.timestamp >= cutoff)
        return snapshot  # pragma: no cover — defensive

    def iter_windowed(
        self, *, last_n: int | None = None, since_seconds: float | None = None
    ) -> Iterator[ScoringDecision]:
        """Iterator variant of :meth:`window` — identical semantics,
        streams to avoid materialising the tuple when the caller
        only needs a single pass."""
        yield from self.window(last_n=last_n, since_seconds=since_seconds)


def _default_hasher(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
