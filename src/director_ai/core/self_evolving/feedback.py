# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — feedback store

"""Append-only store of :class:`FeedbackEvent` records.

Two real backends:

* :class:`InMemoryFeedbackStore` — thread-safe, label-indexed,
  bounded by a caller-configurable capacity (oldest events evicted
  when the capacity is reached). The right choice for CI,
  short-lived worker processes, and deployments where the store
  is rebuilt from another source of truth on restart.
* :class:`JSONLFeedbackStore` — persists every event as one JSON
  line on disk. Reopens in O(n) at construction time. Rotates the
  file once it exceeds ``max_bytes`` — the previous file is kept
  with a ``.1`` suffix for offline audit.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol, runtime_checkable

FeedbackLabel = Literal["safe", "unsafe", "false_positive", "false_negative"]

_VALID_LABELS: frozenset[str] = frozenset(
    ("safe", "unsafe", "false_positive", "false_negative")
)


@dataclass(frozen=True)
class FeedbackEvent:
    """One guardrail decision with operator feedback.

    ``prompt`` is the input the guardrail saw. ``response`` is
    the model output. ``label`` is the operator's verdict.
    ``tenant_id`` and ``metadata`` are optional sidecar fields
    that downstream trainers or calibrators can filter by.
    ``timestamp`` defaults to ``time.time()`` at construction —
    tests pass explicit values for determinism.
    """

    prompt: str
    response: str
    label: FeedbackLabel
    tenant_id: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())

    def __post_init__(self) -> None:
        if self.label not in _VALID_LABELS:
            raise ValueError(
                f"FeedbackEvent.label must be one of {sorted(_VALID_LABELS)}; "
                f"got {self.label!r}"
            )
        if not self.prompt:
            raise ValueError("FeedbackEvent.prompt must be non-empty")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> FeedbackEvent:
        data = json.loads(raw)
        return cls(**data)


@runtime_checkable
class FeedbackStore(Protocol):
    """Append-only event store."""

    def append(self, event: FeedbackEvent) -> None: ...

    def iter_all(self) -> Iterator[FeedbackEvent]: ...

    def iter_labelled(self, label: FeedbackLabel) -> Iterator[FeedbackEvent]: ...

    def __len__(self) -> int: ...


class InMemoryFeedbackStore:
    """Thread-safe in-memory feedback store with label index.

    Parameters
    ----------
    capacity :
        Maximum number of events retained. Oldest events are
        evicted FIFO when the capacity is reached. Default 10 000.
    """

    def __init__(self, *, capacity: int = 10_000) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive; got {capacity!r}")
        self._capacity = capacity
        self._lock = threading.Lock()
        self._events: deque[FeedbackEvent] = deque(maxlen=capacity)
        self._by_label: dict[str, deque[FeedbackEvent]] = {
            lbl: deque(maxlen=capacity) for lbl in _VALID_LABELS
        }

    def append(self, event: FeedbackEvent) -> None:
        with self._lock:
            if len(self._events) == self._capacity:
                dropped = self._events[0]
                # The label-specific deque shares the same FIFO
                # semantics, so evict from there too if the dropped
                # event lived in it.
                bucket = self._by_label[dropped.label]
                try:
                    bucket.remove(dropped)
                except ValueError:  # pragma: no cover — defensive
                    pass
            self._events.append(event)
            self._by_label[event.label].append(event)

    def iter_all(self) -> Iterator[FeedbackEvent]:
        with self._lock:
            snapshot = tuple(self._events)
        yield from snapshot

    def iter_labelled(self, label: FeedbackLabel) -> Iterator[FeedbackEvent]:
        if label not in _VALID_LABELS:
            raise ValueError(f"unknown label {label!r}")
        with self._lock:
            snapshot = tuple(self._by_label[label])
        yield from snapshot

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


class JSONLFeedbackStore:
    """Append-only JSONL feedback store.

    Parameters
    ----------
    path :
        Destination path. Parent directory must exist.
    max_bytes :
        File-size cap. When exceeded, the current file is rotated
        to ``path + ".1"`` (replacing any previous rotation) and a
        fresh file is started. Default 10 MB.
    """

    def __init__(self, path: str, *, max_bytes: int = 10 * 1024 * 1024) -> None:
        if not path:
            raise ValueError("path must be non-empty")
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive; got {max_bytes!r}")
        self._path = path
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        # Seed the cache by reading whatever is already on disk.
        self._cached_count = 0
        if os.path.exists(self._path):
            with open(self._path, encoding="utf-8") as f:
                for _ in f:
                    self._cached_count += 1

    def append(self, event: FeedbackEvent) -> None:
        line = event.to_json() + "\n"
        with self._lock:
            self._maybe_rotate()
            # Atomic append via O_APPEND — the kernel guarantees the
            # write lands at the current end-of-file even under
            # concurrent writers from other processes.
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)
            self._cached_count += 1

    def iter_all(self) -> Iterator[FeedbackEvent]:
        with self._lock:
            if not os.path.exists(self._path):
                return
            with open(self._path, encoding="utf-8") as f:
                lines = tuple(f.readlines())
        for raw in lines:
            raw = raw.strip()
            if raw:
                yield FeedbackEvent.from_json(raw)

    def iter_labelled(self, label: FeedbackLabel) -> Iterator[FeedbackEvent]:
        if label not in _VALID_LABELS:
            raise ValueError(f"unknown label {label!r}")
        for event in self.iter_all():
            if event.label == label:
                yield event

    def __len__(self) -> int:
        with self._lock:
            return self._cached_count

    def _maybe_rotate(self) -> None:
        if not os.path.exists(self._path):
            return
        if os.path.getsize(self._path) < self._max_bytes:
            return
        rotated = self._path + ".1"
        # Use os.replace for atomic rename on POSIX + Windows.
        os.replace(self._path, rotated)
        self._cached_count = 0

    def bulk_append(self, events: Iterable[FeedbackEvent]) -> None:
        """Batched variant that amortises the lock + open() cost.

        Writes via a tmp-file + replace so a crash mid-batch leaves
        the store in its pre-batch state.
        """
        batch = list(events)
        if not batch:
            return
        with self._lock:
            self._maybe_rotate()
            prefix_bytes: bytes
            if os.path.exists(self._path):
                with open(self._path, "rb") as f:
                    prefix_bytes = f.read()
            else:
                prefix_bytes = b""
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=os.path.dirname(os.path.abspath(self._path)) or ".",
                prefix=".jsonl_feedback_",
                delete=False,
            ) as tmp:
                tmp.write(prefix_bytes)
                for event in batch:
                    tmp.write((event.to_json() + "\n").encode("utf-8"))
                tmp_path = tmp.name
            os.replace(tmp_path, self._path)
            self._cached_count += len(batch)
