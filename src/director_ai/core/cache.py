# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Score Cache
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
LRU score cache keyed on (query_hash, prefix_hash) to avoid redundant
NLI and embedding computations during streaming.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class _CacheEntry:
    score: float
    h_logical: float
    h_factual: float
    created_at: float


class ScoreCache:
    """Thread-safe LRU cache for coherence scores.

    Parameters
    ----------
    max_size : int — maximum entries (default 1024).
    ttl_seconds : float — time-to-live per entry (default 300s).
    """

    def __init__(self, max_size: int = 1024, ttl_seconds: float = 300.0) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(query: str, prefix: str) -> str:
        h = hashlib.blake2b(digest_size=16)
        h.update(query.encode("utf-8", errors="replace"))
        h.update(b"\x00")
        h.update(prefix.encode("utf-8", errors="replace"))
        return h.hexdigest()

    def get(self, query: str, prefix: str) -> _CacheEntry | None:
        k = self._key(query, prefix)
        with self._lock:
            entry = self._store.get(k)
            if entry is None:
                self.misses += 1
                return None
            if time.monotonic() - entry.created_at > self._ttl:
                self._store.pop(k, None)
                self.misses += 1
                return None
            self._store.move_to_end(k)
            self.hits += 1
            return entry

    def put(
        self, query: str, prefix: str, score: float, h_logical: float, h_factual: float
    ) -> None:
        k = self._key(query, prefix)
        entry = _CacheEntry(
            score=score,
            h_logical=h_logical,
            h_factual=h_factual,
            created_at=time.monotonic(),
        )
        with self._lock:
            self._store[k] = entry
            self._store.move_to_end(k)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0
