# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Score Cache

"""LRU score cache keyed on (query_hash, prefix_hash) to avoid redundant
NLI and embedding computations during streaming.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

__all__ = ["ScoreCache"]


@dataclass(frozen=True)
class _CacheEntry:
    score: float
    h_logical: float
    h_factual: float
    created_at: float
    generation: int = 0


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
        self._generation = 0

    @staticmethod
    def _key(
        query: str,
        prefix: str,
        tenant_id: str = "",
        scope: str = "",
    ) -> str:
        h = hashlib.blake2b(digest_size=16)
        h.update(query.encode("utf-8", errors="replace"))
        h.update(b"\x00")
        h.update(prefix.encode("utf-8", errors="replace"))
        h.update(b"\x00")
        h.update(tenant_id.encode("utf-8", errors="replace"))
        h.update(b"\x00")
        h.update(scope.encode("utf-8", errors="replace"))
        return h.hexdigest()

    def get(
        self,
        query: str,
        prefix: str,
        tenant_id: str = "",
        scope: str = "",
    ) -> _CacheEntry | None:
        k = self._key(query, prefix, tenant_id, scope)
        with self._lock:
            entry = self._store.get(k)
            if entry is None:
                self.misses += 1
                return None
            if time.monotonic() - entry.created_at > self._ttl:
                self._store.pop(k, None)
                self.misses += 1
                return None
            if entry.generation != self._generation:
                self._store.pop(k, None)
                self.misses += 1
                return None
            self._store.move_to_end(k)
            self.hits += 1
            return entry

    def put(
        self,
        query: str,
        prefix: str,
        score: float,
        h_logical: float,
        h_factual: float,
        tenant_id: str = "",
        scope: str = "",
    ) -> None:
        k = self._key(query, prefix, tenant_id, scope)
        with self._lock:
            entry = _CacheEntry(
                score=score,
                h_logical=h_logical,
                h_factual=h_factual,
                created_at=time.monotonic(),
                generation=self._generation,
            )
            self._store[k] = entry
            self._store.move_to_end(k)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def invalidate(self) -> None:
        """Bump generation counter, lazily expiring all current entries."""
        with self._lock:
            self._generation += 1

    @property
    def generation(self) -> int:
        return self._generation

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0
