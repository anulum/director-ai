# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Multi-GPU Sharded NLI Scorer

"""Distribute NLI inference across multiple CUDA devices via round-robin.

Usage::

    scorer = ShardedNLIScorer(devices=["cuda:0", "cuda:1"])
    score = scorer.score("premise", "hypothesis")
"""

from __future__ import annotations

import itertools
import threading

from .nli import NLIScorer

__all__ = ["ShardedNLIScorer"]


class ShardedNLIScorer:
    """Wraps N NLIScorer instances on different CUDA devices.

    Parameters
    ----------
    devices : list[str] — e.g. ["cuda:0", "cuda:1"].
    **kwargs — forwarded to each NLIScorer (model_name, backend, etc.).

    """

    def __init__(self, devices: list[str], **kwargs) -> None:
        if not devices:
            raise ValueError("devices list must be non-empty")
        self._scorers = [NLIScorer(device=dev, **kwargs) for dev in devices]
        self._cycle = itertools.cycle(range(len(self._scorers)))
        self._lock = threading.Lock()

    def _next_scorer(self) -> NLIScorer:
        with self._lock:
            idx = next(self._cycle)
        return self._scorers[idx]

    def score(self, premise: str, hypothesis: str) -> float:
        return self._next_scorer().score(premise, hypothesis)

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        n = len(self._scorers)
        if len(pairs) <= n:
            return self._next_scorer().score_batch(pairs)
        # Partition across all shards and dispatch concurrently
        chunks: list[list[tuple[str, str]]] = [[] for _ in range(n)]
        indices: list[list[int]] = [[] for _ in range(n)]
        for i, pair in enumerate(pairs):
            shard = i % n
            chunks[shard].append(pair)
            indices[shard].append(i)

        import concurrent.futures

        results = [0.0] * len(pairs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            futures = {
                pool.submit(self._scorers[s].score_batch, chunks[s]): s
                for s in range(n)
                if chunks[s]
            }
            for fut in concurrent.futures.as_completed(futures):
                s = futures[fut]
                for idx, score in zip(indices[s], fut.result(), strict=True):
                    results[idx] = score
        return results

    def score_chunked(
        self,
        premise: str,
        hypothesis: str,
        outer_agg: str = "max",
        inner_agg: str = "max",
    ) -> tuple[float, list[float]]:
        return self._next_scorer().score_chunked(
            premise,
            hypothesis,
            outer_agg=outer_agg,
            inner_agg=inner_agg,
        )

    @property
    def model_available(self) -> bool:
        return any(s.model_available for s in self._scorers)

    @property
    def device_count(self) -> int:
        return len(self._scorers)
