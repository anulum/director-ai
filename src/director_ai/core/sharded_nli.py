# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Multi-GPU Sharded NLI Scorer
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Distribute NLI inference across multiple CUDA devices via round-robin.

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
        return self._next_scorer().score_batch(pairs)

    def score_chunked(
        self,
        premise: str,
        hypothesis: str,
        outer_agg: str = "max",
    ) -> tuple[float, list[float]]:
        return self._next_scorer().score_chunked(premise, hypothesis, outer_agg)

    @property
    def model_available(self) -> bool:
        return any(s.model_available for s in self._scorers)

    @property
    def device_count(self) -> int:
        return len(self._scorers)
