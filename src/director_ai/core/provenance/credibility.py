# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — SourceCredibility

"""Per-source trust score with exponential time decay.

Each observation updates the running score via exponential
moving average with a caller-configurable half-life. Older
observations decay smoothly — a source that was reliable a
month ago but stopped producing quality citations drifts toward
the prior rather than keeping its high score forever.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceScore:
    """One source's current score."""

    source_id: str
    score: float
    last_updated: float
    observation_count: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1]; got {self.score!r}")
        if self.observation_count < 0:
            raise ValueError("observation_count must be non-negative")


class SourceCredibility:
    """Thread-safe per-source credibility tracker.

    Parameters
    ----------
    half_life_seconds :
        Time after which an observation's weight halves. Default
        7 days (``7 * 24 * 3600``).
    prior :
        Score returned for a source with no observations. Default
        0.5 (maximum uncertainty).
    clock :
        Timestamp source; injection point for tests.
    """

    def __init__(
        self,
        *,
        half_life_seconds: float = 7 * 24 * 3600.0,
        prior: float = 0.5,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if half_life_seconds <= 0:
            raise ValueError("half_life_seconds must be positive")
        if not 0.0 <= prior <= 1.0:
            raise ValueError("prior must be in [0, 1]")
        self._half_life = half_life_seconds
        self._prior = prior
        self._clock = clock or time.time
        self._lock = threading.Lock()
        self._scores: dict[str, SourceScore] = {}

    def observe(self, source_id: str, signal: float) -> SourceScore:
        """Fold ``signal`` in ``[0, 1]`` into the source's running
        score. The signal's weight is computed from the elapsed
        time since the last update — ancient observations decay
        via the half-life before the new signal is merged."""
        if not source_id:
            raise ValueError("source_id must be non-empty")
        if not 0.0 <= signal <= 1.0:
            raise ValueError("signal must be in [0, 1]")
        now = float(self._clock())
        with self._lock:
            existing = self._scores.get(source_id)
            if existing is None:
                score = SourceScore(
                    source_id=source_id,
                    score=(self._prior + signal) / 2,
                    last_updated=now,
                    observation_count=1,
                )
            else:
                decay = _decay_weight(now - existing.last_updated, self._half_life)
                # EMA: new = decay * old + (1 - decay) * signal —
                # decay near 1 preserves old score; decay near 0
                # replaces it.
                new_value = decay * existing.score + (1.0 - decay) * signal
                score = SourceScore(
                    source_id=source_id,
                    score=max(0.0, min(1.0, new_value)),
                    last_updated=now,
                    observation_count=existing.observation_count + 1,
                )
            self._scores[source_id] = score
        return score

    def score(self, source_id: str) -> float:
        """Return the current decayed score for ``source_id``.

        Sources with no observations return the prior. Sources
        with old observations decay toward the prior via the
        half-life — the actual return is
        ``decay * stored + (1 - decay) * prior``.
        """
        if not source_id:
            raise ValueError("source_id must be non-empty")
        now = float(self._clock())
        with self._lock:
            existing = self._scores.get(source_id)
        if existing is None:
            return self._prior
        decay = _decay_weight(now - existing.last_updated, self._half_life)
        return decay * existing.score + (1.0 - decay) * self._prior

    def snapshot(self) -> tuple[SourceScore, ...]:
        """Return every tracked source's current record. Useful
        for audit and dashboards."""
        with self._lock:
            return tuple(self._scores.values())


def _decay_weight(elapsed_seconds: float, half_life: float) -> float:
    """Exponential decay: ``0.5`` at ``elapsed == half_life``."""
    if elapsed_seconds < 0:
        return 1.0
    return math.pow(0.5, elapsed_seconds / half_life)
