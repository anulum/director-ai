# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — scorer parallel-future cleanup
"""Regression tests for scorer parallel-future cleanup paths."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from director_ai.core import CoherenceScorer


class _LogicFuture:
    """Future whose result raises like a logical-divergence worker failure."""

    def result(self) -> float:
        """Raise the worker error under test."""

        raise RuntimeError("logic divergence unavailable")


class _FactFuture:
    """Future spy for the factual-divergence worker."""

    cancelled: bool
    collected: bool

    def __init__(self) -> None:
        self.cancelled = False
        self.collected = False

    def cancel(self) -> bool:
        """Record cancellation and report that the worker had not started."""

        self.cancelled = True
        return True

    def result(self) -> tuple[float, None]:
        """Record collection for assertions when cancellation is impossible."""

        self.collected = True
        return 0.0, None


class _Pool:
    """Two-submit pool returning logical then factual futures."""

    fact_future: _FactFuture
    calls: int

    def __init__(self) -> None:
        self.fact_future = _FactFuture()
        self.calls = 0

    def submit(
        self,
        _fn: Callable[..., object],
        *_args: object,
        **_kwargs: object,
    ) -> _LogicFuture | _FactFuture:
        """Return the logical future first and the factual future second."""

        self.calls += 1
        return _LogicFuture() if self.calls == 1 else self.fact_future


def test_parallel_logic_failure_cancels_factual_future(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A logical-worker failure must not leave the factual future orphaned."""

    scorer = CoherenceScorer(use_nli=False)
    pool = _Pool()
    monkeypatch.setattr(scorer, "_get_parallel_pool", lambda: pool)

    with pytest.raises(RuntimeError, match="logic divergence unavailable"):
        scorer._heuristic_coherence("prompt", "response")

    assert pool.fact_future.cancelled is True
    assert pool.fact_future.collected is False
