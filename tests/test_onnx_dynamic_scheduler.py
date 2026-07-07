# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - ONNX Dynamic Scheduler Tests
"""Typed guard tests for the ONNX dynamic scheduler."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Protocol

import pytest

from director_ai.core.nli import OnnxDynamicBatcher

Pair = tuple[str, str]
ScoreFn = Callable[[list[Pair]], list[float]]


class _ProviderSession(Protocol):
    """Subset of ONNX Runtime session provider metadata used by the batcher."""

    def get_providers(self) -> list[str]:
        """Return the active ONNX Runtime provider names."""


class _FakeSession:
    """Session protocol object with deterministic provider metadata."""

    def __init__(self, providers: list[str]) -> None:
        self._providers = providers

    def get_providers(self) -> list[str]:
        """Return configured fake providers."""
        return self._providers


def _constant_scorer(value: float = 0.5) -> ScoreFn:
    """Return a scorer that emits one constant score per pair."""

    def score_fn(pairs: list[Pair]) -> list[float]:
        """Score every requested pair with a constant value."""
        return [value] * len(pairs)

    return score_fn


class TestOnnxDynamicBatcher:
    """Exercise ONNX dynamic batching without loading an ONNX model."""

    def _make_scorer(self) -> ScoreFn:
        """Return the default deterministic scorer for scheduler tests."""
        return _constant_scorer()

    def test_below_max_returns_empty(self) -> None:
        """Sub-threshold submissions should remain buffered."""
        batcher = OnnxDynamicBatcher(self._make_scorer(), max_batch=4)

        results = batcher.submit([("a", "b"), ("c", "d")])

        assert results == []

    def test_explicit_flush_drains_buffer(self) -> None:
        """Explicit flush should score all currently buffered pairs."""
        batcher = OnnxDynamicBatcher(self._make_scorer(), max_batch=4)
        batcher.submit([("a", "b"), ("c", "d")])

        results = batcher.flush()

        assert results == [0.5, 0.5]

    def test_flush_at_max_batch(self) -> None:
        """Reaching max_batch should flush immediately once."""
        call_count = 0

        def counting_fn(pairs: list[Pair]) -> list[float]:
            """Record scorer calls while returning one score per pair."""
            nonlocal call_count
            call_count += 1
            return [0.3] * len(pairs)

        batcher = OnnxDynamicBatcher(counting_fn, max_batch=2)

        results = batcher.submit([("a", "b"), ("c", "d")])

        assert results == [0.3, 0.3]
        assert call_count == 1

    def test_empty_submit(self) -> None:
        """Submitting no pairs to an empty scheduler should be a no-op."""
        batcher = OnnxDynamicBatcher(self._make_scorer(), max_batch=4)

        results = batcher.submit([])

        assert results == []

    def test_timeout_elapsed_flushes_pending_pairs(self) -> None:
        """A later submit should flush pending pairs after timeout expiry."""
        batcher = OnnxDynamicBatcher(
            self._make_scorer(),
            max_batch=4,
            flush_timeout_ms=1.0,
        )

        assert batcher.submit([("a", "b")]) == []
        time.sleep(0.01)

        assert batcher.submit([]) == [0.5]

    def test_zero_timeout_flushes_first_submit(self) -> None:
        """A zero timeout should flush any non-empty submit immediately."""
        batcher = OnnxDynamicBatcher(
            self._make_scorer(),
            max_batch=4,
            flush_timeout_ms=0.0,
        )

        assert batcher.submit([("a", "b")]) == [0.5]

    @pytest.mark.parametrize(
        ("max_batch", "flush_timeout_ms"),
        [(0, 10.0), (-1, 10.0), (4, -0.01), (4, float("inf"))],
    )
    def test_invalid_scheduler_parameters_fail_closed(
        self,
        max_batch: int,
        flush_timeout_ms: float,
    ) -> None:
        """Invalid scheduler parameters should be rejected at construction."""
        with pytest.raises(ValueError):
            OnnxDynamicBatcher(
                self._make_scorer(),
                max_batch=max_batch,
                flush_timeout_ms=flush_timeout_ms,
            )

    def test_score_consistency(self) -> None:
        """Scheduler should preserve scorer output ordering."""

        def score_fn(pairs: list[Pair]) -> list[float]:
            """Return a length-derived score for every pair."""
            return [
                float(len(premise) + len(hypothesis)) / 100.0
                for premise, hypothesis in pairs
            ]

        batcher = OnnxDynamicBatcher(score_fn, max_batch=8)
        batcher.submit([("hello", "world"), ("foo", "bar")])

        results = batcher.flush()

        assert results == [0.1, 0.06]

    def test_cuda_detection_with_session(self) -> None:
        """CUDA provider metadata should enable IO-binding mode."""
        session: _ProviderSession = _FakeSession(
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        batcher = OnnxDynamicBatcher(_constant_scorer(0.0), session=session)

        assert batcher.uses_io_binding is True

    def test_cpu_only_session(self) -> None:
        """CPU-only provider metadata should not enable IO binding."""
        session: _ProviderSession = _FakeSession(["CPUExecutionProvider"])

        batcher = OnnxDynamicBatcher(_constant_scorer(0.0), session=session)

        assert batcher.uses_io_binding is False

    def test_no_session(self) -> None:
        """Schedulers without runtime sessions should not use IO binding."""
        batcher = OnnxDynamicBatcher(_constant_scorer(0.0))

        assert batcher.uses_io_binding is False

    def test_thread_safety(self) -> None:
        """Concurrent submits should not lose flushed scores."""
        results_collector: list[float] = []
        lock = threading.Lock()
        batcher = OnnxDynamicBatcher(self._make_scorer(), max_batch=1)
        barrier = threading.Barrier(4)

        def worker() -> None:
            """Submit ten one-pair batches after all workers are ready."""
            barrier.wait()
            for _ in range(10):
                result = batcher.submit([("a", "b")])
                with lock:
                    results_collector.extend(result)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(results_collector) == 40
        assert all(score == 0.5 for score in results_collector)

    def test_large_batch_flushes_once(self) -> None:
        """A submit larger than max_batch should flush the whole buffer."""
        call_sizes: list[int] = []

        def tracking_fn(pairs: list[Pair]) -> list[float]:
            """Record each flushed batch size."""
            call_sizes.append(len(pairs))
            return [0.5] * len(pairs)

        batcher = OnnxDynamicBatcher(tracking_fn, max_batch=4)

        results = batcher.submit([("a", "b")] * 10)

        assert len(results) == 10
        assert call_sizes == [10]

    def test_max_batch_default(self) -> None:
        """The public default max_batch should remain 16."""
        batcher = OnnxDynamicBatcher(_constant_scorer(0.0))

        assert batcher.max_batch == 16

    @pytest.mark.parametrize("max_batch", [1, 4, 8, 16])
    def test_parametrised_max_batch(self, max_batch: int) -> None:
        """Configured max_batch values should control immediate flush size."""
        batcher = OnnxDynamicBatcher(
            lambda pairs: [0.5] * len(pairs),
            max_batch=max_batch,
        )

        results = batcher.submit([("a", "b")] * max_batch)

        assert batcher.max_batch == max_batch
        assert len(results) == max_batch

    @pytest.mark.parametrize("n_pairs", [4, 8, 16])
    def test_parametrised_submit_sizes(self, n_pairs: int) -> None:
        """Submits at or above max_batch should return one score per pair."""
        batcher = OnnxDynamicBatcher(lambda pairs: [0.5] * len(pairs), max_batch=4)

        results = batcher.submit([("p", "h")] * n_pairs)

        assert len(results) == n_pairs


class TestOnnxBatcherPerformanceDoc:
    """Document ONNX batcher pipeline performance."""

    def test_submit_fast(self) -> None:
        """Submit path should remain cheap for buffered scheduling."""
        batcher = OnnxDynamicBatcher(lambda pairs: [0.5] * len(pairs), max_batch=16)
        start = time.perf_counter()

        for _ in range(100):
            batcher.submit([("a", "b")])

        per_call_ms = (time.perf_counter() - start) / 100 * 1000
        assert per_call_ms < 5.0, f"Submit took {per_call_ms:.1f}ms"

    def test_batcher_returns_score_list_after_flush(self) -> None:
        """A flushed submit should return a list of float scores."""
        batcher = OnnxDynamicBatcher(lambda pairs: [0.5] * len(pairs), max_batch=2)

        result = batcher.submit([("a", "b"), ("c", "d")])

        assert isinstance(result, list)
        assert all(isinstance(score, float) for score in result)
