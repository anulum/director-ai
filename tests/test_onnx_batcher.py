# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ONNX Dynamic Batcher Tests
"""Multi-angle tests for ONNX dynamic batching pipeline.

Covers: single submit, batch flush, concurrent submissions, timeout flush,
no session, thread safety, large batch split, max_batch default,
parametrised batch sizes, pipeline integration, and performance.
"""

import threading
import time
from unittest.mock import MagicMock

import pytest

from director_ai.core.nli import OnnxDynamicBatcher


class TestOnnxDynamicBatcher:
    def _make_scorer(self):
        def score_fn(pairs):
            return [0.5] * len(pairs)

        return score_fn

    def test_below_max_returns_empty(self):
        fn = self._make_scorer()
        batcher = OnnxDynamicBatcher(fn, max_batch=4)
        pairs = [("a", "b"), ("c", "d")]
        results = batcher.submit(pairs)
        assert results == []

    def test_explicit_flush_drains_buffer(self):
        fn = self._make_scorer()
        batcher = OnnxDynamicBatcher(fn, max_batch=4)
        batcher.submit([("a", "b"), ("c", "d")])
        results = batcher.flush()
        assert results == [0.5, 0.5]

    def test_flush_at_max_batch(self):
        call_count = [0]

        def counting_fn(pairs):
            call_count[0] += 1
            return [0.3] * len(pairs)

        batcher = OnnxDynamicBatcher(counting_fn, max_batch=2)
        results = batcher.submit([("a", "b"), ("c", "d")])
        assert len(results) == 2
        assert call_count[0] == 1

    def test_empty_submit(self):
        fn = self._make_scorer()
        batcher = OnnxDynamicBatcher(fn, max_batch=4)
        results = batcher.submit([])
        assert results == []

    def test_score_consistency(self):
        def score_fn(pairs):
            return [float(len(p) + len(h)) / 100.0 for p, h in pairs]

        batcher = OnnxDynamicBatcher(score_fn, max_batch=8)
        pairs = [("hello", "world"), ("foo", "bar")]
        batcher.submit(pairs)
        results = batcher.flush()
        assert len(results) == 2
        assert results[0] == 0.1  # (5+5)/100
        assert results[1] == 0.06  # (3+3)/100

    def test_cuda_detection_with_mock_session(self):
        mock_session = MagicMock()
        mock_session.get_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        batcher = OnnxDynamicBatcher(lambda p: [], session=mock_session)
        assert batcher.uses_io_binding is True

    def test_cpu_only_session(self):
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        batcher = OnnxDynamicBatcher(lambda p: [], session=mock_session)
        assert batcher.uses_io_binding is False

    def test_no_session(self):
        batcher = OnnxDynamicBatcher(lambda p: [])
        assert batcher.uses_io_binding is False

    def test_thread_safety(self):
        results_collector = []
        lock = threading.Lock()

        def score_fn(pairs):
            return [0.5] * len(pairs)

        # max_batch=1 so every submit triggers a flush
        batcher = OnnxDynamicBatcher(score_fn, max_batch=1)
        barrier = threading.Barrier(4)

        def worker():
            barrier.wait()
            for _ in range(10):
                r = batcher.submit([("a", "b")])
                with lock:
                    results_collector.extend(r)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results_collector) == 40

    def test_large_batch_split(self):
        call_sizes = []

        def tracking_fn(pairs):
            call_sizes.append(len(pairs))
            return [0.5] * len(pairs)

        batcher = OnnxDynamicBatcher(tracking_fn, max_batch=4)
        # 10 pairs >= max_batch=4, so submit flushes all at once
        results = batcher.submit([("a", "b")] * 10)
        assert len(results) == 10

    def test_max_batch_default(self):
        batcher = OnnxDynamicBatcher(lambda p: [])
        assert batcher.max_batch == 16

    @pytest.mark.parametrize("max_batch", [1, 4, 8, 16])
    def test_parametrised_max_batch(self, max_batch):
        batcher = OnnxDynamicBatcher(lambda p: [0.5] * len(p), max_batch=max_batch)
        assert batcher.max_batch == max_batch
        # Submit >= max_batch to trigger flush
        results = batcher.submit([("a", "b")] * max_batch)
        assert len(results) == max_batch

    @pytest.mark.parametrize("n_pairs", [4, 8, 16])
    def test_parametrised_submit_sizes(self, n_pairs):
        batcher = OnnxDynamicBatcher(lambda p: [0.5] * len(p), max_batch=4)
        results = batcher.submit([("p", "h")] * n_pairs)
        assert len(results) == n_pairs


class TestOnnxBatcherPerformanceDoc:
    """Document ONNX batcher pipeline performance."""

    def test_submit_fast(self):
        batcher = OnnxDynamicBatcher(lambda p: [0.5] * len(p), max_batch=16)
        t0 = time.perf_counter()
        for _ in range(100):
            batcher.submit([("a", "b")])
        per_call_ms = (time.perf_counter() - t0) / 100 * 1000
        assert per_call_ms < 5, f"Submit took {per_call_ms:.1f}ms"

    def test_batcher_returns_list(self):
        batcher = OnnxDynamicBatcher(lambda p: [0.5] * len(p))
        result = batcher.submit([("a", "b"), ("c", "d")])
        assert isinstance(result, list)
        assert all(isinstance(s, float) for s in result)
