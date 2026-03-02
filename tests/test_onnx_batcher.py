# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — ONNX Dynamic Batcher Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import threading
from unittest.mock import MagicMock

from director_ai.core.nli import OnnxDynamicBatcher


class TestOnnxDynamicBatcher:
    def _make_scorer(self):
        def score_fn(pairs):
            return [0.5] * len(pairs)
        return score_fn

    def test_flush_on_submit(self):
        fn = self._make_scorer()
        batcher = OnnxDynamicBatcher(fn, max_batch=4)
        pairs = [("a", "b"), ("c", "d")]
        results = batcher.submit(pairs)
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
        results = batcher.submit(pairs)
        assert len(results) == 2
        assert results[0] == 0.1  # (5+5)/100
        assert results[1] == 0.06  # (3+3)/100

    def test_cuda_detection_with_mock_session(self):
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
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

        batcher = OnnxDynamicBatcher(score_fn, max_batch=16)
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
        results = batcher.submit([("a", "b")] * 10)
        assert len(results) == 10

    def test_max_batch_default(self):
        batcher = OnnxDynamicBatcher(lambda p: [])
        assert batcher.max_batch == 16
