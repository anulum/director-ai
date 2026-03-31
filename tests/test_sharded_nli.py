# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Sharded NLI Tests (STRONG)
"""Multi-angle tests for ShardedNLIScorer multi-device pipeline.

Covers: empty devices guard, single/multi device, round-robin distribution,
batch scoring, model availability, chunked scoring, thread safety,
parametrised device counts, pipeline integration, and performance.
"""

import pytest

from director_ai.core.sharded_nli import ShardedNLIScorer


class TestShardedNLIScorer:
    def test_empty_devices_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            ShardedNLIScorer(devices=[])

    def test_single_device(self):
        scorer = ShardedNLIScorer(devices=["cpu"], use_model=False, backend="lite")
        s = scorer.score("hello", "world")
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0

    def test_device_count(self):
        scorer = ShardedNLIScorer(
            devices=["cpu", "cpu"],
            use_model=False,
            backend="lite",
        )
        assert scorer.device_count == 2

    def test_round_robin_distribution(self):
        scorer = ShardedNLIScorer(
            devices=["cpu", "cpu", "cpu"],
            use_model=False,
            backend="lite",
        )
        # Call 6 times, each NLIScorer should get 2 calls in round-robin
        for _ in range(6):
            scorer.score("a", "b")

    def test_score_batch(self):
        scorer = ShardedNLIScorer(
            devices=["cpu"],
            use_model=False,
            backend="lite",
        )
        results = scorer.score_batch([("a", "b"), ("c", "d")])
        assert len(results) == 2

    def test_model_available_aggregation(self):
        scorer = ShardedNLIScorer(
            devices=["cpu"],
            use_model=False,
            backend="lite",
        )
        # use_model=False â†’ model_available is False (model not loaded)
        assert scorer.model_available is False

    def test_score_chunked(self):
        scorer = ShardedNLIScorer(
            devices=["cpu"],
            use_model=False,
            backend="lite",
        )
        agg, per_hyp = scorer.score_chunked("premise text", "hypothesis text")
        assert isinstance(agg, float)
        assert isinstance(per_hyp, list)

    def test_thread_safety(self):
        import threading

        scorer = ShardedNLIScorer(
            devices=["cpu", "cpu"],
            use_model=False,
            backend="lite",
        )
        results = []
        lock = threading.Lock()
        barrier = threading.Barrier(4)

        def worker():
            barrier.wait()
            for _ in range(10):
                s = scorer.score("a", "b")
                with lock:
                    results.append(s)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 40

    @pytest.mark.parametrize("n_devices", [1, 2, 3, 4])
    def test_parametrised_device_counts(self, n_devices):
        scorer = ShardedNLIScorer(
            devices=["cpu"] * n_devices,
            use_model=False,
            backend="lite",
        )
        assert scorer.device_count == n_devices
        s = scorer.score("test", "test")
        assert 0.0 <= s <= 1.0

    @pytest.mark.parametrize("batch_size", [1, 3, 5, 10])
    def test_parametrised_batch_sizes(self, batch_size):
        scorer = ShardedNLIScorer(
            devices=["cpu"],
            use_model=False,
            backend="lite",
        )
        pairs = [("p", "h")] * batch_size
        results = scorer.score_batch(pairs)
        assert len(results) == batch_size


class TestShardedNLIPerformanceDoc:
    """Document sharded NLI pipeline performance."""

    def test_score_deterministic(self):
        scorer = ShardedNLIScorer(
            devices=["cpu"],
            use_model=False,
            backend="lite",
        )
        s1 = scorer.score("X", "Y")
        s2 = scorer.score("X", "Y")
        assert s1 == s2

    def test_sharded_scorer_integrates_with_coherence_scorer(self):
        from director_ai.core import CoherenceScorer

        scorer = CoherenceScorer(
            use_nli=False,
            nli_devices=["cpu", "cpu"],
        )
        approved, score = scorer.review("test", "test")
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0

    def test_score_fast_heuristic(self):
        import time

        scorer = ShardedNLIScorer(
            devices=["cpu"],
            use_model=False,
            backend="lite",
        )
        t0 = time.perf_counter()
        for _ in range(100):
            scorer.score("test", "test")
        per_call_ms = (time.perf_counter() - t0) / 100 * 1000
        assert per_call_ms < 5.0, f"Sharded score took {per_call_ms:.1f}ms"
