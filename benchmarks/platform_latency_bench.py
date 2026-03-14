# SPDX-License-Identifier: AGPL-3.0-or-later
# © 1998–2026 Miroslav Šotek. All rights reserved.
"""Cross-platform latency profiling with memory and GC overhead.

Measures Director-AI scorer latency across backends, reports platform
info, peak RSS, and GC pause overhead.

Run:
    python -m benchmarks.platform_latency_bench
    python -m benchmarks.platform_latency_bench --iterations 100
"""

from __future__ import annotations

import gc
import json
import platform
import statistics
import time
from pathlib import Path


def platform_info() -> dict:
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "cpu": platform.processor() or "unknown",
    }


def measure_gc_overhead(n_objects: int = 50_000) -> dict:
    """Measure GC pause by forcing collection after creating throwaway objects."""
    _throwaway = [{"k": i, "v": [i] * 10} for i in range(n_objects)]
    del _throwaway

    gc.collect()
    gc.disable()

    pauses = []
    for _ in range(20):
        _objs = [{"k": i} for i in range(n_objects)]
        del _objs
        t0 = time.perf_counter()
        gc.collect()
        pauses.append((time.perf_counter() - t0) * 1000)

    gc.enable()
    return {
        "gc_median_ms": round(statistics.median(pauses), 3),
        "gc_p95_ms": round(sorted(pauses)[int(len(pauses) * 0.95)], 3),
        "gc_max_ms": round(max(pauses), 3),
        "n_objects": n_objects,
    }


def get_peak_rss_mb() -> float:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        pass
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def bench_heuristic(iterations: int) -> dict:
    """Heuristic scorer (no NLI model)."""
    from director_ai.core.knowledge import GroundTruthStore
    from director_ai.core.scorer import CoherenceScorer

    store = GroundTruthStore()
    store.add("sky", "The sky is blue")
    scorer = CoherenceScorer(threshold=0.5, use_nli=False, ground_truth_store=store)

    # Warmup
    for _ in range(5):
        scorer.review("What color is the sky?", "The sky is blue.")

    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        scorer.review("What color is the sky?", "The sky is blue.")
        timings.append((time.perf_counter() - t0) * 1000)

    return {
        "name": "heuristic",
        "iterations": iterations,
        "median_ms": round(statistics.median(timings), 3),
        "p95_ms": round(sorted(timings)[int(len(timings) * 0.95)], 3),
        "min_ms": round(min(timings), 3),
        "max_ms": round(max(timings), 3),
    }


def bench_streaming(iterations: int) -> dict:
    """StreamingKernel token throughput."""
    from director_ai.core.streaming import StreamingKernel

    tokens = [f"token_{i}" for i in range(100)]

    timings = []
    for _ in range(iterations):
        kernel = StreamingKernel(hard_limit=0.3, window_size=10)
        t0 = time.perf_counter()
        kernel.stream_tokens(tokens, lambda _text: 0.8)
        timings.append((time.perf_counter() - t0) * 1000)

    med = statistics.median(timings)
    return {
        "name": "streaming_100tok",
        "iterations": iterations,
        "median_ms": round(med, 3),
        "p95_ms": round(sorted(timings)[int(len(timings) * 0.95)], 3),
        "us_per_token": round(med * 1000 / 100, 2),
    }


def bench_lite(iterations: int) -> dict:
    """Lite scorer backend."""
    from director_ai.core.knowledge import GroundTruthStore
    from director_ai.core.scorer import CoherenceScorer

    store = GroundTruthStore()
    store.add("sky", "The sky is blue")
    scorer = CoherenceScorer(
        threshold=0.5,
        scorer_backend="lite",
        ground_truth_store=store,
    )

    for _ in range(5):
        scorer.review("What color is the sky?", "The sky is blue.")

    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        scorer.review("What color is the sky?", "The sky is blue.")
        timings.append((time.perf_counter() - t0) * 1000)

    return {
        "name": "lite",
        "iterations": iterations,
        "median_ms": round(statistics.median(timings), 3),
        "p95_ms": round(sorted(timings)[int(len(timings) * 0.95)], 3),
        "min_ms": round(min(timings), 3),
        "max_ms": round(max(timings), 3),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cross-platform latency profiler")
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    info = platform_info()
    print(f"Platform: {info['os']} {info['os_version']} ({info['arch']})")
    print(f"Python:   {info['python']}")
    print(f"CPU:      {info['cpu']}")
    print()

    results = {"platform": info, "benchmarks": []}

    gc_info = measure_gc_overhead()
    results["gc_overhead"] = gc_info
    print(
        f"GC overhead: median={gc_info['gc_median_ms']:.3f}ms  p95={gc_info['gc_p95_ms']:.3f}ms",
    )
    print()

    for bench_fn in [bench_heuristic, bench_lite, bench_streaming]:
        r = bench_fn(args.iterations)
        results["benchmarks"].append(r)
        print(
            f"{r['name']:20s}  median={r['median_ms']:.3f}ms  p95={r['p95_ms']:.3f}ms",
        )

    rss = get_peak_rss_mb()
    results["peak_rss_mb"] = round(rss, 1)
    if rss > 0:
        print(f"\nPeak RSS: {rss:.1f} MB")

    out = Path(__file__).parent / "results" / "platform_latency_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
