# SPDX-License-Identifier: AGPL-3.0-or-later
# © 1998–2026 Miroslav Šotek. All rights reserved.
"""Quantify PyO3 FFI overhead: Rust-native vs Python-via-FFI round-trip.

Compares:
  1. Pure Python StreamingKernel (streaming.py)
  2. Rust FFI StreamingKernel (backfire_kernel.RustStreamingKernel)
  3. Pure Python CoherenceScorer.review()
  4. Rust FFI RustCoherenceScorer.review()
  5. Pure Python UPDE step (if available)
  6. Rust FFI RustUPDEStepper.run()

Run:
    python -m benchmarks.ffi_overhead_bench
    python -m benchmarks.ffi_overhead_bench --iterations 100
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

TOKENS_PER_STREAM = 500
DEFAULT_ITERATIONS = 50


def bench_python_streaming(iterations: int) -> dict:
    from director_ai.core.streaming import StreamingKernel

    tokens = [f"tok_{i}" for i in range(TOKENS_PER_STREAM)]

    timings = []
    for _ in range(iterations):
        kernel = StreamingKernel(hard_limit=0.3, window_size=10)
        t0 = time.perf_counter()
        kernel.stream_tokens(tokens, lambda _text: 0.8)
        timings.append((time.perf_counter() - t0) * 1000)

    med = statistics.median(timings)
    return {
        "name": "python_streaming",
        "tokens": TOKENS_PER_STREAM,
        "iterations": iterations,
        "median_ms": round(med, 3),
        "p95_ms": round(sorted(timings)[int(len(timings) * 0.95)], 3),
        "us_per_token": round(med * 1000 / TOKENS_PER_STREAM, 2),
    }


def bench_rust_streaming(iterations: int) -> dict | None:
    try:
        from backfire_kernel import BackfireConfig, RustStreamingKernel
    except ImportError:
        print("  backfire_kernel not installed — skipping Rust streaming")
        return None

    config = BackfireConfig(hard_limit=0.5, window_size=10)
    timings = []
    for _ in range(iterations):
        kernel = RustStreamingKernel(config)
        tokens = [f"tok_{i}" for i in range(TOKENS_PER_STREAM)]
        t0 = time.perf_counter()
        kernel.stream_tokens(tokens, lambda _t: 0.8)
        timings.append((time.perf_counter() - t0) * 1000)

    med = statistics.median(timings)
    return {
        "name": "rust_ffi_streaming",
        "tokens": TOKENS_PER_STREAM,
        "iterations": iterations,
        "median_ms": round(med, 3),
        "p95_ms": round(sorted(timings)[int(len(timings) * 0.95)], 3),
        "us_per_token": round(med * 1000 / TOKENS_PER_STREAM, 2),
    }


def bench_python_scorer(iterations: int) -> dict:
    from director_ai.core.knowledge import GroundTruthStore
    from director_ai.core.scorer import CoherenceScorer

    store = GroundTruthStore()
    store.add("capital", "The capital of France is Paris")
    scorer = CoherenceScorer(threshold=0.5, use_nli=False, ground_truth_store=store)

    # Warmup
    for _ in range(5):
        scorer.review(
            "What is the capital of France?", "Paris is the capital of France."
        )

    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        scorer.review(
            "What is the capital of France?", "Paris is the capital of France."
        )
        timings.append((time.perf_counter() - t0) * 1000)

    return {
        "name": "python_scorer_review",
        "iterations": iterations,
        "median_ms": round(statistics.median(timings), 3),
        "p95_ms": round(sorted(timings)[int(len(timings) * 0.95)], 3),
    }


def bench_rust_scorer(iterations: int) -> dict | None:
    try:
        from backfire_kernel import RustCoherenceScorer
    except ImportError:
        print("  backfire_kernel not installed — skipping Rust scorer")
        return None

    scorer = RustCoherenceScorer()

    for _ in range(5):
        scorer.review(
            "What is the capital of France?", "Paris is the capital of France."
        )

    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        scorer.review(
            "What is the capital of France?", "Paris is the capital of France."
        )
        timings.append((time.perf_counter() - t0) * 1000)

    return {
        "name": "rust_ffi_scorer_review",
        "iterations": iterations,
        "median_ms": round(statistics.median(timings), 3),
        "p95_ms": round(sorted(timings)[int(len(timings) * 0.95)], 3),
    }


def _kuramoto_step_python(theta: list[float], omega: list[float], k: float, dt: float):
    """Minimal Kuramoto stepper in pure Python (no numpy)."""
    import math

    n = len(theta)
    dtheta = [0.0] * n
    for i in range(n):
        coupling = 0.0
        for j in range(n):
            coupling += math.sin(theta[j] - theta[i])
        dtheta[i] = omega[i] + k / n * coupling
    for i in range(n):
        theta[i] = (theta[i] + dt * dtheta[i]) % (2 * math.pi)


def bench_python_upde(iterations: int) -> dict:
    """Pure-Python Kuramoto stepper (mirrors Rust RustUPDEStepper)."""
    import math

    n = 16
    omega = [1.0 + 0.1 * i for i in range(n)]
    k = 0.45

    timings = []
    for _ in range(iterations):
        theta = [math.sin(i * 0.4) * math.pi for i in range(n)]
        t0 = time.perf_counter()
        for _ in range(100):
            _kuramoto_step_python(theta, omega, k, 0.01)
        timings.append((time.perf_counter() - t0) * 1000)

    return {
        "name": "python_upde_100steps",
        "iterations": iterations,
        "median_ms": round(statistics.median(timings), 3),
        "p95_ms": round(sorted(timings)[int(len(timings) * 0.95)], 3),
    }


def bench_rust_upde(iterations: int) -> dict | None:
    try:
        from backfire_kernel import RustUPDEStepper
    except ImportError:
        return None

    stepper = RustUPDEStepper(dt=0.01)

    import math

    theta = [math.sin(i * 0.4) * math.pi for i in range(16)]

    for _ in range(5):
        stepper.run(theta[:], 100)

    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        stepper.run(theta[:], 100)
        timings.append((time.perf_counter() - t0) * 1000)

    return {
        "name": "rust_ffi_upde_100steps",
        "iterations": iterations,
        "median_ms": round(statistics.median(timings), 3),
        "p95_ms": round(sorted(timings)[int(len(timings) * 0.95)], 3),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PyO3 FFI overhead benchmark")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    args = parser.parse_args()

    print(f"PyO3 FFI Overhead Benchmark — {args.iterations} iterations\n")

    results = []
    pairs = [
        ("Streaming", bench_python_streaming, bench_rust_streaming),
        ("Scorer", bench_python_scorer, bench_rust_scorer),
        ("UPDE", bench_python_upde, bench_rust_upde),
    ]

    for label, py_fn, rs_fn in pairs:
        print(f"{label}:")
        py = py_fn(args.iterations)
        results.append(py)
        rs = rs_fn(args.iterations)
        if rs:
            results.append(rs)
            if "median_ms" in py and "median_ms" in rs:
                ratio = (
                    py["median_ms"] / rs["median_ms"]
                    if rs["median_ms"] > 0
                    else float("inf")
                )
                print(
                    f"  Python: {py['median_ms']:.3f}ms  Rust FFI: {rs['median_ms']:.3f}ms  -> {ratio:.1f}x speedup"
                )
            else:
                print(f"  Python: {py}  Rust: {rs}")
        else:
            print(f"  Python: {py['median_ms']:.3f}ms  Rust: not available")
        print()

    out = Path(__file__).parent / "results" / "ffi_overhead_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
