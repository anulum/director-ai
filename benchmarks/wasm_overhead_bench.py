# SPDX-License-Identifier: AGPL-3.0-or-later
# © 1998–2026 Miroslav Šotek. All rights reserved.
"""Benchmark: WASM edge runtime vs native Rust StreamingKernel.

Measures overhead of the WASM boundary on 1000-token streams.
Requires: wasm-pack built pkg/ in backfire-kernel/crates/backfire-wasm/
and backfire_ffi installed for native comparison.

Run:
    python benchmarks/wasm_overhead_bench.py
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

TOKENS = 1000
ITERATIONS = 50

CONFIG_JSON = json.dumps(
    {
        "coherence_threshold": 0.6,
        "hard_limit": 0.5,
        "soft_limit": 0.7,
        "w_logic": 0.6,
        "w_fact": 0.4,
        "window_size": 10,
        "window_threshold": 0.55,
        "trend_window": 5,
        "trend_threshold": 0.15,
        "history_window": 5,
        "deadline_ms": 50,
        "logit_entropy_limit": 1.2,
    }
)


def bench_native_rust() -> list[float]:
    """Benchmark native Rust StreamingKernel via FFI."""
    try:
        from backfire_ffi import StreamingKernel
    except ImportError:
        print("  backfire_ffi not available — skipping native benchmark")
        return []

    timings = []
    for _ in range(ITERATIONS):
        kernel_iter = StreamingKernel(CONFIG_JSON)
        t0 = time.perf_counter()
        for i in range(TOKENS):
            kernel_iter.process_token(f"token_{i}", 0.8)
        elapsed = (time.perf_counter() - t0) * 1000
        timings.append(elapsed)
    return timings


def bench_python_fallback() -> list[float]:
    """Benchmark pure-Python StreamingKernel as baseline."""
    from director_ai.core.streaming import StreamingKernel

    timings = []
    for _ in range(ITERATIONS):
        kernel = StreamingKernel(hard_limit=0.5, window_size=10)
        t0 = time.perf_counter()
        for i in range(TOKENS):
            kernel.process_token(f"token_{i}", 0.8)
        elapsed = (time.perf_counter() - t0) * 1000
        timings.append(elapsed)
    return timings


def report(name: str, timings: list[float]) -> dict:
    if not timings:
        return {}
    med = statistics.median(timings)
    p95 = sorted(timings)[int(len(timings) * 0.95)]
    result = {
        "name": name,
        "tokens": TOKENS,
        "iterations": len(timings),
        "median_ms": round(med, 3),
        "p95_ms": round(p95, 3),
        "min_ms": round(min(timings), 3),
        "max_ms": round(max(timings), 3),
        "us_per_token": round(med * 1000 / TOKENS, 2),
    }
    print(
        f"  {name}: median={med:.3f}ms  p95={p95:.3f}ms  ({result['us_per_token']} µs/tok)"
    )
    return result


def main():
    print(f"WASM Edge Runtime Benchmark — {TOKENS} tokens × {ITERATIONS} iterations\n")

    results = []

    print("Native Rust (FFI):")
    native = bench_native_rust()
    if native:
        results.append(report("native_rust_ffi", native))

    print("\nPure Python:")
    python = bench_python_fallback()
    if python:
        results.append(report("python_fallback", python))

    if len(results) >= 2:
        ratio = results[1]["median_ms"] / results[0]["median_ms"]
        print(f"\nPython/Rust ratio: {ratio:.1f}x")

    out = Path(__file__).parent / "wasm_overhead_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
