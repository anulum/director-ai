# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — federated privacy hot-path benchmark
"""Measure the federated-privacy aggregation hot paths, Rust vs Python.

Two reductions back the package's compute: integer count aggregation in the
federated histogram/counter (``rust_sum_i64``) and floating-point epsilon
composition in the privacy accountant (``rust_sum_f64``). Both have a Rust path
and a pure-Python reference; this benchmark asserts they agree before reporting
per-call latency.

The result records the host-load/isolation context required by the 2026-06-04
policy. A workstation run is labelled ``isolated: false`` and is valid as
functional, parity, and local-regression evidence only.

Usage::

    python -m benchmarks.federated_privacy
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import director_ai.core.federated_privacy.accountant as accountant_mod
import director_ai.core.federated_privacy.aggregator as aggregator_mod

RESULTS_DIR = Path(__file__).parent / "results"

SIZES = (64, 1_024, 16_384)
ITERATIONS = 5000


def _counts(n: int) -> list[int]:
    return [(i * 7 + 3) % 1000 for i in range(n)]


def _epsilons(n: int) -> list[float]:
    return [0.01 + (i % 50) * 0.002 for i in range(n)]


def _time(fn, *, iterations: int) -> dict:
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e6)
    samples.sort()
    return {
        "median_us": round(statistics.median(samples), 3),
        "p95_us": round(samples[int(len(samples) * 0.95)], 3),
    }


def _bench_sum_int(size: int) -> dict:
    values = _counts(size)
    previous = aggregator_mod._RUST_AGGREGATOR
    try:
        aggregator_mod._RUST_AGGREGATOR = True
        rust_total = aggregator_mod._sum_int(values)
        rust = _time(lambda: aggregator_mod._sum_int(values), iterations=ITERATIONS)
        aggregator_mod._RUST_AGGREGATOR = False
        python_total = aggregator_mod._sum_int(values)
        python = _time(lambda: aggregator_mod._sum_int(values), iterations=ITERATIONS)
    finally:
        aggregator_mod._RUST_AGGREGATOR = previous
    if rust_total != python_total:
        raise AssertionError(
            f"sum_int parity mismatch at {size}: rust={rust_total} python={python_total}"
        )
    return _row(size, rust_total, rust, python)


def _bench_sum_float(size: int) -> dict:
    values = _epsilons(size)
    previous = accountant_mod._RUST_ACCOUNTANT
    try:
        accountant_mod._RUST_ACCOUNTANT = True
        rust_total = accountant_mod._sum_float(values)
        rust = _time(lambda: accountant_mod._sum_float(values), iterations=ITERATIONS)
        accountant_mod._RUST_ACCOUNTANT = False
        python_total = accountant_mod._sum_float(values)
        python = _time(lambda: accountant_mod._sum_float(values), iterations=ITERATIONS)
    finally:
        accountant_mod._RUST_ACCOUNTANT = previous
    if not math.isclose(rust_total, python_total, rel_tol=1e-9, abs_tol=1e-12):
        raise AssertionError(
            f"sum_float parity mismatch at {size}: "
            f"rust={rust_total} python={python_total}"
        )
    return _row(size, rust_total, rust, python)


def _row(size: int, total: float, rust: dict, python: dict) -> dict:
    speedup = (
        round(python["median_us"] / rust["median_us"], 2)
        if rust["median_us"] > 0
        else None
    )
    return {
        "size": size,
        "total": total,
        "parity": "rust == python",
        "rust": rust,
        "python": python,
        "speedup_rust_over_python": speedup,
    }


def _host_context() -> dict:
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:  # pragma: no cover - platform without loadavg
        load1 = load5 = load15 = -1.0
    return {
        "isolated": False,
        "isolation_method": "none",
        "evidence_class": "functional+parity+local-regression",
        "command": "python -m benchmarks.federated_privacy",
        "cpu_count": os.cpu_count(),
        "loadavg_1_5_15": [round(load1, 2), round(load5, 2), round(load15, 2)],
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print(f"{'Size':>8} {'Python (us)':>14} {'Rust (us)':>12} {'Speedup':>10}")
    print("-" * 48)
    for row in rows:
        speedup = row["speedup_rust_over_python"]
        speedup_str = f"{speedup:.2f}x" if speedup is not None else "N/A"
        print(
            f"{row['size']:>8} {row['python']['median_us']:>14.3f} "
            f"{row['rust']['median_us']:>12.3f} {speedup_str:>10}"
        )


def main() -> None:
    print("Federated privacy hot-path benchmark")
    print("=" * 48)
    count_rows = [_bench_sum_int(size) for size in SIZES]
    epsilon_rows = [_bench_sum_float(size) for size in SIZES]
    _print_table("Count aggregation (rust_sum_i64)", count_rows)
    _print_table("Epsilon composition (rust_sum_f64)", epsilon_rows)
    output = {
        "benchmark": "federated_privacy",
        "host_context": _host_context(),
        "count_aggregation": count_rows,
        "epsilon_composition": epsilon_rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "federated_privacy.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
