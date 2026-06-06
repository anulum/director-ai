# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — provenance ledger commitment benchmark
"""Measure the content-commitment Merkle root, Rust path versus Python.

The Rust kernel and the pure-Python reference must agree bit-for-bit on
the root; this benchmark asserts that parity at every leaf count before
reporting per-call latency, then records end-to-end ledger append
throughput for context.

The result file records the host-load and isolation context required by
the 2026-06-04 benchmark-core-isolation policy. A workstation run is
labelled ``isolated: false`` and is valid as functional, parity, and
local-regression evidence only — not as an isolated production latency
claim.

Usage::

    python -m benchmarks.provenance_ledger_commitment
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import director_ai.core.provenance.content_commitment as cc
from director_ai.core.provenance import KnowledgeProvenanceLedger, commit_root

RESULTS_DIR = Path(__file__).parent / "results"

LEAF_COUNTS = (16, 256, 4096)
ROOT_ITERATIONS = 2000
APPEND_ITERATIONS = 2000


def _leaves(count: int) -> list[bytes]:
    return [hashlib.sha256(f"chunk-{i}".encode()).digest() for i in range(count)]


def _time_root(leaves: list[bytes], *, use_rust: bool, iterations: int) -> dict:
    """Return median/p95 microseconds for commit_root over ``leaves``."""
    previous = cc._RUST_MERKLE
    cc._RUST_MERKLE = use_rust
    try:
        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            commit_root(leaves)
            samples.append((time.perf_counter() - start) * 1e6)
    finally:
        cc._RUST_MERKLE = previous
    samples.sort()
    return {
        "median_us": round(statistics.median(samples), 3),
        "p95_us": round(samples[int(len(samples) * 0.95)], 3),
        "backend": "rust" if use_rust else "python",
    }


def _assert_parity(leaves: list[bytes]) -> str:
    """Return the shared root hex after asserting Rust == Python."""
    previous = cc._RUST_MERKLE
    try:
        cc._RUST_MERKLE = True
        rust_root = commit_root(leaves)
        cc._RUST_MERKLE = False
        python_root = commit_root(leaves)
    finally:
        cc._RUST_MERKLE = previous
    if rust_root != python_root:
        raise AssertionError(
            f"parity mismatch at {len(leaves)} leaves: "
            f"rust={rust_root.hex()} python={python_root.hex()}"
        )
    return rust_root.hex()


def _bench_root() -> list[dict]:
    rows = []
    for count in LEAF_COUNTS:
        leaves = _leaves(count)
        root_hex = _assert_parity(leaves)
        rust = _time_root(leaves, use_rust=True, iterations=ROOT_ITERATIONS)
        python = _time_root(leaves, use_rust=False, iterations=ROOT_ITERATIONS)
        speedup = (
            round(python["median_us"] / rust["median_us"], 2)
            if rust["median_us"] > 0
            else None
        )
        rows.append(
            {
                "leaf_count": count,
                "root_hex": root_hex,
                "parity": "rust == python",
                "rust": rust,
                "python": python,
                "speedup_rust_over_python": speedup,
            }
        )
    return rows


def _bench_append() -> dict:
    """Measure in-memory ledger ingest-append latency (Rust commitment)."""
    ledger = KnowledgeProvenanceLedger(secret=b"k" * 32, path=None)
    leaf = hashlib.sha256(b"chunk").digest()
    samples = []
    for index in range(APPEND_ITERATIONS):
        chunk_leaves = [(f"doc{index}:c0", leaf), (f"doc{index}:c1", leaf)]
        start = time.perf_counter()
        ledger.record_ingest(
            doc_id=f"doc{index}",
            tenant_id="t",
            source="bench",
            content_hash="h",
            chunk_leaves=chunk_leaves,
        )
        samples.append((time.perf_counter() - start) * 1e6)
    samples.sort()
    return {
        "iterations": APPEND_ITERATIONS,
        "median_us": round(statistics.median(samples), 3),
        "p95_us": round(samples[int(len(samples) * 0.95)], 3),
    }


def _host_context() -> dict:
    """Record host-load and isolation context for the result artefact."""
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:  # pragma: no cover - platform without loadavg
        load1 = load5 = load15 = -1.0
    return {
        "isolated": False,
        "isolation_method": "none",
        "evidence_class": "functional+parity+local-regression",
        "command": "python -m benchmarks.provenance_ledger_commitment",
        "cpu_count": os.cpu_count(),
        "loadavg_1_5_15": [round(load1, 2), round(load5, 2), round(load15, 2)],
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def main() -> None:
    print("Provenance ledger commitment benchmark")
    print("=" * 66)
    root_rows = _bench_root()
    print(f"\n{'Leaves':>8} {'Python (us)':>14} {'Rust (us)':>12} {'Speedup':>10}")
    print("-" * 66)
    for row in root_rows:
        speedup = row["speedup_rust_over_python"]
        speedup_str = f"{speedup:.2f}x" if speedup is not None else "N/A"
        print(
            f"{row['leaf_count']:>8} {row['python']['median_us']:>14.3f} "
            f"{row['rust']['median_us']:>12.3f} {speedup_str:>10}"
        )

    append = _bench_append()
    print(
        f"\nLedger ingest append: median {append['median_us']:.3f} us, "
        f"p95 {append['p95_us']:.3f} us ({append['iterations']} appends)"
    )

    output = {
        "benchmark": "provenance_ledger_commitment",
        "host_context": _host_context(),
        "root_commitment": root_rows,
        "ledger_append": append,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "provenance_ledger_commitment.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
