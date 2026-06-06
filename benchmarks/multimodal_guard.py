# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — multimodal hash-bag guard benchmark
"""Measure the dependency-free multimodal guard hot path, Rust vs Python.

The hash-bag encoder and cross-modal verifier sum/normalise through the Rust
``rust_sum_f64`` kernel with a pure-Python reference. This benchmark asserts
the two paths agree on the cross-modal similarity at every input size before
reporting per-call latency for the encode+verify round trip.

The result records the host-load/isolation context required by the
2026-06-04 policy. A workstation run is labelled ``isolated: false`` and is
valid as functional, parity, and local-regression evidence only.

Usage::

    python -m benchmarks.multimodal_guard
"""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import director_ai.core.multimodal_guard.encoders as encoders_mod
import director_ai.core.multimodal_guard.verifier as verifier_mod
from director_ai.core.multimodal_guard import (
    HashBagCrossModalVerifier,
    HashBagImageEncoder,
)

RESULTS_DIR = Path(__file__).parent / "results"

IMAGE_SIZES = (1_024, 16_384, 262_144)
CLAIM = "a photograph of a mountain lake at sunrise with pine trees"
ITERATIONS = 2000


def _image(size: int) -> bytes:
    return bytes((i * 31 + 7) % 256 for i in range(size))


def _set_rust(enabled: bool) -> None:
    encoders_mod._RUST_MULTIMODAL_ENCODERS = enabled
    verifier_mod._RUST_MULTIMODAL_VERIFIER = enabled


def _round_trip(image: bytes) -> float:
    encoder = HashBagImageEncoder(dim=512)
    verifier = HashBagCrossModalVerifier(dim=512)
    return verifier.verify(encoder.encode(image), CLAIM)


def _time_round_trip(image: bytes, *, use_rust: bool, iterations: int) -> dict:
    previous = (
        encoders_mod._RUST_MULTIMODAL_ENCODERS,
        verifier_mod._RUST_MULTIMODAL_VERIFIER,
    )
    _set_rust(use_rust)
    try:
        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            _round_trip(image)
            samples.append((time.perf_counter() - start) * 1e6)
    finally:
        encoders_mod._RUST_MULTIMODAL_ENCODERS = previous[0]
        verifier_mod._RUST_MULTIMODAL_VERIFIER = previous[1]
    samples.sort()
    return {
        "median_us": round(statistics.median(samples), 3),
        "p95_us": round(samples[int(len(samples) * 0.95)], 3),
        "backend": "rust" if use_rust else "python",
    }


def _assert_parity(image: bytes) -> float:
    previous = (
        encoders_mod._RUST_MULTIMODAL_ENCODERS,
        verifier_mod._RUST_MULTIMODAL_VERIFIER,
    )
    try:
        _set_rust(True)
        rust_sim = _round_trip(image)
        _set_rust(False)
        python_sim = _round_trip(image)
    finally:
        encoders_mod._RUST_MULTIMODAL_ENCODERS = previous[0]
        verifier_mod._RUST_MULTIMODAL_VERIFIER = previous[1]
    if not math.isclose(rust_sim, python_sim, rel_tol=1e-9, abs_tol=1e-12):
        raise AssertionError(
            f"parity mismatch at {len(image)} bytes: rust={rust_sim} python={python_sim}"
        )
    return rust_sim


def _bench() -> list[dict]:
    rows = []
    for size in IMAGE_SIZES:
        image = _image(size)
        similarity = _assert_parity(image)
        rust = _time_round_trip(image, use_rust=True, iterations=ITERATIONS)
        python = _time_round_trip(image, use_rust=False, iterations=ITERATIONS)
        speedup = (
            round(python["median_us"] / rust["median_us"], 2)
            if rust["median_us"] > 0
            else None
        )
        rows.append(
            {
                "image_bytes": size,
                "similarity": round(similarity, 6),
                "parity": "rust == python",
                "rust": rust,
                "python": python,
                "speedup_rust_over_python": speedup,
            }
        )
    return rows


def _host_context() -> dict:
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:  # pragma: no cover - platform without loadavg
        load1 = load5 = load15 = -1.0
    return {
        "isolated": False,
        "isolation_method": "none",
        "evidence_class": "functional+parity+local-regression",
        "command": "python -m benchmarks.multimodal_guard",
        "cpu_count": os.cpu_count(),
        "loadavg_1_5_15": [round(load1, 2), round(load5, 2), round(load15, 2)],
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def main() -> None:
    print("Multimodal hash-bag guard benchmark")
    print("=" * 66)
    rows = _bench()
    print(
        f"\n{'Image bytes':>12} {'Python (us)':>14} {'Rust (us)':>12} {'Speedup':>10}"
    )
    print("-" * 66)
    for row in rows:
        speedup = row["speedup_rust_over_python"]
        speedup_str = f"{speedup:.2f}x" if speedup is not None else "N/A"
        print(
            f"{row['image_bytes']:>12} {row['python']['median_us']:>14.3f} "
            f"{row['rust']['median_us']:>12.3f} {speedup_str:>10}"
        )
    output = {
        "benchmark": "multimodal_guard",
        "host_context": _host_context(),
        "encode_verify_round_trip": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "multimodal_guard.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
