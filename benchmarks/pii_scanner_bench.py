# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Python vs Rust PII scanner benchmark

"""Measure the speedup of the Rust ``backfire_kernel.PiiScanner``
fast-path over the Python fallback in
:class:`director_ai.core.safety.moderation.pii.RegexPIIDetector`.

Run::

    python -m benchmarks.pii_scanner_bench

Prints a markdown table suitable for pasting into
``docs/BENCHMARKS.md``. Writes the raw run data to
``benchmarks/results/pii_scanner_bench.json`` for regression
tracking. If ``backfire_kernel`` is not installed, only the Python
column is filled in and the report flags the skip.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import random
import statistics
import time
from pathlib import Path

from director_ai.core.safety.moderation.pii import RegexPIIDetector

# ---------------------------------------------------------------------
# Synthetic corpora — the scanner's patterns target IDs that rarely
# occur in ordinary prose, so a realistic benchmark injects known
# fractions of hits.
# ---------------------------------------------------------------------

_PLAIN = (
    "The guardrail monitors every response against the configured "
    "knowledge base and halts generation when coherence drops below "
    "the hard limit. No personally identifying information surfaces "
    "in this paragraph and the scanner should walk it quickly. "
)


def _synthetic_block(hits: int, seed: int) -> str:
    rng = random.Random(seed)
    parts: list[str] = []
    for _ in range(hits):
        parts.extend(
            [
                _PLAIN,
                f" user email alice.{rng.randint(0, 10_000)}@example.com",
                _PLAIN,
                f" card 4111-1111-1111-{rng.randint(1000, 9999)}",
                _PLAIN,
                f" ssn 123-45-{rng.randint(1000, 9999)}",
                _PLAIN,
                f" server {rng.randint(1, 254)}.{rng.randint(1, 254)}"
                f".{rng.randint(1, 254)}.{rng.randint(1, 254)}",
            ]
        )
    return "".join(parts)


CORPORA: list[tuple[str, str]] = [
    ("clean-1kb", _PLAIN * 5),
    ("clean-10kb", _PLAIN * 50),
    ("mixed-1kb", _synthetic_block(hits=2, seed=1)),
    ("mixed-10kb", _synthetic_block(hits=20, seed=2)),
]


# ---------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------


def _bench(detector: RegexPIIDetector, text: str, *, rounds: int) -> list[float]:
    """Return per-round wall times in milliseconds. Forces a GC
    between rounds so allocation noise does not contaminate the
    measurement."""
    times: list[float] = []
    for _ in range(rounds):
        gc.collect()
        start = time.perf_counter_ns()
        detector.analyse(text)
        times.append((time.perf_counter_ns() - start) / 1_000_000)
    return times


def _report(records: list[dict[str, float | str]]) -> str:
    lines = [
        "| corpus | size (B) | Python ms/call (median) | Rust ms/call (median) | "
        "speedup (×) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in records:
        py = r["python_median_ms"]
        rs = r.get("rust_median_ms")
        if isinstance(rs, (int, float)) and rs > 0:
            speedup = f"{(float(py) / float(rs)):.2f}"
        else:
            speedup = "—"
        lines.append(
            f"| `{r['corpus']}` | {int(r['size_bytes'])} | "
            f"{float(py):.3f} | "
            f"{('—' if rs is None else f'{float(rs):.3f}')} | "
            f"{speedup} |"
        )
    return "\n".join(lines)


def run(rounds: int = 500) -> list[dict[str, float | str]]:
    py_detector = RegexPIIDetector(prefer_rust=False)
    has_rust = importlib.util.find_spec("backfire_kernel") is not None
    rust_detector = RegexPIIDetector(prefer_rust=True) if has_rust else None

    if rust_detector is not None and rust_detector.backend != "rust":
        # Backend reports Python — the Rust wheel imported but the
        # scanner failed to build. Surface this rather than silently
        # faking the numbers.
        rust_detector = None

    records: list[dict[str, float | str]] = []
    for name, text in CORPORA:
        # Warm-up: 20 rounds to evict first-call caches.
        _bench(py_detector, text, rounds=20)
        if rust_detector is not None:
            _bench(rust_detector, text, rounds=20)

        py_times = _bench(py_detector, text, rounds=rounds)
        rec: dict[str, float | str] = {
            "corpus": name,
            "size_bytes": len(text.encode("utf-8")),
            "rounds": rounds,
            "python_median_ms": statistics.median(py_times),
            "python_p95_ms": sorted(py_times)[math.floor(0.95 * len(py_times))],
        }
        if rust_detector is not None:
            rust_times = _bench(rust_detector, text, rounds=rounds)
            rec["rust_median_ms"] = statistics.median(rust_times)
            rec["rust_p95_ms"] = sorted(rust_times)[math.floor(0.95 * len(rust_times))]
        records.append(rec)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pii_scanner_bench")
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "pii_scanner_bench.json",
    )
    args = parser.parse_args(argv)
    records = run(rounds=args.rounds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")

    has_rust = importlib.util.find_spec("backfire_kernel") is not None
    if not has_rust:
        print("# note: backfire_kernel not installed — Python-only numbers")
    print(_report(records))
    print(f"\nraw → {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
