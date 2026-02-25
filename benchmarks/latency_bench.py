# ---------------------------------------------------------------------
# Director-Class AI — End-to-End Latency Benchmark
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ---------------------------------------------------------------------
"""
Measure wall-clock latency of the full Director-AI pipeline.

Benchmarks:
  1. review()        — GroundTruthStore lookup + NLI + dual-entropy scoring
  2. NLI only        — raw NLI model forward pass
  3. Lightweight      — review() with use_nli=False (embedding-only)
  4. Streaming token  — per-token cost in StreamingKernel

Usage:
    python -m benchmarks.latency_bench              # lightweight (no NLI model)
    python -m benchmarks.latency_bench --nli         # full NLI pipeline
    python -m benchmarks.latency_bench --nli --warmup 5 --iterations 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class LatencyResult:
    name: str
    times_ms: list[float] = field(default_factory=list, repr=False)

    @property
    def mean(self) -> float:
        return float(np.mean(self.times_ms)) if self.times_ms else 0.0

    @property
    def median(self) -> float:
        return float(np.median(self.times_ms)) if self.times_ms else 0.0

    @property
    def p95(self) -> float:
        return float(np.percentile(self.times_ms, 95)) if self.times_ms else 0.0

    @property
    def p99(self) -> float:
        return float(np.percentile(self.times_ms, 99)) if self.times_ms else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.times_ms)) if self.times_ms else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n": len(self.times_ms),
            "mean_ms": round(self.mean, 2),
            "median_ms": round(self.median, 2),
            "p95_ms": round(self.p95, 2),
            "p99_ms": round(self.p99, 2),
            "std_ms": round(self.std, 2),
        }


FACTS = {
    "capital_france": "Paris is the capital of France.",
    "capital_germany": "Berlin is the capital of Germany.",
    "sky_color": "The sky appears blue due to Rayleigh scattering.",
    "water_formula": "Water has the chemical formula H2O.",
    "earth_sun": "The Earth orbits the Sun at a mean distance of 149.6 million km.",
}

QUERIES = [
    ("What is the capital of France?", "Paris is the capital of France."),
    ("What is the capital of France?", "London is the capital of France."),
    ("What color is the sky?", "The sky is blue because of Rayleigh scattering."),
    ("What is water made of?", "Water is composed of hydrogen and oxygen atoms."),
    ("How far is the Earth from the Sun?", "About 150 million kilometers."),
]


def bench_lightweight(iterations: int, warmup: int) -> LatencyResult:
    from director_ai.core import CoherenceScorer, GroundTruthStore

    store = GroundTruthStore()
    for k, v in FACTS.items():
        store.add(k, v)
    scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store, use_nli=False)

    for _ in range(warmup):
        scorer.review(QUERIES[0][0], QUERIES[0][1])

    result = LatencyResult("review (no NLI)")
    for i in range(iterations):
        q, r = QUERIES[i % len(QUERIES)]
        t0 = time.perf_counter()
        scorer.review(q, r)
        result.times_ms.append((time.perf_counter() - t0) * 1000)
    return result


def bench_nli_only(iterations: int, warmup: int) -> LatencyResult:
    from director_ai.core.nli import NLIScorer

    nli = NLIScorer(use_model=True)

    for _ in range(warmup):
        nli.score("The sky is blue.", "The sky is blue.")

    result = LatencyResult("NLI forward pass")
    for i in range(iterations):
        q, r = QUERIES[i % len(QUERIES)]
        t0 = time.perf_counter()
        nli.score(q, r)
        result.times_ms.append((time.perf_counter() - t0) * 1000)
    return result


def bench_full_pipeline(iterations: int, warmup: int) -> LatencyResult:
    from director_ai.core import CoherenceScorer, GroundTruthStore

    store = GroundTruthStore()
    for k, v in FACTS.items():
        store.add(k, v)
    scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store, use_nli=True)

    for _ in range(warmup):
        scorer.review(QUERIES[0][0], QUERIES[0][1])

    result = LatencyResult("review (full NLI)")
    for i in range(iterations):
        q, r = QUERIES[i % len(QUERIES)]
        t0 = time.perf_counter()
        scorer.review(q, r)
        result.times_ms.append((time.perf_counter() - t0) * 1000)
    return result


def bench_streaming(iterations: int, warmup: int) -> LatencyResult:
    from director_ai.core import StreamingKernel

    good_scores = [0.9, 0.85, 0.88, 0.92, 0.87, 0.90, 0.86, 0.91, 0.89]
    bad_scores = [0.8, 0.7, 0.6, 0.55, 0.45, 0.35, 0.3, 0.25, 0.2, 0.1]
    tokens = ["The", " sky", " is", " blue", " due",
              " to", " Rayleigh", " scattering", "."]

    def make_callback(scores):
        idx = [0]
        def cb(_token):
            s = scores[idx[0] % len(scores)]
            idx[0] += 1
            return s
        return cb

    sk = StreamingKernel()
    for _ in range(warmup):
        sk.stream_tokens(iter(tokens), make_callback(good_scores))

    result = LatencyResult("streaming session")
    for i in range(iterations):
        scores = good_scores if i % 2 == 0 else bad_scores
        sk = StreamingKernel()
        t0 = time.perf_counter()
        sk.stream_tokens(iter(tokens), make_callback(scores))
        result.times_ms.append((time.perf_counter() - t0) * 1000)
    return result


def print_result(r: LatencyResult) -> None:
    print(f"  {r.name:25s}  mean={r.mean:7.2f} ms  median={r.median:7.2f} ms  "
          f"p95={r.p95:7.2f} ms  p99={r.p99:7.2f} ms  (n={len(r.times_ms)})")


def main():
    parser = argparse.ArgumentParser(description="Director-AI latency benchmark")
    parser.add_argument(
        "--nli", action="store_true",
        help="Include NLI model benchmarks (requires torch)",
    )
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="Iterations per benchmark (default: 100)",
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Warmup iterations (default: 10)",
    )
    args = parser.parse_args()

    print("\nDirector-AI Latency Benchmark")
    print(f"  iterations={args.iterations}  warmup={args.warmup}  nli={args.nli}")
    print(f"{'-' * 80}")

    results = []

    r = bench_lightweight(args.iterations, args.warmup)
    print_result(r)
    results.append(r)

    r = bench_streaming(args.iterations, args.warmup)
    print_result(r)
    results.append(r)

    if args.nli:
        r = bench_nli_only(args.iterations, args.warmup)
        print_result(r)
        results.append(r)

        r = bench_full_pipeline(args.iterations, args.warmup)
        print_result(r)
        results.append(r)

    print(f"{'-' * 80}")

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "nli": args.nli,
        "python": sys.version.split()[0],
        "results": [r.to_dict() for r in results],
    }
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "latency.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
