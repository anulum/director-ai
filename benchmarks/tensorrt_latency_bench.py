# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — TensorRT Latency Benchmark

"""Measures per-pair NLI inference latency across backends:

  - PyTorch (FP32 / FP16)
  - ONNX Runtime (CPU / CUDA)
  - ONNX + TensorRT EP (FP16)

Usage::

    python -m benchmarks.tensorrt_latency_bench \
        --onnx-dir factcg_onnx \
        --warmup 20 --trials 200

Requires GPU with CUDA. For TRT, pre-build with::

    director-ai export --format tensorrt --onnx-dir factcg_onnx
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

PAIRS = [
    (
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
        "in Paris. It was built in 1889 as the entrance arch for the 1889 "
        "World's Fair.",
        "The Eiffel Tower was constructed in 1889.",
    ),
    (
        "Water consists of hydrogen and oxygen atoms. It boils at 100 degrees "
        "Celsius at standard atmospheric pressure.",
        "Water boils at 50 degrees Celsius.",
    ),
    (
        "The speed of light in vacuum is approximately 299,792,458 metres per "
        "second. Einstein's special relativity postulates this as constant.",
        "Light travels at roughly 300,000 km/s.",
    ),
    (
        "Mount Everest is Earth's highest mountain above sea level, located in "
        "the Mahalangur Himal sub-range of the Himalayas.",
        "Mount Everest is in the Andes mountain range.",
    ),
]


def _bench_pytorch(device: str, warmup: int, trials: int) -> dict:
    from director_ai.core.nli import NLIScorer

    scorer = NLIScorer(backend="deberta", device=device)
    if not scorer.model_available:
        return {"backend": f"pytorch-{device}", "error": "model unavailable"}

    for _ in range(warmup):
        scorer.score_batch(PAIRS)

    latencies = []
    for _ in range(trials):
        t0 = time.perf_counter()
        scorer.score_batch(PAIRS)
        latencies.append((time.perf_counter() - t0) / len(PAIRS) * 1000)

    return _stats(f"pytorch-{device}", latencies)


def _bench_onnx(onnx_dir: str, device: str, warmup: int, trials: int) -> dict:
    if not os.path.isdir(onnx_dir):
        return {"backend": f"onnx-{device}", "error": f"not found: {onnx_dir}"}

    os.environ.pop("DIRECTOR_ENABLE_TRT", None)
    from director_ai.core.nli import NLIScorer

    scorer = NLIScorer(backend="onnx", onnx_path=onnx_dir, device=device)
    if not scorer.model_available:
        return {"backend": f"onnx-{device}", "error": "session unavailable"}

    for _ in range(warmup):
        scorer.score_batch(PAIRS)

    latencies = []
    for _ in range(trials):
        t0 = time.perf_counter()
        scorer.score_batch(PAIRS)
        latencies.append((time.perf_counter() - t0) / len(PAIRS) * 1000)

    return _stats(f"onnx-{device}", latencies)


def _bench_trt(onnx_dir: str, warmup: int, trials: int) -> dict:
    trt_cache = os.path.join(onnx_dir, "trt_cache")
    if not os.path.isdir(trt_cache):
        return {"backend": "onnx-trt", "error": f"no TRT cache at {trt_cache}"}

    os.environ["DIRECTOR_ENABLE_TRT"] = "1"
    from director_ai.core.nli import NLIScorer

    scorer = NLIScorer(backend="onnx", onnx_path=onnx_dir, device="cuda")
    if not scorer.model_available:
        return {"backend": "onnx-trt", "error": "TRT session unavailable"}

    for _ in range(warmup):
        scorer.score_batch(PAIRS)

    latencies = []
    for _ in range(trials):
        t0 = time.perf_counter()
        scorer.score_batch(PAIRS)
        latencies.append((time.perf_counter() - t0) / len(PAIRS) * 1000)

    os.environ.pop("DIRECTOR_ENABLE_TRT", None)
    return _stats("onnx-trt-fp16", latencies)


def _stats(backend: str, latencies_ms: list[float]) -> dict:
    latencies_ms.sort()
    return {
        "backend": backend,
        "trials": len(latencies_ms),
        "pairs_per_trial": len(PAIRS),
        "ms_per_pair_mean": round(statistics.mean(latencies_ms), 3),
        "ms_per_pair_median": round(statistics.median(latencies_ms), 3),
        "ms_per_pair_p95": round(latencies_ms[int(len(latencies_ms) * 0.95)], 3),
        "ms_per_pair_p99": round(latencies_ms[int(len(latencies_ms) * 0.99)], 3),
        "ms_per_pair_min": round(latencies_ms[0], 3),
        "ms_per_pair_max": round(latencies_ms[-1], 3),
    }


def main():
    parser = argparse.ArgumentParser(description="TensorRT latency benchmark")
    parser.add_argument("--onnx-dir", default="factcg_onnx")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--backends",
        default="pytorch-cuda,onnx-cuda,trt",
        help="comma-separated: pytorch-cuda,pytorch-cpu,onnx-cpu,onnx-cuda,trt",
    )
    args = parser.parse_args()

    backends = args.backends.split(",")
    results = []

    for b in backends:
        b = b.strip()
        print(f"Benchmarking {b}...", flush=True)
        if b == "pytorch-cuda":
            results.append(_bench_pytorch("cuda", args.warmup, args.trials))
        elif b == "pytorch-cpu":
            results.append(_bench_pytorch("cpu", args.warmup, args.trials))
        elif b == "onnx-cpu":
            results.append(_bench_onnx(args.onnx_dir, "cpu", args.warmup, args.trials))
        elif b == "onnx-cuda":
            results.append(_bench_onnx(args.onnx_dir, "cuda", args.warmup, args.trials))
        elif b == "trt":
            results.append(_bench_trt(args.onnx_dir, args.warmup, args.trials))
        else:
            print(f"  Unknown backend: {b}")

    print("\n" + "=" * 72)
    print(f"{'Backend':<20} {'Mean':>8} {'Median':>8} {'P95':>8} {'P99':>8} ms/pair")
    print("-" * 72)
    for r in results:
        if "error" in r:
            print(f"{r['backend']:<20} ERROR: {r['error']}")
        else:
            print(
                f"{r['backend']:<20} "
                f"{r['ms_per_pair_mean']:>8.3f} "
                f"{r['ms_per_pair_median']:>8.3f} "
                f"{r['ms_per_pair_p95']:>8.3f} "
                f"{r['ms_per_pair_p99']:>8.3f}",
            )
    print("=" * 72)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
