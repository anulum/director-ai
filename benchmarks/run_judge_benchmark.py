#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Local Judge E2E Benchmark (UpCloud L40S)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Compare NLI-only vs local judge on HaluEval at scale.

Runs:
  1. NLI-only baseline (no judge)
  2. NLI + local DeBERTa-base judge (hybrid scorer)

Usage on UpCloud:
    source /opt/director-bench/bin/activate
    cd /opt/director-bench/work/director-ai
    python benchmarks/run_judge_benchmark.py --samples 500 2>&1 | tee /tmp/judge_bench.log

    # Full 10K (expensive, ~3h on L40S):
    python benchmarks/run_judge_benchmark.py --samples 10000 2>&1 | tee /tmp/judge_bench_10k.log
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("JudgeBench")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
JUDGE_MODEL_PATH = str(REPO_ROOT / "training" / "output" / "deberta-v3-base-judge")


def _save(data: dict, name: str) -> None:
    path = RESULTS_DIR / name
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved %s (%d bytes)", path, path.stat().st_size)


def _gpu_info() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"gpu": "none", "cuda": False}
        return {
            "gpu": torch.cuda.get_device_name(0),
            "vram_gb": round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            ),
            "cuda": True,
            "torch": torch.__version__,
        }
    except ImportError:
        return {"gpu": "unavailable", "cuda": False}


def run_nli_only(max_samples: int) -> dict:
    """NLI-only baseline (no judge escalation)."""
    logger.info("=== NLI-only baseline (%d samples/task) ===", max_samples)
    from benchmarks.e2e_eval import print_e2e_results, run_e2e_benchmark

    t0 = time.time()
    m = run_e2e_benchmark(
        max_samples_per_task=max_samples,
        threshold=0.5,
        soft_limit=0.6,
        use_nli=True,
        scorer_backend="deberta",
    )
    elapsed = time.time() - t0
    print_e2e_results(m)
    result = {
        "benchmark": "E2E-NLI-Only",
        "samples_per_task": max_samples,
        "elapsed_s": round(elapsed, 1),
        **m.to_dict(),
        "hw": _gpu_info(),
    }
    _save(result, f"judge_bench_nli_only_{max_samples}.json")
    return result


def run_local_judge(max_samples: int) -> dict:
    """NLI + local DeBERTa-base judge (hybrid scorer)."""
    logger.info("=== NLI + local judge (%d samples/task) ===", max_samples)
    from benchmarks.e2e_eval import print_e2e_results, run_e2e_benchmark

    judge_path = Path(JUDGE_MODEL_PATH)
    if not judge_path.exists():
        logger.error("Judge model not found at %s", judge_path)
        raise FileNotFoundError(f"Judge model not found: {judge_path}")

    t0 = time.time()
    m = run_e2e_benchmark(
        max_samples_per_task=max_samples,
        threshold=0.5,
        soft_limit=0.6,
        use_nli=True,
        scorer_backend="hybrid",
        llm_judge_provider="local",
        llm_judge_model=str(judge_path),
    )
    elapsed = time.time() - t0
    print_e2e_results(m)
    result = {
        "benchmark": "E2E-Local-Judge",
        "samples_per_task": max_samples,
        "elapsed_s": round(elapsed, 1),
        **m.to_dict(),
        "hw": _gpu_info(),
    }
    _save(result, f"judge_bench_local_judge_{max_samples}.json")
    return result


def run_judge_latency(n_iters: int = 200) -> dict:
    """Measure pure judge inference latency (no NLI overhead)."""
    logger.info("=== Judge inference latency (%d iterations) ===", n_iters)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    judge_path = Path(JUDGE_MODEL_PATH)
    if not judge_path.exists():
        raise FileNotFoundError(f"Judge model not found: {judge_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(judge_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(judge_path), low_cpu_mem_usage=False
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    test_input = (
        "NLI divergence: 0.45\n"
        "Context: The Earth orbits the Sun at an average distance of 150 million km.\n"
        "Response: The Earth orbits the Sun at roughly 150 million kilometers."
    )
    inputs = tokenizer(test_input, return_tensors="pt", max_length=384, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(**inputs)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_iters):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(**inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    import numpy as np

    times_arr = np.array(times)
    result = {
        "benchmark": "Judge-Latency",
        "device": device,
        "n_iters": n_iters,
        "median_ms": round(float(np.median(times_arr)), 2),
        "mean_ms": round(float(np.mean(times_arr)), 2),
        "p5_ms": round(float(np.percentile(times_arr, 5)), 2),
        "p95_ms": round(float(np.percentile(times_arr, 95)), 2),
        "min_ms": round(float(np.min(times_arr)), 2),
        "max_ms": round(float(np.max(times_arr)), 2),
        "hw": _gpu_info(),
    }
    logger.info(
        "Judge latency on %s: median=%.1fms mean=%.1fms p95=%.1fms",
        device,
        result["median_ms"],
        result["mean_ms"],
        result["p95_ms"],
    )
    _save(result, "judge_bench_latency.json")
    return result


def print_comparison(nli_only: dict, local_judge: dict) -> None:
    """Print side-by-side comparison table."""
    print(f"\n{'=' * 76}")
    print("  Local Judge vs NLI-Only Comparison")
    print(f"{'=' * 76}")
    hdr = f"  {'Metric':<25} {'NLI-Only':>12} {'+ Local Judge':>14} {'Delta':>12}"
    print(hdr)
    print(f"  {'-' * 63}")

    metrics = [
        ("Catch rate", "catch_rate"),
        ("False positive rate", "false_positive_rate"),
        ("Precision", "precision"),
        ("F1", "f1"),
        ("Accuracy", "accuracy"),
    ]
    for label, key in metrics:
        nv = nli_only.get(key, 0)
        jv = local_judge.get(key, 0)
        delta = jv - nv
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<25} {nv:>11.1%} {jv:>13.1%} {sign}{delta:>10.1%}")

    print(f"\n  {'Latency avg (ms)':<25} {nli_only.get('avg_latency_ms', 0):>11.0f} "
          f"{local_judge.get('avg_latency_ms', 0):>13.0f}")
    print(f"  {'Runtime (s)':<25} {nli_only.get('elapsed_s', 0):>11.0f} "
          f"{local_judge.get('elapsed_s', 0):>13.0f}")

    # Per-task breakdown
    print(f"\n  {'Per-task F1:':<25}")
    for task in ["qa", "summarization", "dialogue"]:
        nf1 = nli_only.get("per_task", {}).get(task, {}).get("f1", 0)
        jf1 = local_judge.get("per_task", {}).get(task, {}).get("f1", 0)
        delta = jf1 - nf1
        sign = "+" if delta >= 0 else ""
        print(f"    {task:<23} {nf1:>11.1%} {jf1:>13.1%} {sign}{delta:>10.1%}")

    print(f"{'=' * 76}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local judge E2E benchmark")
    parser.add_argument(
        "--samples", type=int, default=500,
        help="Max samples per task (default: 500)",
    )
    parser.add_argument(
        "--latency-iters", type=int, default=200,
        help="Iterations for latency measurement (default: 200)",
    )
    parser.add_argument(
        "--skip-latency", action="store_true",
        help="Skip latency benchmark",
    )
    parser.add_argument(
        "--skip-nli-only", action="store_true",
        help="Skip NLI-only baseline (if already have results)",
    )
    args = parser.parse_args()

    logger.info("GPU: %s", json.dumps(_gpu_info()))
    logger.info("Samples per task: %d", args.samples)
    logger.info("Judge model: %s", JUDGE_MODEL_PATH)

    results = {}
    total_t0 = time.time()

    # 1. Latency benchmark
    if not args.skip_latency:
        try:
            results["latency"] = run_judge_latency(args.latency_iters)
        except Exception as e:
            logger.error("Latency benchmark failed: %s", e, exc_info=True)
            results["latency"] = {"status": "failed", "error": str(e)}

    # 2. NLI-only baseline
    nli_result = None
    if not args.skip_nli_only:
        try:
            nli_result = run_nli_only(args.samples)
            results["nli_only"] = {"status": "ok", "elapsed_s": nli_result["elapsed_s"]}
        except Exception as e:
            logger.error("NLI-only benchmark failed: %s", e, exc_info=True)
            results["nli_only"] = {"status": "failed", "error": str(e)}

    # 3. Local judge
    judge_result = None
    try:
        judge_result = run_local_judge(args.samples)
        results["local_judge"] = {"status": "ok", "elapsed_s": judge_result["elapsed_s"]}
    except Exception as e:
        logger.error("Local judge benchmark failed: %s", e, exc_info=True)
        results["local_judge"] = {"status": "failed", "error": str(e)}

    # 4. Comparison
    if nli_result and judge_result:
        print_comparison(nli_result, judge_result)

    total_elapsed = time.time() - total_t0
    results["total_elapsed_s"] = round(total_elapsed, 1)
    results["hw"] = _gpu_info()
    _save(results, f"judge_bench_summary_{args.samples}.json")

    logger.info("=== ALL DONE (%.0fs total) ===", total_elapsed)
    for name, info in results.items():
        if isinstance(info, dict) and "status" in info:
            logger.info("  %s: %s", name, info["status"])


if __name__ == "__main__":
    main()
