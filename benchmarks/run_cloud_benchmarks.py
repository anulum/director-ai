#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Cloud Benchmark Runner (UpCloud L40S)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Run all GPU-dependent benchmarks on UpCloud L40S and collect results.

Benchmarks:
  1. RAGTruth (NLI, full dataset)
  2. FreshQA (NLI, full dataset)
  3. E2E hybrid (NLI + Claude judge, HaluEval 300 traces)
  4. E2E hybrid (NLI + GPT-4o-mini judge, HaluEval 300 traces)
  5. AggreFact sweep (NLI, full 29K samples — re-run for L40S latency)

Upload this file to the UpCloud server and run:

    source /opt/director-bench/bin/activate
    cd /opt/director-bench/work/director-ai
    python benchmarks/run_cloud_benchmarks.py 2>&1 | tee /tmp/bench.log

Results are saved to benchmarks/results/*.json and printed to stdout.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("CloudBench")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _save(data: dict, name: str) -> None:
    path = RESULTS_DIR / name
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved %s (%d bytes)", path, path.stat().st_size)


def _gpu_info() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"gpu": "none", "cuda": False}
    return {
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
        "cuda": True,
        "torch": torch.__version__,
    }


def bench_ragtruth() -> None:
    logger.info("=== RAGTruth (NLI, full dataset) ===")
    from benchmarks.e2e_eval import print_e2e_results
    from benchmarks.ragtruth_eval import run_ragtruth

    t0 = time.time()
    m = run_ragtruth(use_nli=True)
    elapsed = time.time() - t0
    print_e2e_results(m)
    result = {
        "benchmark": "RAGTruth-NLI",
        "elapsed_s": round(elapsed, 1),
        **m.to_dict(),
    }
    result["hw"] = _gpu_info()
    _save(result, "ragtruth_nli_results.json")


def bench_freshqa() -> None:
    logger.info("=== FreshQA (NLI, full dataset) ===")
    from benchmarks.e2e_eval import print_e2e_results
    from benchmarks.freshqa_eval import run_freshqa

    t0 = time.time()
    m = run_freshqa(use_nli=True)
    elapsed = time.time() - t0
    print_e2e_results(m)
    result = {"benchmark": "FreshQA-NLI", "elapsed_s": round(elapsed, 1), **m.to_dict()}
    result["hw"] = _gpu_info()
    _save(result, "freshqa_nli_results.json")


def bench_hybrid_claude() -> None:
    logger.info("=== E2E Hybrid: NLI + Claude judge (300 traces) ===")
    from benchmarks.e2e_eval import print_e2e_results, run_e2e_benchmark

    t0 = time.time()
    m = run_e2e_benchmark(
        max_samples_per_task=100,
        threshold=0.35,
        soft_limit=0.45,
        use_nli=True,
        scorer_backend="hybrid",
        llm_judge_provider="anthropic",
        llm_judge_model="claude-sonnet-4-20250514",
    )
    elapsed = time.time() - t0
    print_e2e_results(m)
    result = {
        "benchmark": "E2E-Hybrid-Claude",
        "elapsed_s": round(elapsed, 1),
        **m.to_dict(),
    }
    result["hw"] = _gpu_info()
    _save(result, "e2e_hybrid_claude_results.json")


def bench_hybrid_openai() -> None:
    logger.info("=== E2E Hybrid: NLI + GPT-4o-mini judge (300 traces) ===")
    from benchmarks.e2e_eval import print_e2e_results, run_e2e_benchmark

    t0 = time.time()
    m = run_e2e_benchmark(
        max_samples_per_task=100,
        threshold=0.35,
        soft_limit=0.45,
        use_nli=True,
        scorer_backend="hybrid",
        llm_judge_provider="openai",
        llm_judge_model="gpt-4o-mini",
    )
    elapsed = time.time() - t0
    print_e2e_results(m)
    result = {
        "benchmark": "E2E-Hybrid-OpenAI",
        "elapsed_s": round(elapsed, 1),
        **m.to_dict(),
    }
    result["hw"] = _gpu_info()
    _save(result, "e2e_hybrid_openai_results.json")


def bench_aggrefact_l40s() -> None:
    logger.info("=== AggreFact sweep (29K samples, L40S latency) ===")
    from benchmarks.aggrefact_eval import run_aggrefact

    t0 = time.time()
    result = run_aggrefact(sweep=True)
    elapsed = time.time() - t0
    result["elapsed_s"] = round(elapsed, 1)
    result["hw"] = _gpu_info()
    _save(result, "aggrefact_l40s_results.json")


def main() -> None:
    logger.info("GPU info: %s", json.dumps(_gpu_info()))
    logger.info("ANTHROPIC_API_KEY set: %s", bool(os.environ.get("ANTHROPIC_API_KEY")))
    logger.info("OPENAI_API_KEY set: %s", bool(os.environ.get("OPENAI_API_KEY")))

    benchmarks = [
        ("RAGTruth", bench_ragtruth),
        ("FreshQA", bench_freshqa),
        ("Hybrid-Claude", bench_hybrid_claude),
        ("Hybrid-OpenAI", bench_hybrid_openai),
        ("AggreFact-L40S", bench_aggrefact_l40s),
    ]

    results_summary = {}
    for name, fn in benchmarks:
        try:
            t0 = time.time()
            fn()
            results_summary[name] = {
                "status": "ok",
                "elapsed_s": round(time.time() - t0, 1),
            }
        except Exception as e:
            logger.error("FAILED %s: %s", name, e, exc_info=True)
            results_summary[name] = {"status": "failed", "error": str(e)}

    _save(results_summary, "cloud_benchmark_summary.json")
    logger.info("=== DONE ===")
    for name, info in results_summary.items():
        logger.info("  %s: %s", name, info["status"])


if __name__ == "__main__":
    main()
