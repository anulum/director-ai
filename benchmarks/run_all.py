# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Run All Benchmarks & Generate Comparison Table
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Run the complete held-out benchmark suite and produce a comparison table.

Runs each model against: MNLI, ANLI R1/R2/R3, FEVER dev, VitaminC dev,
PAWS, and false-positive rate on clean RAG data.

Usage::

    # Compare baseline vs fine-tuned
    python -m benchmarks.run_all \\
        --models baseline=MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli \\
                 finetuned=training/output/deberta-v3-base-hallucination \\
        --max-samples 500

    # Single model, all benchmarks
    python -m benchmarks.run_all --max-samples 300
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

from benchmarks._common import RESULTS_DIR, save_results

logger = logging.getLogger("DirectorAI.Benchmark.RunAll")


def _run_suite(model_name: str | None, max_samples: int | None) -> dict:
    """Run all benchmarks for a single model. Returns results dict."""
    from benchmarks.anli_eval import run_anli_benchmark
    from benchmarks.falsepositive_eval import run_falsepositive_benchmark
    from benchmarks.fever_eval import run_fever_benchmark
    from benchmarks.mnli_eval import run_mnli_benchmark
    from benchmarks.paws_eval import run_paws_benchmark
    from benchmarks.vitaminc_eval import run_vitaminc_benchmark

    results = {"model": model_name or "default"}
    t_total = time.time()

    # MNLI matched + mismatched
    for split in ["validation_matched", "validation_mismatched"]:
        label = split.replace("validation_", "mnli_")
        logger.info("=== %s ===", label)
        m = run_mnli_benchmark(split, max_samples=max_samples, model_name=model_name)
        results[label] = m.to_dict()

    # ANLI R1, R2, R3
    for rnd in ["r1", "r2", "r3"]:
        label = f"anli_{rnd}"
        logger.info("=== %s ===", label)
        m = run_anli_benchmark(rnd, "test", max_samples=max_samples, model_name=model_name)
        results[label] = m.to_dict()

    # FEVER dev
    logger.info("=== fever_dev ===")
    m = run_fever_benchmark(max_samples=max_samples, model_name=model_name)
    results["fever_dev"] = m.to_dict()

    # VitaminC dev
    logger.info("=== vitaminc_dev ===")
    m = run_vitaminc_benchmark("validation", max_samples=max_samples, model_name=model_name)
    results["vitaminc_dev"] = m.to_dict()

    # PAWS
    logger.info("=== paws ===")
    m = run_paws_benchmark(max_samples=max_samples, model_name=model_name)
    results["paws"] = m.to_dict()

    # False-positive rate
    logger.info("=== false_positive ===")
    fp = run_falsepositive_benchmark(max_samples=max_samples, model_name=model_name)
    results["false_positive"] = fp.to_dict()

    # LLM-AggreFact (gated — skip if no HF_TOKEN)
    if os.environ.get("HF_TOKEN"):
        from benchmarks.aggrefact_eval import run_aggrefact_benchmark
        logger.info("=== aggrefact ===")
        af = run_aggrefact_benchmark(max_samples=max_samples, model_name=model_name)
        results["aggrefact"] = af.to_dict()
    else:
        logger.info("=== aggrefact === SKIPPED (no HF_TOKEN)")

    results["total_time_seconds"] = round(time.time() - t_total, 1)
    return results


def _print_comparison_table(all_results: dict[str, dict]) -> None:
    """Print markdown comparison table across models."""
    benchmarks = [
        ("mnli_matched", "accuracy", "MNLI Matched Acc"),
        ("mnli_mismatched", "accuracy", "MNLI Mismatched Acc"),
        ("anli_r1", "accuracy", "ANLI R1 Acc"),
        ("anli_r2", "accuracy", "ANLI R2 Acc"),
        ("anli_r3", "accuracy", "ANLI R3 Acc"),
        ("fever_dev", "macro_f1", "FEVER Dev F1"),
        ("vitaminc_dev", "macro_f1", "VitaminC Dev F1"),
        ("paws", "accuracy", "PAWS Acc"),
        ("false_positive", "false_positive_rate", "FP Rate (lower=better)"),
        ("aggrefact", "avg_balanced_accuracy_pct", "AggreFact Bal Acc %"),
    ]

    models = list(all_results.keys())
    header = f"| {'Benchmark':<25} |" + "".join(f" {m:>15} |" for m in models)
    sep = f"|{'-' * 27}|" + "".join(f"{'-' * 17}|" for _ in models)

    print(f"\n{'=' * (28 + 18 * len(models))}")
    print("  Benchmark Comparison Table")
    print(f"{'=' * (28 + 18 * len(models))}")
    print(header)
    print(sep)

    for bm_key, metric_key, display_name in benchmarks:
        row = f"| {display_name:<25} |"
        for model in models:
            result = all_results[model].get(bm_key, {})
            val = result.get(metric_key)
            if val is not None:
                if metric_key == "false_positive_rate":
                    row += f" {val:>14.2%} |"
                else:
                    row += f" {val:>14.4f} |"
            else:
                row += f" {'—':>15} |"
        print(row)

    print(f"{'=' * (28 + 18 * len(models))}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run all NLI benchmarks")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--models", nargs="+", metavar="NAME=PATH",
                        help="Model specs as name=path pairs (e.g., baseline=model_id finetuned=./path)")
    args = parser.parse_args()

    if args.models:
        model_specs = {}
        for spec in args.models:
            if "=" in spec:
                name, path = spec.split("=", 1)
                model_specs[name] = path
            else:
                model_specs[spec] = spec
    else:
        model_specs = {"default": None}

    all_results = {}
    for name, path in model_specs.items():
        logger.info("\n\n########## Running benchmarks for: %s ##########\n", name)
        all_results[name] = _run_suite(path, args.max_samples)

    _print_comparison_table(all_results)

    save_results(
        {"benchmark_suite": "all", "models": all_results},
        "benchmark_comparison.json",
    )
