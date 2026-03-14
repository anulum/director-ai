# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Per-Dataset Threshold Analysis
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Per-dataset optimal threshold sweep for LLM-AggreFact.

The FactCG paper (NAACL 2025) reports 77.2% balanced accuracy — 1.34pp
above our measured 75.86% with a global threshold. The gap likely comes
from per-dataset optimal thresholds macro-averaged, vs our single
global threshold.

Modes:
  --per-dataset     Sweep threshold independently per dataset, report
                    macro-average of per-dataset-optimal BAs.
  --global          Standard global sweep (same as aggrefact_eval --sweep).
  --task-type       Group datasets by task type, sweep per group.

Usage::

    python -m benchmarks.threshold_analysis --per-dataset
    python -m benchmarks.threshold_analysis --task-type
    python -m benchmarks.threshold_analysis --global --per-dataset --compare
"""

from __future__ import annotations

import logging
import time

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from benchmarks._common import save_results
from benchmarks.aggrefact_eval import (
    REFERENCE_SCORES,
    _binary_class_metrics,
    _BinaryNLIPredictor,
    _load_aggrefact,
)

logger = logging.getLogger("DirectorAI.Benchmark.ThresholdAnalysis")

# Task-type grouping for adaptive thresholding
TASK_TYPE_MAP = {
    "AggreFact-CNN": "summarization",
    "AggreFact-XSum": "summarization",
    "TofuEval-MediaS": "summarization",
    "TofuEval-MeetB": "summarization",
    "Wice": "fact_check",
    "Reveal": "fact_check",
    "ClaimVerify": "fact_check",
    "FactCheck-GPT": "fact_check",
    "ExpertQA": "qa",
    "Lfqa": "qa",
    "RAGTruth": "rag",
}


def _score_all(
    model_name: str | None = None,
    max_samples: int | None = None,
) -> dict[str, list[tuple[int, float]]]:
    """Score all samples once, return {dataset: [(label, ent_prob), ...]}."""
    predictor = _BinaryNLIPredictor(model_name=model_name)
    rows = _load_aggrefact(max_samples)

    by_dataset: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        doc = row.get("doc", "")
        claim = row.get("claim", "")
        label = row.get("label")
        ds_name = row.get("dataset", "unknown")
        if label is None or not doc or not claim:
            continue
        ent_prob = predictor.score(doc, claim)
        if ds_name not in by_dataset:
            by_dataset[ds_name] = []
        by_dataset[ds_name].append((int(label), ent_prob))
    return by_dataset


def sweep_per_dataset(
    by_dataset: dict[str, list[tuple[int, float]]],
) -> dict[str, dict]:
    """Find optimal threshold per dataset independently."""
    results: dict[str, dict] = {}
    for ds_name in sorted(by_dataset.keys()):
        pairs = by_dataset[ds_name]
        y_true = [p[0] for p in pairs]
        y_scores = [p[1] for p in pairs]

        best_thresh, best_ba = 0.5, 0.0
        for t_int in range(10, 91):
            t = t_int / 100.0
            y_pred = [1 if s >= t else 0 for s in y_scores]
            ba = balanced_accuracy_score(y_true, y_pred)
            if ba > best_ba:
                best_ba = ba
                best_thresh = t

        y_pred_opt = [1 if s >= best_thresh else 0 for s in y_scores]
        results[ds_name] = {
            "total": len(pairs),
            "positive": sum(y_true),
            "negative": len(y_true) - sum(y_true),
            "optimal_threshold": best_thresh,
            "balanced_acc": float(best_ba),
            **_binary_class_metrics(y_true, y_pred_opt),
        }
    return results


def sweep_global(
    by_dataset: dict[str, list[tuple[int, float]]],
) -> tuple[float, dict[str, dict]]:
    """Find single global threshold maximising macro-avg BA."""
    best_thresh, best_avg = 0.5, 0.0
    for t_int in range(10, 91):
        t = t_int / 100.0
        accs = []
        for pairs in by_dataset.values():
            y_true = [p[0] for p in pairs]
            y_pred = [1 if p[1] >= t else 0 for p in pairs]
            accs.append(balanced_accuracy_score(y_true, y_pred))
        avg = float(np.mean(accs))
        if avg > best_avg:
            best_avg = avg
            best_thresh = t

    results: dict[str, dict] = {}
    for ds_name in sorted(by_dataset.keys()):
        pairs = by_dataset[ds_name]
        y_true = [p[0] for p in pairs]
        y_pred = [1 if p[1] >= best_thresh else 0 for p in pairs]
        results[ds_name] = {
            "total": len(pairs),
            "positive": sum(y_true),
            "negative": len(y_true) - sum(y_true),
            "optimal_threshold": best_thresh,
            "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
            **_binary_class_metrics(y_true, y_pred),
        }
    return best_thresh, results


def sweep_per_task_type(
    by_dataset: dict[str, list[tuple[int, float]]],
) -> dict[str, dict]:
    """Group datasets by task type, find optimal threshold per group."""
    by_type: dict[str, list[tuple[int, float]]] = {}
    ds_to_type: dict[str, str] = {}
    for ds_name, pairs in by_dataset.items():
        task_type = TASK_TYPE_MAP.get(ds_name, "other")
        ds_to_type[ds_name] = task_type
        if task_type not in by_type:
            by_type[task_type] = []
        by_type[task_type].extend(pairs)

    type_thresholds: dict[str, float] = {}
    for task_type, pairs in by_type.items():
        y_true = [p[0] for p in pairs]
        y_scores = [p[1] for p in pairs]
        best_thresh, best_ba = 0.5, 0.0
        for t_int in range(10, 91):
            t = t_int / 100.0
            y_pred = [1 if s >= t else 0 for s in y_scores]
            ba = balanced_accuracy_score(y_true, y_pred)
            if ba > best_ba:
                best_ba = ba
                best_thresh = t
        type_thresholds[task_type] = best_thresh

    results: dict[str, dict] = {}
    for ds_name in sorted(by_dataset.keys()):
        pairs = by_dataset[ds_name]
        task_type = ds_to_type[ds_name]
        t = type_thresholds[task_type]
        y_true = [p[0] for p in pairs]
        y_pred = [1 if p[1] >= t else 0 for p in pairs]
        results[ds_name] = {
            "total": len(pairs),
            "positive": sum(y_true),
            "negative": len(y_true) - sum(y_true),
            "task_type": task_type,
            "optimal_threshold": t,
            "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
            **_binary_class_metrics(y_true, y_pred),
        }
    return results


def _print_results(
    label: str,
    results: dict[str, dict],
    global_thresh: float | None = None,
) -> None:
    accs = [d["balanced_acc"] for d in results.values() if d["total"] > 0]
    avg_ba = float(np.mean(accs)) if accs else 0.0
    total = sum(d["total"] for d in results.values())

    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"{'=' * 72}")
    if global_thresh is not None:
        print(f"  Global threshold: {global_thresh:.2f}")
    print(f"  Samples: {total}")
    print(f"  Macro-avg BA: {avg_ba:.1%}")
    print()

    hdr = f"  {'Dataset':<20} {'N':>5} {'Thresh':>6} {'BalAcc':>7}"
    if any("task_type" in d for d in results.values()):
        hdr += f" {'TaskType':>14}"
    print(hdr)
    print(f"  {'-' * len(hdr.strip())}")
    for ds_name, d in sorted(results.items()):
        line = (
            f"  {ds_name:<20} {d['total']:>5}"
            f" {d['optimal_threshold']:>5.2f}"
            f" {d['balanced_acc']:>6.1%}"
        )
        if "task_type" in d:
            line += f" {d['task_type']:>14}"
        print(line)
    print()

    our_pct = avg_ba * 100
    print(f"  {'Model':<30} {'Bal Acc':>8}  {'vs Ours':>8}")
    print(f"  {'-' * 50}")
    inserted = False
    for ref_name, ref_score in sorted(
        REFERENCE_SCORES.items(),
        key=lambda x: -x[1],
    ):
        if not inserted and our_pct >= ref_score:
            print(f"  {'>>> OURS <<<':<30} {our_pct:>7.1f}%")
            inserted = True
        print(
            f"  {ref_name:<30} {ref_score:>7.1f}%  {our_pct - ref_score:>+7.1f}",
        )
    if not inserted:
        print(f"  {'>>> OURS <<<':<30} {our_pct:>7.1f}%")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Per-dataset threshold analysis for LLM-AggreFact",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--per-dataset",
        action="store_true",
        help="Per-dataset optimal thresholds",
    )
    parser.add_argument(
        "--global",
        dest="do_global",
        action="store_true",
        help="Global threshold sweep",
    )
    parser.add_argument(
        "--task-type",
        action="store_true",
        help="Per-task-type thresholds",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print comparison table",
    )
    args = parser.parse_args()

    if not (args.per_dataset or args.do_global or args.task_type):
        args.per_dataset = True
        args.do_global = True
        args.task_type = True
        args.compare = True

    t0 = time.time()
    logger.info("Scoring all samples (one inference pass)...")
    by_dataset = _score_all(model_name=args.model, max_samples=args.max_samples)
    logger.info("Scoring complete in %.1f s", time.time() - t0)

    all_results = {}

    if args.do_global:
        g_thresh, g_results = sweep_global(by_dataset)
        _print_results("Global Threshold Sweep", g_results, global_thresh=g_thresh)
        g_accs = [d["balanced_acc"] for d in g_results.values()]
        all_results["global"] = {
            "threshold": g_thresh,
            "macro_avg_ba": round(float(np.mean(g_accs)) * 100, 2),
            "per_dataset": g_results,
        }

    if args.per_dataset:
        pd_results = sweep_per_dataset(by_dataset)
        _print_results("Per-Dataset Optimal Thresholds", pd_results)
        pd_accs = [d["balanced_acc"] for d in pd_results.values()]
        all_results["per_dataset"] = {
            "macro_avg_ba": round(float(np.mean(pd_accs)) * 100, 2),
            "per_dataset": pd_results,
        }

    if args.task_type:
        tt_results = sweep_per_task_type(by_dataset)
        _print_results("Per-Task-Type Thresholds", tt_results)
        tt_accs = [d["balanced_acc"] for d in tt_results.values()]
        all_results["task_type"] = {
            "macro_avg_ba": round(float(np.mean(tt_accs)) * 100, 2),
            "per_dataset": tt_results,
        }

    if args.compare and len(all_results) > 1:
        print(f"\n{'=' * 72}")
        print("  Comparison: Global vs Per-Dataset vs Per-Task-Type")
        print(f"{'=' * 72}")
        for mode, data in all_results.items():
            print(f"  {mode:<20} Macro-avg BA: {data['macro_avg_ba']:.2f}%")

        if "global" in all_results and "per_dataset" in all_results:
            delta = (
                all_results["per_dataset"]["macro_avg_ba"]
                - all_results["global"]["macro_avg_ba"]
            )
            print(f"\n  Per-dataset gain over global: {delta:+.2f}pp")
        if "global" in all_results and "task_type" in all_results:
            delta = (
                all_results["task_type"]["macro_avg_ba"]
                - all_results["global"]["macro_avg_ba"]
            )
            print(f"  Per-task-type gain over global: {delta:+.2f}pp")

        # Per-dataset breakdown
        ds_names = sorted(
            set().union(
                *(data.get("per_dataset", {}).keys() for data in all_results.values()),
            ),
        )
        print(f"\n  {'Dataset':<20}", end="")
        for mode in all_results:
            print(f" {mode:>12}", end="")
        print()
        print(f"  {'-' * (20 + 13 * len(all_results))}")
        for ds_name in ds_names:
            print(f"  {ds_name:<20}", end="")
            for _mode, data in all_results.items():
                ds_data = data.get("per_dataset", {}).get(ds_name, {})
                ba = ds_data.get("balanced_acc", 0)
                print(f" {ba:>11.1%}", end="")
            print()
        print(f"{'=' * 72}")

    save_results(
        {"benchmark": "threshold_analysis", **all_results},
        "threshold_analysis.json",
    )
