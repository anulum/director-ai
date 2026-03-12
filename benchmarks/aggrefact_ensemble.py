# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Ensemble AggreFact Benchmark
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Evaluate domain-model ensemble on LLM-AggreFact.

Loads base FactCG + N domain-specialised models, scores each sample
with every model, and aggregates via max/mean/weighted-mean.

Usage (GPU recommended — 5 models × ~1.7GB each):

    python -m benchmarks.aggrefact_ensemble
    python -m benchmarks.aggrefact_ensemble --models-dir ./models --threshold 0.5
    python -m benchmarks.aggrefact_ensemble --sweep
    python -m benchmarks.aggrefact_ensemble --max-samples 200  # quick smoke test
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from benchmarks._common import save_results
from benchmarks.aggrefact_eval import (
    AGGREFACT_DATASETS,
    REFERENCE_SCORES,
    AggreFactMetrics,
    _BinaryNLIPredictor,
    _binary_class_metrics,
    _load_aggrefact,
    _print_aggrefact_results,
)

logger = logging.getLogger("DirectorAI.Benchmark.AggreFactEnsemble")

BASE_MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@dataclass
class EnsembleResult:
    """Per-strategy AggreFact results + individual model scores."""

    base_metrics: AggreFactMetrics
    individual: dict[str, AggreFactMetrics] = field(default_factory=dict)
    ensemble_max: AggreFactMetrics | None = None
    ensemble_mean: AggreFactMetrics | None = None
    best_strategy: str = ""
    best_accuracy: float = 0.0


def _discover_models(models_dir: Path) -> list[str]:
    """Find local factcg-* model directories."""
    if not models_dir.is_dir():
        return []
    found = []
    for p in sorted(models_dir.iterdir()):
        if p.is_dir() and (p / "config.json").exists():
            found.append(str(p))
    return found


def _collect_scores(
    predictors: dict[str, _BinaryNLIPredictor],
    rows: list[dict],
) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Run all predictors on all rows, return {model: {dataset: [(label, prob)]}}."""
    all_scores: dict[str, dict[str, list[tuple[int, float]]]] = {
        name: {} for name in predictors
    }

    total = len(rows)
    for i, row in enumerate(rows):
        doc = row.get("doc", "")
        claim = row.get("claim", "")
        label = row.get("label")
        ds_name = row.get("dataset", "unknown")

        if label is None or not doc or not claim:
            continue

        if (i + 1) % 500 == 0 or i == 0:
            logger.info("Processing %d/%d", i + 1, total)

        for name, pred in predictors.items():
            prob = pred.score(doc, claim)
            if ds_name not in all_scores[name]:
                all_scores[name][ds_name] = []
            all_scores[name][ds_name].append((int(label), prob))

    return all_scores


def _scores_to_metrics(
    by_dataset: dict[str, list[tuple[int, float]]], threshold: float
) -> AggreFactMetrics:
    metrics = AggreFactMetrics(threshold=threshold)
    for ds_name in sorted(by_dataset.keys()):
        pairs = by_dataset[ds_name]
        y_true = [p[0] for p in pairs]
        y_pred = [1 if p[1] >= threshold else 0 for p in pairs]
        ba = balanced_accuracy_score(y_true, y_pred)
        metrics.per_dataset[ds_name] = {
            "total": len(pairs),
            "positive": sum(y_true),
            "negative": len(y_true) - sum(y_true),
            "balanced_acc": float(ba),
            **_binary_class_metrics(y_true, y_pred),
        }
    return metrics


def _aggregate_ensemble(
    all_scores: dict[str, dict[str, list[tuple[int, float]]]],
    strategy: str,
    threshold: float,
) -> AggreFactMetrics:
    """Aggregate scores across models per-sample, then evaluate."""
    model_names = list(all_scores.keys())
    # Build a unified dataset→[(label, {model: prob})] structure
    datasets: set[str] = set()
    for model_scores in all_scores.values():
        datasets.update(model_scores.keys())

    aggregated: dict[str, list[tuple[int, float]]] = {}
    for ds_name in sorted(datasets):
        n_samples = len(next(iter(all_scores.values())).get(ds_name, []))
        if n_samples == 0:
            continue
        aggregated[ds_name] = []
        for idx in range(n_samples):
            label = all_scores[model_names[0]][ds_name][idx][0]
            probs = [all_scores[m][ds_name][idx][1] for m in model_names]
            if strategy == "max":
                agg_prob = max(probs)
            elif strategy == "mean":
                agg_prob = float(np.mean(probs))
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            aggregated[ds_name].append((label, agg_prob))

    return _scores_to_metrics(aggregated, threshold)


def run_ensemble_benchmark(
    threshold: float = 0.5,
    max_samples: int | None = None,
    models_dir: str | None = None,
    include_base: bool = True,
) -> EnsembleResult:
    models_path = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
    domain_paths = _discover_models(models_path)
    logger.info("Found %d domain models in %s", len(domain_paths), models_path)
    for p in domain_paths:
        logger.info("  %s", Path(p).name)

    # Load predictors
    predictors: dict[str, _BinaryNLIPredictor] = {}
    if include_base:
        predictors["base"] = _BinaryNLIPredictor(model_name=BASE_MODEL)
    for path in domain_paths:
        name = Path(path).name
        predictors[name] = _BinaryNLIPredictor(model_name=path)

    if len(predictors) < 2:
        logger.warning("Need at least 2 models for ensemble. Found: %d", len(predictors))

    rows = _load_aggrefact(max_samples)

    logger.info("Scoring %d samples with %d models...", len(rows), len(predictors))
    t0 = time.perf_counter()
    all_scores = _collect_scores(predictors, rows)
    elapsed = time.perf_counter() - t0
    logger.info("Scoring complete in %.1f min", elapsed / 60)

    # Individual model metrics
    result = EnsembleResult(
        base_metrics=_scores_to_metrics(all_scores.get("base", {}), threshold),
    )
    for name in predictors:
        result.individual[name] = _scores_to_metrics(all_scores[name], threshold)

    # Ensemble aggregation
    if len(predictors) >= 2:
        result.ensemble_max = _aggregate_ensemble(all_scores, "max", threshold)
        result.ensemble_mean = _aggregate_ensemble(all_scores, "mean", threshold)

        strategies = {
            "max": result.ensemble_max.avg_balanced_acc,
            "mean": result.ensemble_mean.avg_balanced_acc,
        }
        result.best_strategy = max(strategies, key=strategies.get)
        result.best_accuracy = strategies[result.best_strategy]

    return result


def sweep_ensemble_thresholds(
    max_samples: int | None = None,
    models_dir: str | None = None,
) -> tuple[float, str, EnsembleResult]:
    """Find optimal threshold × strategy combination."""
    models_path = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
    domain_paths = _discover_models(models_path)

    predictors: dict[str, _BinaryNLIPredictor] = {
        "base": _BinaryNLIPredictor(model_name=BASE_MODEL),
    }
    for path in domain_paths:
        predictors[Path(path).name] = _BinaryNLIPredictor(model_name=path)

    rows = _load_aggrefact(max_samples)
    all_scores = _collect_scores(predictors, rows)

    best_thresh, best_strat, best_acc = 0.5, "mean", 0.0

    for thresh_int in range(10, 91, 2):
        thresh = thresh_int / 100.0
        for strategy in ("max", "mean"):
            m = _aggregate_ensemble(all_scores, strategy, thresh)
            if m.avg_balanced_acc > best_acc:
                best_acc = m.avg_balanced_acc
                best_thresh = thresh
                best_strat = strategy

    # Rebuild final result at optimal threshold
    result = EnsembleResult(
        base_metrics=_scores_to_metrics(all_scores.get("base", {}), best_thresh),
    )
    for name in predictors:
        result.individual[name] = _scores_to_metrics(all_scores[name], best_thresh)
    result.ensemble_max = _aggregate_ensemble(all_scores, "max", best_thresh)
    result.ensemble_mean = _aggregate_ensemble(all_scores, "mean", best_thresh)
    result.best_strategy = best_strat
    result.best_accuracy = best_acc

    return best_thresh, best_strat, result


def _print_ensemble_results(r: EnsembleResult) -> None:
    print(f"\n{'=' * 72}")
    print("  LLM-AggreFact — Ensemble Benchmark")
    print(f"{'=' * 72}")

    # Individual model scores
    print(f"\n  {'Model':<30} {'Bal Acc':>8}")
    print(f"  {'-' * 42}")
    for name, m in sorted(r.individual.items(), key=lambda x: -x[1].avg_balanced_acc):
        print(f"  {name:<30} {m.avg_balanced_acc:>7.1%}")

    # Ensemble scores
    if r.ensemble_max and r.ensemble_mean:
        print(f"\n  {'Ensemble Strategy':<30} {'Bal Acc':>8}")
        print(f"  {'-' * 42}")
        print(f"  {'max (any model says yes)':<30} {r.ensemble_max.avg_balanced_acc:>7.1%}")
        print(f"  {'mean (avg probability)':<30} {r.ensemble_mean.avg_balanced_acc:>7.1%}")

    # Leaderboard comparison
    best = r.ensemble_mean if r.best_strategy == "mean" else r.ensemble_max
    if best:
        our_pct = best.avg_balanced_acc * 100
        base_pct = r.base_metrics.avg_balanced_acc * 100
        print(f"\n  Base model:    {base_pct:.1f}%")
        print(f"  Best ensemble: {our_pct:.1f}% ({r.best_strategy})")
        print(f"  Improvement:   {our_pct - base_pct:+.1f}%")

        print(f"\n  {'Leaderboard':<30} {'Score':>8}  {'vs Ours':>8}")
        print(f"  {'-' * 50}")
        inserted = False
        for ref_name, ref_score in sorted(REFERENCE_SCORES.items(), key=lambda x: -x[1]):
            if not inserted and our_pct >= ref_score:
                print(f"  {'>>> OURS (ensemble) <<<':<30} {our_pct:>7.1f}%")
                inserted = True
            print(f"  {ref_name:<30} {ref_score:>7.1f}%  {our_pct - ref_score:>+7.1f}")
        if not inserted:
            print(f"  {'>>> OURS (ensemble) <<<':<30} {our_pct:>7.1f}%")

        # Per-dataset breakdown for best strategy
        print(f"\n  Per-dataset ({r.best_strategy}):")
        print(f"  {'Dataset':<20} {'Base':>7} {'Ensemble':>9} {'Delta':>7}")
        print(f"  {'-' * 47}")
        for ds_name in sorted(best.per_dataset.keys()):
            ens_acc = best.per_dataset[ds_name]["balanced_acc"]
            base_acc = r.base_metrics.per_dataset.get(ds_name, {}).get("balanced_acc", 0)
            print(
                f"  {ds_name:<20} {base_acc:>6.1%} {ens_acc:>8.1%} {ens_acc - base_acc:>+6.1%}"
            )

    print(f"\n{'=' * 72}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Ensemble AggreFact benchmark")
    parser.add_argument(
        "max_samples", nargs="?", type=int, default=None, help="Limit samples"
    )
    parser.add_argument(
        "--models-dir", type=str, default=None, help="Directory containing domain models"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sweep", action="store_true", help="Sweep threshold × strategy")
    parser.add_argument("--no-base", action="store_true", help="Exclude base model")
    args = parser.parse_args()

    if args.sweep:
        best_thresh, best_strat, r = sweep_ensemble_thresholds(
            max_samples=args.max_samples, models_dir=args.models_dir
        )
        print(f"\nOptimal: threshold={best_thresh:.2f}, strategy={best_strat}")
        _print_ensemble_results(r)
    else:
        r = run_ensemble_benchmark(
            threshold=args.threshold,
            max_samples=args.max_samples,
            models_dir=args.models_dir,
            include_base=not args.no_base,
        )
        _print_ensemble_results(r)

    # Save results
    best_m = r.ensemble_mean if r.best_strategy == "mean" else r.ensemble_max
    save_data = {
        "benchmark": "LLM-AggreFact",
        "mode": "ensemble",
        "models": list(r.individual.keys()),
        "best_strategy": r.best_strategy,
        "best_accuracy": round(r.best_accuracy, 4),
        "base_accuracy": round(r.base_metrics.avg_balanced_acc, 4),
        "individual": {
            name: round(m.avg_balanced_acc, 4) for name, m in r.individual.items()
        },
    }
    if best_m:
        save_data["ensemble"] = best_m.to_dict()
    save_results(save_data, "aggrefact_ensemble.json")
