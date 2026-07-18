# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — LLM-AggreFact Benchmark (Factual Consistency)

"""Evaluate NLI model on LLM-AggreFact — the standard factual consistency
benchmark aggregating 11 datasets across summarization, RAG, and
grounding tasks.

This is the benchmark used by the official leaderboard at
https://llm-aggrefact.github.io/. Published top scores (balanced acc):

    Bespoke-MiniCheck-7B    77.4%
    Claude-3.5 Sonnet       77.2%
    FactCG-DeBERTa-L        75.6%   (0.4B — our weight class)
    MiniCheck-Flan-T5-L     75.0%   (0.8B)
    HHEM-2.1                71.8%

The dataset is gated on HuggingFace. Authenticate first::

    export HF_TOKEN=hf_...
    # or: huggingface-cli login

Usage::

    python -m benchmarks.aggrefact_eval
    python -m benchmarks.aggrefact_eval --model yaxili96/FactCG-DeBERTa-v3-Large
    python -m benchmarks.aggrefact_eval --threshold 0.6
    python -m benchmarks.aggrefact_eval --sweep
    python -m benchmarks.aggrefact_eval --per-dataset
    python -m benchmarks.aggrefact_eval --agg-sweep

NLI entailment prob > threshold ->supported (1), else ->not supported (0).
Balanced accuracy per dataset, macro-averaged (same as leaderboard).

Modes:
    --sweep         Global threshold sweep (0.10-0.90, step 0.01)
    --per-dataset   Per-dataset threshold sweep (oracle upper bound)
    --agg-sweep     Compare aggregation strategies (max, mean, trimmed_mean)
    --bidirectional Compare SummaC vs bidirectional chunking
    --save-scores   Score all samples, dump raw (dataset,label,score) to JSON
    --load-scores   Load cached scores — skip inference, run threshold analysis locally
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pytest

from benchmarks._common import add_common_args, save_results

# Split by responsibility: data loading, metric primitives, and model
# predictors live in sibling modules; every name is re-exported here so
# the ``benchmarks.aggrefact_eval`` surface (imports, ``--load-scores``
# replay, and test patch targets) is unchanged.
from benchmarks.aggrefact_data import (
    AGGREFACT_DATASETS as AGGREFACT_DATASETS,
)
from benchmarks.aggrefact_data import (
    REFERENCE_SCORES as REFERENCE_SCORES,
)
from benchmarks.aggrefact_data import (
    _load_aggrefact as _load_aggrefact,
)
from benchmarks.aggrefact_metrics import (
    AggreFactMetrics as AggreFactMetrics,
)
from benchmarks.aggrefact_metrics import (
    _binary_class_metrics as _binary_class_metrics,
)
from benchmarks.aggrefact_metrics import (
    _compute_sample_pooled_ba as _compute_sample_pooled_ba,
)
from benchmarks.aggrefact_metrics import (
    _precision_recall_f1_for_label as _precision_recall_f1_for_label,
)
from benchmarks.aggrefact_metrics import (
    balanced_accuracy_score as balanced_accuracy_score,
)
from benchmarks.aggrefact_predictors import (
    _FACTCG_TEMPLATE as _FACTCG_TEMPLATE,
)
from benchmarks.aggrefact_predictors import (
    _BinaryNLIPredictor as _BinaryNLIPredictor,
)
from benchmarks.aggrefact_predictors import (
    _chunk_source as _chunk_source,
)
from benchmarks.aggrefact_predictors import (
    _NLIScorerPredictor as _NLIScorerPredictor,
)
from benchmarks.aggrefact_predictors import (
    _normalise_scorer_template as _normalise_scorer_template,
)
from benchmarks.aggrefact_predictors import (
    _uses_factcg_template as _uses_factcg_template,
)

logger = logging.getLogger("DirectorAI.Benchmark.AggreFact")

SCHEMA_VERSION = 2  # bump when JSON layout changes


def score_and_save(
    out_path: str | Path,
    max_samples: int | None = None,
    model_name: str | None = None,
    max_length: int = 2048,
    scorer_template: str | None = None,
) -> Path:
    """Score all samples once and save in the ensemble-compatible JSON schema.

    The output mirrors ``benchmarks/gemma_aggrefact_eval.py`` so a downstream
    ensemble analyser can load Gemma and FactCG predictions uniformly.

    **Four** balanced-accuracy numbers are saved, in two metric families
    × two threshold strategies. Do not confuse them:

    | Metric × Threshold | Single global threshold | Per-dataset thresholds |
    |--------------------|-------------------------|------------------------|
    | per-dataset mean   | ``per_dataset_mean_balanced_accuracy_at_global_threshold`` (= AggreFact leaderboard convention) | ``per_dataset_mean_balanced_accuracy_at_per_dataset_thresholds`` (post-hoc tuned) |
    | sample-pooled      | ``sample_pooled_balanced_accuracy_at_global_threshold`` | ``sample_pooled_balanced_accuracy_at_per_dataset_thresholds`` |

    **The leaderboard convention is per-dataset mean** (verified
    verbatim from ``https://llm-aggrefact.github.io/`` on 2026-04-12).
    Sample-pooled is reported in addition because it's convenient for
    fast comparison of judges on the same data distribution.

    Legacy aliases (``global_balanced_accuracy`` and
    ``per_dataset_avg_balanced_accuracy``) are kept for back-compat,
    they map to the per-dataset-mean variants.

    The file is also readable by ``load_cached_scores()`` (schema-version
    aware — old v1 files keep loading too).
    """
    predictor = _BinaryNLIPredictor(
        model_name=model_name,
        max_length=max_length,
        scorer_template=scorer_template,
    )
    rows = _load_aggrefact(max_samples)

    scores: list[float] = []
    labels: list[int] = []
    datasets: list[str] = []
    latencies: list[float] = []
    t_start = time.time()

    for i, row in enumerate(rows):
        doc = row.get("doc", "")
        claim = row.get("claim", "")
        label = row.get("label")
        ds_name = row.get("dataset", "unknown")
        if label is None or not doc or not claim:
            continue
        t0 = time.time()
        ent_prob = predictor.score(doc, claim)
        elapsed = time.time() - t0
        scores.append(round(float(ent_prob), 6))
        labels.append(int(label))
        datasets.append(ds_name)
        latencies.append(elapsed)
        if (i + 1) % 1000 == 0:
            logger.info("Scored %d / %d", i + 1, len(rows))

    total_time = time.time() - t_start

    by_dataset: dict[str, list[tuple[int, float]]] = {}
    for lbl, scr, ds in zip(labels, scores, datasets, strict=True):
        by_dataset.setdefault(ds, []).append((lbl, scr))

    # Per-dataset optimal thresholds (each dataset swept independently).
    per_ds_t, per_ds_metrics = sweep_on_cached(by_dataset, per_dataset=True)
    per_dataset_avg_ba = per_ds_metrics.per_dataset_mean_balanced_acc

    # Per-dataset MEAN at a single global threshold (this is the
    # AggreFact leaderboard convention — verified verbatim from
    # https://llm-aggrefact.github.io/ on 2026-04-12).
    global_thresh, global_metrics = sweep_on_cached(by_dataset, per_dataset=False)
    per_ds_mean_at_global_t = global_metrics.per_dataset_mean_balanced_acc

    # Predictions from per-dataset thresholds (most useful for ensemble work).
    predictions: list[int] = [
        1 if s >= per_ds_t.get(d, 0.5) else 0
        for s, d in zip(scores, datasets, strict=True)
    ]

    # **TRUE sample-pooled balanced accuracy.** Computed once across
    # the flat (predictions, labels) pool, weighted by dataset size.
    # This is NOT the leaderboard metric but is useful for fast
    # comparison of judges on the same data distribution. Computed
    # at the per-dataset thresholds (most-tuned variant).
    sample_pooled_ba = _compute_sample_pooled_ba(predictions, labels)
    # Also report sample-pooled at the single global threshold for
    # apples-to-apples comparison with judges that use a single
    # threshold (e.g. the Gemma routed champion).
    preds_at_global_t = [1 if s >= global_thresh else 0 for s in scores]
    sample_pooled_ba_global_t = _compute_sample_pooled_ba(preds_at_global_t, labels)

    results = {
        "schema_version": SCHEMA_VERSION,
        "model": model_name or "yaxili96/FactCG-DeBERTa-v3-Large",
        "backend": "transformers",
        "samples": len(scores),
        # ── Leaderboard metric (per-dataset mean) — both variants ─
        "per_dataset_mean_balanced_accuracy_at_global_threshold": (
            per_ds_mean_at_global_t
        ),
        "per_dataset_mean_balanced_accuracy_at_per_dataset_thresholds": (
            per_dataset_avg_ba
        ),
        # ── Sample-pooled (NOT the leaderboard metric) — both variants
        "sample_pooled_balanced_accuracy_at_global_threshold": (
            sample_pooled_ba_global_t
        ),
        "sample_pooled_balanced_accuracy_at_per_dataset_thresholds": (sample_pooled_ba),
        # ── Legacy aliases (DEPRECATED — kept for back-compat only)
        # `global_balanced_accuracy` was historically the per-dataset
        # mean at the global threshold, NOT sample-pooled, despite
        # the misleading name. Migrate to the explicit fields above.
        "global_balanced_accuracy": per_ds_mean_at_global_t,
        "per_dataset_avg_balanced_accuracy": per_dataset_avg_ba,
        # ── Thresholds + per-dataset breakdown ────────────────────
        "global_threshold": global_thresh,
        "per_dataset": {
            ds: {"samples": m["total"], "balanced_accuracy": m["balanced_acc"]}
            for ds, m in per_ds_metrics.per_dataset.items()
        },
        "per_dataset_thresholds": per_ds_t,
        # ── Raw arrays (for ensemble fusion + replay) ─────────────
        "scores": scores,
        "predictions": predictions,
        "labels": labels,
        "datasets_per_sample": datasets,
        "latencies_per_sample": latencies,
        "unknown_predictions": 0,
        "total_time_seconds": total_time,
        "mean_latency_ms": 1000 * sum(latencies) / len(latencies) if latencies else 0,
        "p50_latency_ms": (
            1000 * sorted(latencies)[len(latencies) // 2] if latencies else 0
        ),
        "p99_latency_ms": (
            1000 * sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0
        ),
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    logger.info("Saved ensemble-compatible results to %s", out)
    return out


def load_cached_scores(path: str | Path) -> dict[str, list[tuple[int, float]]]:
    """Load cached scores from ``score_and_save()`` output.

    Returns ``{dataset_name: [(label, entailment_prob), ...]}``. Schema-aware:
    accepts both the legacy v1 layout (``scores: list[{dataset,label,score}]``)
    and the v2 ensemble layout (parallel ``scores``/``labels``/``datasets_per_sample``
    lists).
    """
    data = json.loads(Path(path).read_text())
    by_dataset: dict[str, list[tuple[int, float]]] = {}
    raw_scores = data.get("scores", [])

    if raw_scores and isinstance(raw_scores[0], dict):
        # Legacy v1 layout
        for entry in raw_scores:
            by_dataset.setdefault(entry["dataset"], []).append(
                (int(entry["label"]), float(entry["score"]))
            )
    else:
        # v2 ensemble layout
        labels = data.get("labels", [])
        datasets = data.get("datasets_per_sample", [])
        if not (len(raw_scores) == len(labels) == len(datasets)):
            msg = (
                f"Cached file {path} has inconsistent list lengths: "
                f"scores={len(raw_scores)} labels={len(labels)} datasets={len(datasets)}"
            )
            raise ValueError(msg)
        for ds, lbl, scr in zip(datasets, labels, raw_scores, strict=True):
            by_dataset.setdefault(ds, []).append((int(lbl), float(scr)))

    logger.info(
        "Loaded %d cached scores across %d datasets from %s",
        sum(len(v) for v in by_dataset.values()),
        len(by_dataset),
        path,
    )
    return by_dataset


def sweep_on_cached(
    by_dataset: dict[str, list[tuple[int, float]]],
    per_dataset: bool = False,
) -> tuple[float | dict[str, float], AggreFactMetrics]:
    """Run threshold sweep on pre-computed scores (no inference needed).

    Args:
        by_dataset: {dataset: [(label, score), ...]} from load_cached_scores()
        per_dataset: if True, sweep independently per dataset (oracle bound)

    Returns (threshold_or_dict, metrics).
    """
    if per_dataset:
        per_ds_t: dict[str, float] = {}
        metrics = AggreFactMetrics(threshold=0.0)
        for ds_name in sorted(by_dataset):
            pairs = by_dataset[ds_name]
            y_true = [p[0] for p in pairs]
            y_scores = [p[1] for p in pairs]
            best_t, best_ba = 0.5, 0.0
            for thresh_int in range(10, 91):
                thresh = thresh_int / 100.0
                y_pred = [1 if s >= thresh else 0 for s in y_scores]
                ba = balanced_accuracy_score(y_true, y_pred)
                if ba > best_ba:
                    best_ba = ba
                    best_t = thresh
            per_ds_t[ds_name] = best_t
            y_pred = [1 if s >= best_t else 0 for s in y_scores]
            metrics.per_dataset[ds_name] = {
                "total": len(pairs),
                "positive": sum(y_true),
                "negative": len(y_true) - sum(y_true),
                "balanced_acc": float(best_ba),
                "threshold": best_t,
                **_binary_class_metrics(y_true, y_pred),
            }
        metrics.per_dataset_thresholds = per_ds_t
        metrics.threshold = float(np.mean(list(per_ds_t.values())))
        return per_ds_t, metrics

    # Global sweep
    best_thresh, best_avg = 0.5, 0.0
    for thresh_int in range(10, 91):
        thresh = thresh_int / 100.0
        accs = []
        for pairs in by_dataset.values():
            y_true = [p[0] for p in pairs]
            y_pred = [1 if p[1] >= thresh else 0 for p in pairs]
            accs.append(balanced_accuracy_score(y_true, y_pred))
        avg = float(np.mean(accs))
        if avg > best_avg:
            best_avg = avg
            best_thresh = thresh

    metrics = AggreFactMetrics(threshold=best_thresh)
    for ds_name in sorted(by_dataset):
        pairs = by_dataset[ds_name]
        y_true = [p[0] for p in pairs]
        y_pred = [1 if p[1] >= best_thresh else 0 for p in pairs]
        ba = balanced_accuracy_score(y_true, y_pred)
        metrics.per_dataset[ds_name] = {
            "total": len(pairs),
            "positive": sum(y_true),
            "negative": len(y_true) - sum(y_true),
            "balanced_acc": float(ba),
            **_binary_class_metrics(y_true, y_pred),
        }
    return best_thresh, metrics


def run_aggrefact_benchmark(
    threshold: float = 0.5,
    max_samples: int | None = None,
    model_name: str | None = None,
    bidirectional: bool = False,
    overlap_ratio: float = 0.0,
    scorer_template: str | None = None,
) -> AggreFactMetrics:
    if bidirectional:
        predictor = _NLIScorerPredictor(
            model_name=model_name,
            overlap_ratio=overlap_ratio,
        )
    else:
        predictor = _BinaryNLIPredictor(
            model_name=model_name,
            scorer_template=scorer_template,
        )
    rows = _load_aggrefact(max_samples)

    # Collect predictions grouped by dataset
    by_dataset: dict[str, list[tuple[int, float]]] = {}
    metrics = AggreFactMetrics(threshold=threshold)

    for row in rows:
        doc = row.get("doc", "")
        claim = row.get("claim", "")
        label = row.get("label")
        ds_name = row.get("dataset", "unknown")

        if label is None or not doc or not claim:
            continue

        t0 = time.perf_counter()
        ent_prob = predictor.score(doc, claim)
        metrics.inference_times.append(time.perf_counter() - t0)

        if ds_name not in by_dataset:
            by_dataset[ds_name] = []
        by_dataset[ds_name].append((int(label), ent_prob))

    for ds_name in sorted(by_dataset.keys()):
        pairs = by_dataset[ds_name]
        y_true = [p[0] for p in pairs]
        y_scores = [p[1] for p in pairs]
        y_pred = [1 if s >= threshold else 0 for s in y_scores]
        ba = balanced_accuracy_score(y_true, y_pred)
        n_pos = sum(y_true)
        n_neg = len(y_true) - n_pos
        metrics.per_dataset[ds_name] = {
            "total": len(pairs),
            "positive": n_pos,
            "negative": n_neg,
            "balanced_acc": float(ba),
            **_binary_class_metrics(y_true, y_pred),
        }

    return metrics


def sweep_thresholds(
    max_samples: int | None = None,
    model_name: str | None = None,
    scorer_template: str | None = None,
) -> tuple[float, AggreFactMetrics]:
    """Find the threshold that maximises average balanced accuracy."""
    predictor = _BinaryNLIPredictor(
        model_name=model_name,
        scorer_template=scorer_template,
    )
    rows = _load_aggrefact(max_samples)

    by_dataset: dict[str, list[tuple[int, float]]] = {}
    inference_times: list[float] = []

    for row in rows:
        doc = row.get("doc", "")
        claim = row.get("claim", "")
        label = row.get("label")
        ds_name = row.get("dataset", "unknown")
        if label is None or not doc or not claim:
            continue
        t0 = time.perf_counter()
        ent_prob = predictor.score(doc, claim)
        inference_times.append(time.perf_counter() - t0)
        if ds_name not in by_dataset:
            by_dataset[ds_name] = []
        by_dataset[ds_name].append((int(label), ent_prob))

    best_thresh, best_avg = 0.5, 0.0
    for thresh_int in range(10, 91):
        thresh = thresh_int / 100.0
        accs = []
        for pairs in by_dataset.values():
            y_true = [p[0] for p in pairs]
            y_pred = [1 if p[1] >= thresh else 0 for p in pairs]
            accs.append(balanced_accuracy_score(y_true, y_pred))
        avg = float(np.mean(accs))
        if avg > best_avg:
            best_avg = avg
            best_thresh = thresh

    metrics = AggreFactMetrics(threshold=best_thresh)
    metrics.inference_times = inference_times
    for ds_name in sorted(by_dataset.keys()):
        pairs = by_dataset[ds_name]
        y_true = [p[0] for p in pairs]
        y_pred = [1 if p[1] >= best_thresh else 0 for p in pairs]
        ba = balanced_accuracy_score(y_true, y_pred)
        metrics.per_dataset[ds_name] = {
            "total": len(pairs),
            "positive": sum(y_true),
            "negative": len(y_true) - sum(y_true),
            "balanced_acc": float(ba),
            **_binary_class_metrics(y_true, y_pred),
        }

    return best_thresh, metrics


def sweep_thresholds_per_dataset(
    max_samples: int | None = None,
    model_name: str | None = None,
    scorer_template: str | None = None,
) -> tuple[dict[str, float], AggreFactMetrics]:
    """Sweep thresholds independently per dataset (oracle upper bound).

    Each dataset gets its own optimal threshold. The macro BA reported
    is the mean of per-dataset BAs at their respective optimal thresholds.
    This measures the ceiling achievable with per-dataset calibration.
    """
    predictor = _BinaryNLIPredictor(
        model_name=model_name,
        scorer_template=scorer_template,
    )
    rows = _load_aggrefact(max_samples)

    by_dataset: dict[str, list[tuple[int, float]]] = {}
    inference_times: list[float] = []

    for row in rows:
        doc = row.get("doc", "")
        claim = row.get("claim", "")
        label = row.get("label")
        ds_name = row.get("dataset", "unknown")
        if label is None or not doc or not claim:
            continue
        t0 = time.perf_counter()
        ent_prob = predictor.score(doc, claim)
        inference_times.append(time.perf_counter() - t0)
        if ds_name not in by_dataset:
            by_dataset[ds_name] = []
        by_dataset[ds_name].append((int(label), ent_prob))

    per_ds_thresholds: dict[str, float] = {}
    metrics = AggreFactMetrics(threshold=0.0)
    metrics.inference_times = inference_times

    for ds_name in sorted(by_dataset.keys()):
        pairs = by_dataset[ds_name]
        y_true = [p[0] for p in pairs]
        y_scores = [p[1] for p in pairs]

        best_t, best_ba = 0.5, 0.0
        for thresh_int in range(10, 91):
            thresh = thresh_int / 100.0
            y_pred = [1 if s >= thresh else 0 for s in y_scores]
            ba = balanced_accuracy_score(y_true, y_pred)
            if ba > best_ba:
                best_ba = ba
                best_t = thresh

        per_ds_thresholds[ds_name] = best_t
        y_pred = [1 if s >= best_t else 0 for s in y_scores]
        metrics.per_dataset[ds_name] = {
            "total": len(pairs),
            "positive": sum(y_true),
            "negative": len(y_true) - sum(y_true),
            "balanced_acc": float(best_ba),
            "threshold": best_t,
            **_binary_class_metrics(y_true, y_pred),
        }

    metrics.per_dataset_thresholds = per_ds_thresholds
    metrics.threshold = float(np.mean(list(per_ds_thresholds.values())))
    return per_ds_thresholds, metrics


def sweep_aggregation(
    max_samples: int | None = None,
    model_name: str | None = None,
) -> dict[str, tuple[float, AggreFactMetrics]]:
    """Compare inner aggregation strategies: max, mean, trimmed_mean.

    Uses the production NLIScorer with bidirectional chunking to test
    different outer aggregation methods, each with its own threshold sweep.
    """
    from director_ai.core.nli import NLIScorer

    rows = _load_aggrefact(max_samples)
    import torch

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = NLIScorer(
        use_model=True,
        model_name=model_name
        or os.environ.get("DIRECTOR_NLI_MODEL", "yaxili96/FactCG-DeBERTa-v3-Large"),
        device=_device,
    )

    agg_strategies = ["max", "mean", "trimmed_mean"]
    # Score once per strategy (each produces different raw scores)
    results: dict[str, tuple[float, AggreFactMetrics]] = {}

    for strategy in agg_strategies:
        by_dataset: dict[str, list[tuple[int, float]]] = {}
        inference_times: list[float] = []

        for row in rows:
            doc = row.get("doc", "")
            claim = row.get("claim", "")
            label = row.get("label")
            ds_name = row.get("dataset", "unknown")
            if label is None or not doc or not claim:
                continue

            t0 = time.perf_counter()
            # NLIScorer returns divergence (0=entailed, 1=contradicted)
            div_score, _ = scorer.score_chunked(
                doc,
                claim,
                outer_agg=strategy,
                inner_agg="max",
            )
            # Convert divergence ->entailment probability
            ent_prob = 1.0 - div_score
            inference_times.append(time.perf_counter() - t0)

            if ds_name not in by_dataset:
                by_dataset[ds_name] = []
            by_dataset[ds_name].append((int(label), ent_prob))

        # Sweep global threshold
        best_thresh, best_avg = 0.5, 0.0
        for thresh_int in range(10, 91):
            thresh = thresh_int / 100.0
            accs = []
            for pairs in by_dataset.values():
                y_true = [p[0] for p in pairs]
                y_pred = [1 if p[1] >= thresh else 0 for p in pairs]
                accs.append(balanced_accuracy_score(y_true, y_pred))
            avg = float(np.mean(accs))
            if avg > best_avg:
                best_avg = avg
                best_thresh = thresh

        m = AggreFactMetrics(threshold=best_thresh)
        m.inference_times = inference_times
        for ds_name in sorted(by_dataset.keys()):
            pairs = by_dataset[ds_name]
            y_true = [p[0] for p in pairs]
            y_pred = [1 if p[1] >= best_thresh else 0 for p in pairs]
            ba = balanced_accuracy_score(y_true, y_pred)
            m.per_dataset[ds_name] = {
                "total": len(pairs),
                "positive": sum(y_true),
                "negative": len(y_true) - sum(y_true),
                "balanced_acc": float(ba),
                **_binary_class_metrics(y_true, y_pred),
            }
        results[strategy] = (best_thresh, m)
        logger.info(
            "Aggregation %s: %.1f%% BA @ t=%.2f", strategy, best_avg * 100, best_thresh
        )

    return results


def _print_aggrefact_results(m: AggreFactMetrics, model_label: str = "") -> None:
    title = "LLM-AggreFact — Factual Consistency Benchmark"
    if model_label:
        title += f" ({model_label})"
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")
    if m.per_dataset_thresholds:
        print(f"  Threshold:  per-dataset (avg {m.threshold:.2f})")
    else:
        print(f"  Threshold:  {m.threshold:.2f}")
    print(f"  Samples:    {m.total_samples}")
    print(f"  Avg Bal Acc: {m.avg_balanced_acc:.1%}")
    if m.inference_times:
        print(f"  Latency:    {m.avg_latency_ms:.1f} ms avg")
    print()

    has_pr = any("hallucination_recall" in d for d in m.per_dataset.values())
    has_per_ds_t = bool(m.per_dataset_thresholds)
    if has_pr:
        hdr = (
            f"  {'Dataset':<20} {'N':>5} {'BalAcc':>7}"
            f" {'H-Prec':>7} {'H-Rec':>7} {'H-F1':>7}"
        )
    elif has_per_ds_t:
        hdr = f"  {'Dataset':<20} {'N':>5} {'Pos':>5} {'Neg':>5} {'Thr':>5} {'Bal Acc':>9}"
    else:
        hdr = f"  {'Dataset':<20} {'N':>5} {'Pos':>5} {'Neg':>5} {'Bal Acc':>9}"
    print(hdr)
    print(f"  {'-' * len(hdr.strip())}")
    for ds_name, d in sorted(m.per_dataset.items()):
        if has_pr:
            print(
                f"  {ds_name:<20} {d['total']:>5}"
                f" {d['balanced_acc']:>6.1%}"
                f" {d.get('hallucination_precision', 0):>6.1%}"
                f" {d.get('hallucination_recall', 0):>6.1%}"
                f" {d.get('hallucination_f1', 0):>6.1%}",
            )
        elif has_per_ds_t:
            t = m.per_dataset_thresholds.get(ds_name, m.threshold)
            print(
                f"  {ds_name:<20} {d['total']:>5}"
                f" {d['positive']:>5} {d['negative']:>5}"
                f" {t:>5.2f}"
                f" {d['balanced_acc']:>8.1%}",
            )
        else:
            print(
                f"  {ds_name:<20} {d['total']:>5}"
                f" {d['positive']:>5} {d['negative']:>5}"
                f" {d['balanced_acc']:>8.1%}",
            )
    print()

    # Comparison with published scores
    our_pct = m.avg_balanced_acc * 100
    print(f"  {'Model':<30} {'Bal Acc':>8}  {'vs Ours':>8}")
    print(f"  {'-' * 50}")
    inserted = False
    for ref_name, ref_score in sorted(REFERENCE_SCORES.items(), key=lambda x: -x[1]):
        if not inserted and our_pct >= ref_score:
            print(f"  {'>>> OURS <<<':<30} {our_pct:>7.1f}%")
            inserted = True
        print(f"  {ref_name:<30} {ref_score:>7.1f}%  {our_pct - ref_score:>+7.1f}")
    if not inserted:
        print(f"  {'>>> OURS <<<':<30} {our_pct:>7.1f}%")

    print(f"{'=' * 72}")


# ── Pytest ─────────────────────────────────────────────────────────


@pytest.mark.slow
def test_aggrefact_sample():
    """Smoke test on a small sample (requires HF_TOKEN)."""
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("HF_TOKEN required for gated LLM-AggreFact dataset")
    m = run_aggrefact_benchmark(max_samples=100)
    _print_aggrefact_results(m)
    assert m.avg_balanced_acc > 0.50


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="LLM-AggreFact factual consistency benchmark",
    )
    add_common_args(parser)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Entailment probability threshold (default: 0.5)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep thresholds 0.10-0.90 to find optimal",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Use NLIScorer.score_chunked() bidirectional path",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Overlap ratio for sliding-window chunking (0.0-0.5, bidirectional only)",
    )
    parser.add_argument(
        "--per-dataset",
        action="store_true",
        help="Sweep thresholds independently per dataset (oracle upper bound)",
    )
    parser.add_argument(
        "--agg-sweep",
        action="store_true",
        help="Compare aggregation strategies (max, mean, trimmed_mean) via NLIScorer",
    )
    parser.add_argument(
        "--save-scores",
        type=str,
        default=None,
        metavar="PATH",
        help="Score all samples, save raw (dataset,label,score) JSON to PATH",
    )
    parser.add_argument(
        "--load-scores",
        type=str,
        default=None,
        metavar="PATH",
        help="Load cached scores from PATH — skip inference, run threshold analysis",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Max NLI tokenizer length per chunk (lower = less VRAM, default 2048)",
    )
    parser.add_argument(
        "--scorer-template",
        choices=("auto", "factcg", "sequence-pair"),
        default=os.environ.get("DIRECTOR_SCORER_TEMPLATE", "auto"),
        help=(
            "Scorer input template. Use factcg for managed FactCG artefacts "
            "after cloud storage resolution to a local cache path."
        ),
    )
    args = parser.parse_args()

    if args.save_scores:
        out = score_and_save(
            args.save_scores,
            max_samples=args.max_samples,
            model_name=args.model,
            max_length=args.max_length,
            scorer_template=args.scorer_template,
        )
        print(f"\nEnsemble-compatible results saved to {out}")
        # Print summary from the saved JSON
        with open(out) as f:
            r = json.load(f)
        print(
            f"Global Balanced Accuracy (per-dataset optimal): {r['global_balanced_accuracy']:.2%}"
        )

    elif args.load_scores:
        by_dataset = load_cached_scores(args.load_scores)
        if args.per_dataset:
            per_ds_t, m = sweep_on_cached(by_dataset, per_dataset=True)
            print("\nPer-dataset optimal thresholds:")
            for ds, t in sorted(per_ds_t.items()):
                print(f"  {ds:<20} {t:.2f}")
            _print_aggrefact_results(m, "per-dataset thresholds")
            # Compare vs global
            best_global, m_global = sweep_on_cached(by_dataset, per_dataset=False)
            print(f"\n{'=' * 60}")
            print("  Comparison: Per-Dataset vs Global Threshold")
            print(f"{'=' * 60}")
            print(
                f"  Global (t={best_global:.2f}):      {m_global.avg_balanced_acc:.2%}"
            )
            print(f"  Per-dataset:             {m.avg_balanced_acc:.2%}")
            delta = m.avg_balanced_acc - m_global.avg_balanced_acc
            print(f"  Delta:                   {delta:+.2%}")
            print()
            for ds in sorted(set(m.per_dataset) | set(m_global.per_dataset)):
                g_ba = m_global.per_dataset.get(ds, {}).get("balanced_acc", 0)
                p_ba = m.per_dataset.get(ds, {}).get("balanced_acc", 0)
                t = per_ds_t.get(ds, best_global)
                print(
                    f"  {ds:<20} {g_ba:.1%} ->{p_ba:.1%}  ({p_ba - g_ba:+.1%})  t={t:.2f}"
                )
            print(f"{'=' * 60}")
        else:
            best_thresh, m = sweep_on_cached(by_dataset, per_dataset=False)
            print(f"\nOptimal threshold: {best_thresh:.2f}")
            _print_aggrefact_results(m, "cached scores")
    elif args.per_dataset:
        per_ds_t, m = sweep_thresholds_per_dataset(
            max_samples=args.max_samples,
            model_name=args.model,
            scorer_template=args.scorer_template,
        )
        print("\nPer-dataset optimal thresholds:")
        for ds, t in sorted(per_ds_t.items()):
            print(f"  {ds:<20} {t:.2f}")
        _print_aggrefact_results(m, "per-dataset thresholds")

        # Compare vs global sweep
        best_global, m_global = sweep_thresholds(
            max_samples=args.max_samples,
            model_name=args.model,
            scorer_template=args.scorer_template,
        )
        print(f"\n{'=' * 60}")
        print("  Comparison: Per-Dataset vs Global Threshold")
        print(f"{'=' * 60}")
        print(f"  Global (t={best_global:.2f}):      {m_global.avg_balanced_acc:.2%}")
        print(f"  Per-dataset:             {m.avg_balanced_acc:.2%}")
        delta = m.avg_balanced_acc - m_global.avg_balanced_acc
        print(f"  Delta:                   {delta:+.2%}")
        print()
        for ds in sorted(set(m.per_dataset) | set(m_global.per_dataset)):
            g_ba = m_global.per_dataset.get(ds, {}).get("balanced_acc", 0)
            p_ba = m.per_dataset.get(ds, {}).get("balanced_acc", 0)
            t = per_ds_t.get(ds, best_global)
            print(
                f"  {ds:<20} {g_ba:.1%} ->{p_ba:.1%}  ({p_ba - g_ba:+.1%})  t={t:.2f}"
            )
        print(f"{'=' * 60}")
    elif args.agg_sweep:
        results = sweep_aggregation(
            max_samples=args.max_samples,
            model_name=args.model,
            scorer_template=args.scorer_template,
        )
        print(f"\n{'=' * 60}")
        print("  Aggregation Strategy Comparison")
        print(f"{'=' * 60}")
        for strategy, (thresh, m_agg) in sorted(
            results.items(), key=lambda x: -x[1][1].avg_balanced_acc
        ):
            print(f"  {strategy:<15} {m_agg.avg_balanced_acc:.2%}  (t={thresh:.2f})")
        print(f"{'=' * 60}")
        # Print detailed results for the best strategy
        best_strat = max(results, key=lambda k: results[k][1].avg_balanced_acc)
        _, m = results[best_strat]
        _print_aggrefact_results(m, f"best: {best_strat}")
    elif args.sweep:
        best_thresh, m = sweep_thresholds(
            max_samples=args.max_samples,
            model_name=args.model,
        )
        print(f"\nOptimal threshold: {best_thresh:.2f}")
        _print_aggrefact_results(m, args.model or "default")
    elif args.bidirectional:
        m_summac = run_aggrefact_benchmark(
            threshold=args.threshold,
            max_samples=args.max_samples,
            model_name=args.model,
            bidirectional=False,
            scorer_template=args.scorer_template,
        )
        _print_aggrefact_results(m_summac, "SummaC chunking")
        m_bidir = run_aggrefact_benchmark(
            threshold=args.threshold,
            max_samples=args.max_samples,
            model_name=args.model,
            bidirectional=True,
            overlap_ratio=args.overlap,
        )
        _print_aggrefact_results(
            m_bidir,
            f"Bidirectional chunking (overlap={args.overlap})",
        )
        print(f"\n{'=' * 55}")
        print("  Delta: Bidirectional vs SummaC")
        print(f"{'=' * 55}")
        for ds in sorted(set(m_summac.per_dataset) | set(m_bidir.per_dataset)):
            s = m_summac.per_dataset.get(ds, {}).get("balanced_acc", 0)
            b = m_bidir.per_dataset.get(ds, {}).get("balanced_acc", 0)
            print(f"  {ds:<20} {s:.1%} ->{b:.1%}  ({b - s:+.1%})")
        delta = m_bidir.avg_balanced_acc - m_summac.avg_balanced_acc
        print(
            f"\n  Overall: {m_summac.avg_balanced_acc:.1%} ->"
            f"{m_bidir.avg_balanced_acc:.1%}  ({delta:+.1%})",
        )
        print(f"{'=' * 55}")
    else:
        m = run_aggrefact_benchmark(
            threshold=args.threshold,
            max_samples=args.max_samples,
            model_name=args.model,
            scorer_template=args.scorer_template,
        )
        _print_aggrefact_results(m, args.model or "default")

    # Derive output filename from model name to avoid overwriting.
    # Skipped when ``--save-scores`` / ``--bidirectional`` ran — those modes
    # have already written their own output and ``m`` is not defined here.
    if "m" in locals():
        model_tag = (args.model or "default").replace("/", "_").replace("\\", "_")
        outfile = f"aggrefact_{model_tag}.json"
        save_results(
            {
                "benchmark": "LLM-AggreFact",
                "model": args.model or "default",
                **m.to_dict(),
            },
            outfile,
        )
