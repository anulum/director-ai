"""Analyze pre-computed score JSONs from benchmarks/scores/.

Each JSON maps dataset_name -> [(label, score), ...] as produced by
_bench_model.py and run_ensemble_seq.py.

Usage:
    python -m benchmarks.analyze_scores
    python -m benchmarks.analyze_scores --scores-dir benchmarks/scores
    python -m benchmarks.analyze_scores --top 5 --ensemble
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DIRECTOR_AI_BASE = 0.7586  # base model full-29K reference


def load_scores(scores_dir: Path) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Load all *.json score files. Returns {model_name: {dataset: [(label, score)]}}."""
    out: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for f in sorted(scores_dir.glob("*.json")):
        name = f.stem
        data: dict[str, list] = json.loads(f.read_text())
        # Normalize: values can be [[label, score], ...] or {"label":..., "score":...}
        parsed: dict[str, list[tuple[int, float]]] = {}
        for ds, rows in data.items():
            parsed[ds] = [(int(r[0]), float(r[1])) for r in rows if len(r) >= 2]
        out[name] = parsed
    return out


def macro_ba(data: dict[str, list[tuple[int, float]]], t: float) -> float:
    bas = []
    for rows in data.values():
        labels = np.array([r[0] for r in rows])
        scores = np.array([r[1] for r in rows])
        if len(np.unique(labels)) < 2:
            continue
        preds = (scores >= t).astype(int)
        recalls = [(preds[labels == c] == c).mean() for c in np.unique(labels)]
        bas.append(float(np.mean(recalls)))
    return float(np.mean(bas)) if bas else 0.0


def best_threshold(data: dict) -> tuple[float, float]:
    best_ba, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.96, 0.05):
        ba = macro_ba(data, t)
        if ba > best_ba:
            best_ba, best_t = ba, float(t)
    return best_t, best_ba


def per_dataset_ba(data: dict, t: float) -> dict[str, float]:
    result = {}
    for ds, rows in data.items():
        labels = np.array([r[0] for r in rows])
        scores = np.array([r[1] for r in rows])
        if len(np.unique(labels)) < 2:
            result[ds] = float("nan")
            continue
        preds = (scores >= t).astype(int)
        recalls = [(preds[labels == c] == c).mean() for c in np.unique(labels)]
        result[ds] = float(np.mean(recalls))
    return result


def ensemble_max(
    models: dict[str, dict[str, list[tuple[int, float]]]],
) -> dict[str, list[tuple[int, float]]]:
    """Max-aggregate scores across models. Requires identical row order per dataset."""
    datasets = set()
    for m in models.values():
        datasets |= set(m.keys())
    out: dict[str, list[tuple[int, float]]] = {}
    for ds in datasets:
        # Align by index within dataset
        model_ds = [m[ds] for m in models.values() if ds in m]
        if not model_ds:
            continue
        n = min(len(r) for r in model_ds)
        rows = []
        for i in range(n):
            label = model_ds[0][i][0]
            max_score = max(m[i][1] for m in model_ds)
            rows.append((label, max_score))
        out[ds] = rows
    return out


def ensemble_mean(
    models: dict[str, dict[str, list[tuple[int, float]]]],
) -> dict[str, list[tuple[int, float]]]:
    datasets = set()
    for m in models.values():
        datasets |= set(m.keys())
    out: dict[str, list[tuple[int, float]]] = {}
    for ds in datasets:
        model_ds = [m[ds] for m in models.values() if ds in m]
        if not model_ds:
            continue
        n = min(len(r) for r in model_ds)
        rows = []
        for i in range(n):
            label = model_ds[0][i][0]
            mean_score = float(np.mean([m[i][1] for m in model_ds]))
            rows.append((label, mean_score))
        out[ds] = rows
    return out


def print_ranking(
    all_scores: dict[str, dict],
    base_name: str = "base",
) -> list[tuple[str, float, float]]:
    """Print ranked table. Returns sorted [(name, ba, threshold)]."""
    ranked = []
    for name, data in all_scores.items():
        t, ba = best_threshold(data)
        ranked.append((name, ba, t))
    ranked.sort(key=lambda x: x[1], reverse=True)

    base_ba = next((ba for n, ba, _ in ranked if n == base_name), DIRECTOR_AI_BASE)

    print(f"\n{'Model':<32s}  {'Best BA':>8s}  {'Thresh':>7s}  {'vs Base':>9s}")
    print("-" * 65)
    for name, ba, t in ranked:
        diff = ba - base_ba
        sign = "+" if diff >= 0 else ""
        tag = (
            "  ** BEATS BASE"
            if diff > 0.005
            else ("  -- hurts" if diff < -0.005 else "")
        )
        print(
            f"  {name:<30s}  {ba * 100:.2f}%  {t:.2f}    {sign}{diff * 100:.2f}%{tag}"
        )
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze pre-computed AggreFact score JSONs"
    )
    parser.add_argument("--scores-dir", default="benchmarks/scores", type=Path)
    parser.add_argument("--top", type=int, default=3, help="Top-N for ensemble search")
    parser.add_argument(
        "--ensemble", action="store_true", help="Try all 2-model combinations"
    )
    parser.add_argument(
        "--per-dataset", action="store_true", help="Print per-dataset breakdown"
    )
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir)
    if not scores_dir.is_dir():
        print(f"ERROR: {scores_dir} not found")
        sys.exit(1)

    all_scores = load_scores(scores_dir)
    if not all_scores:
        print(f"No .json files in {scores_dir}")
        sys.exit(1)

    print(f"\nLoaded {len(all_scores)} score files from {scores_dir}")
    ranked = print_ranking(all_scores)

    base_ba = next((ba for n, ba, _ in ranked if n == "base"), DIRECTOR_AI_BASE)

    if args.per_dataset:
        base_data = all_scores.get("base")
        if base_data:
            base_t = next(t for n, _, t in ranked if n == "base")
            print(f"\nPer-dataset breakdown (base @ t={base_t:.2f}):")
            per_ds = per_dataset_ba(base_data, base_t)
            for ds, ba in sorted(per_ds.items(), key=lambda x: x[1]):
                n = len(base_data.get(ds, []))
                print(f"  {ds:<30s} {ba:.3f}  (n={n})")

    if args.ensemble:
        print(f"\n{'Ensemble combinations (top models)'}")
        print("-" * 65)
        # Only test models that individually improve or are top-N
        top_names = [n for n, _, _ in ranked[: args.top + 1]]
        candidates = {n: all_scores[n] for n in top_names if n in all_scores}

        best_ens_ba = 0.0
        best_ens_desc = ""
        for n in range(2, min(len(candidates) + 1, 5)):
            for combo in combinations(candidates.keys(), n):
                combo_data = {k: candidates[k] for k in combo}
                for agg_name, agg_fn in [
                    ("max", ensemble_max),
                    ("mean", ensemble_mean),
                ]:
                    merged = agg_fn(combo_data)
                    t, ba = best_threshold(merged)
                    diff = ba - base_ba
                    sign = "+" if diff >= 0 else ""
                    label = f"{agg_name}({', '.join(combo)})"
                    tag = "  **" if diff > 0.005 else ""
                    print(
                        f"  {label[:58]:<60s} {ba * 100:.2f}%  {sign}{diff * 100:.2f}%{tag}"
                    )
                    if ba > best_ens_ba:
                        best_ens_ba = ba
                        best_ens_desc = label

        if best_ens_ba > base_ba + 0.005:
            print(f"\nBest ensemble: {best_ens_desc}")
            print(
                f"  BA: {best_ens_ba * 100:.2f}%  (+{(best_ens_ba - base_ba) * 100:.2f}pp vs base)"
            )
        else:
            print(f"\nNo ensemble beats base by >0.5pp. Base BA: {base_ba * 100:.2f}%")


if __name__ == "__main__":
    main()
