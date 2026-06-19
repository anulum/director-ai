# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RAGTruth false-positive audit and calibration

"""Audit RAGTruth false positives and sweep cheap example-level calibrators."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

try:
    from training.eval_ragtruth_token import (
        _confusion_metrics,
        _enrich_cached_records,
    )
except ModuleNotFoundError:  # pragma: no cover - flat script mode
    from eval_ragtruth_token import _confusion_metrics, _enrich_cached_records


DEFAULT_FP_A_RESULT = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "fp_a/diagnostics/example_eval_result_with_diagnostics.json"
)
DEFAULT_FP_A_CACHE = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "fp_a/diagnostics/token_eval_probs.json"
)
DEFAULT_HARDNEG_A_RESULT = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "hardneg_a/extracted_metrics/example_eval_result.json"
)
DEFAULT_HARDNEG_B_RESULT = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "hardneg_b/remote_metrics/ragtruth-hardneg-b-evidence/"
    "checkpoint_2832_example_eval_result.json"
)
DEFAULT_OUTPUT_JSON = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "fp_a/diagnostics/ragtruth_fp_audit_and_calibration.json"
)
DEFAULT_OUTPUT_MD = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "fp_a/diagnostics/ragtruth_fp_audit_and_calibration.md"
)


def _load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _best(result: dict[str, Any]) -> dict[str, Any]:
    best = result.get("best")
    if not isinstance(best, dict):
        raise ValueError("result JSON is missing object field 'best'")
    return best


def _normalise_result(path: Path) -> dict[str, Any]:
    result = _load_json(path)
    best = _best(result)
    return {
        "path": str(path),
        "model_dir": result.get("model_dir"),
        "model_sha256": result.get("model_sha256"),
        "f1": float(best["f1"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "balanced_accuracy": float(best["balanced_accuracy"]),
        "fpr": float(best["fpr"]),
        "tp": int(best["tp"]),
        "fp": int(best["fp"]),
        "tn": int(best["tn"]),
        "fn": int(best["fn"]),
        "p": float(best["p"]),
        "k": int(best["k"]),
    }


def _group_rows(result: dict[str, Any], diagnostic_key: str) -> list[dict[str, Any]]:
    rows = result.get("diagnostics", {}).get(diagnostic_key, [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _worst_groups(result: dict[str, Any], diagnostic_key: str) -> list[dict[str, Any]]:
    rows = _group_rows(result, diagnostic_key)
    return sorted(
        rows,
        key=lambda row: (int(row.get("fp", 0)), float(row.get("fpr", 0.0))),
        reverse=True,
    )


def token_features(record: dict[str, Any], *, p: float) -> dict[str, float]:
    """Return deterministic token-score features for one RAGTruth row."""
    probs = [float(x) for x in record.get("resp_probs", [])]
    token_count = max(1, len(probs))
    threshold_count = sum(1 for value in probs if value >= p)
    sorted_probs = sorted(probs, reverse=True)
    return {
        "tokens_at_threshold": float(threshold_count),
        "threshold_density": threshold_count / token_count,
        "max_token_probability": sorted_probs[0] if sorted_probs else 0.0,
        "mean_top5_token_probability": (
            float(np.mean(sorted_probs[:5])) if sorted_probs else 0.0
        ),
    }


def apply_rule(
    records: list[dict[str, Any]],
    *,
    p: float,
    k: int,
    min_max_probability: float,
    max_threshold_density: float,
) -> np.ndarray:
    """Apply an example-level threshold/density calibrator."""
    flags = []
    for record in records:
        features = token_features(record, p=p)
        flags.append(
            features["tokens_at_threshold"] >= k
            and features["max_token_probability"] >= min_max_probability
            and features["threshold_density"] <= max_threshold_density
        )
    return np.array(flags, dtype=bool)


def sweep_calibrators(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sweep cheap post-hoc calibrator rules over cached token probabilities."""
    labels = np.array([int(record["label"]) for record in records])
    rows: list[dict[str, Any]] = []
    for p in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.99):
        for k in (1, 2, 3, 5, 8, 13, 21):
            for min_max_probability in (0.0, 0.7, 0.8, 0.9, 0.95, 0.98):
                for max_threshold_density in (1.0, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05):
                    flagged = apply_rule(
                        records,
                        p=p,
                        k=k,
                        min_max_probability=min_max_probability,
                        max_threshold_density=max_threshold_density,
                    )
                    metrics = _confusion_metrics(labels, flagged)
                    rows.append(
                        {
                            **metrics,
                            "p": p,
                            "k": k,
                            "min_max_probability": min_max_probability,
                            "max_threshold_density": max_threshold_density,
                        }
                    )
    return rows


def _rank_by_f1(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["f1"]),
            -float(row["fpr"]),
            float(row["precision"]),
            float(row["recall"]),
        ),
        reverse=True,
    )


def _rank_gate(rows: Iterable[dict[str, Any]], max_fpr: float) -> list[dict[str, Any]]:
    return sorted(
        [row for row in rows if float(row["fpr"]) <= max_fpr],
        key=lambda row: (
            float(row["f1"]),
            float(row["precision"]),
            float(row["recall"]),
        ),
        reverse=True,
    )


def build_audit(
    *,
    cache_path: Path,
    result_paths: list[Path],
    max_fpr: float = 0.08,
    min_f1: float = 0.763,
) -> dict[str, Any]:
    """Build a false-positive audit and post-hoc calibration decision packet."""
    records = _enrich_cached_records(_load_json(cache_path))
    result_jsons = {str(path): _load_json(path) for path in result_paths}
    result_summaries = [_normalise_result(path) for path in result_paths]
    calibrators = sweep_calibrators(records)
    best_by_f1 = _rank_by_f1(calibrators)[:20]
    best_under_fpr_gate = _rank_gate(calibrators, max_fpr)[:20]
    selected_low_fpr = best_under_fpr_gate[0] if best_under_fpr_gate else None
    selected_overall = best_by_f1[0] if best_by_f1 else None
    smoke_ready = bool(
        selected_low_fpr
        and float(selected_low_fpr["f1"]) >= min_f1
        and float(selected_low_fpr["fpr"]) <= max_fpr
    )

    fp_a_result = result_jsons.get(str(result_paths[0]), {})
    return {
        "cache_path": str(cache_path),
        "result_summaries": result_summaries,
        "worst_false_positive_groups": {
            "task_type": _worst_groups(fp_a_result, "by_task_type")[:5],
            "response_token_bucket": _worst_groups(
                fp_a_result, "by_response_token_bucket"
            )[:5],
            "context_token_bucket": _worst_groups(
                fp_a_result, "by_context_token_bucket"
            )[:5],
            "context_char_bucket": _worst_groups(fp_a_result, "by_context_char_bucket")[
                :5
            ],
        },
        "calibration": {
            "max_fpr_gate": max_fpr,
            "min_f1_gate": min_f1,
            "selected_overall": selected_overall,
            "selected_low_fpr": selected_low_fpr,
            "best_by_f1": best_by_f1,
            "best_under_fpr_gate": best_under_fpr_gate,
            "smoke_ready": smoke_ready,
            "decision": (
                "eligible_for_one_limited_smoke"
                if smoke_ready
                else "do_not_launch_jarvis_from_posthoc_calibration"
            ),
        },
    }


def _metric_line(row: dict[str, Any] | None) -> str:
    if not row:
        return "none"
    return (
        f"F1 {float(row['f1']):.4f}, precision {float(row['precision']):.4f}, "
        f"recall {float(row['recall']):.4f}, FPR {float(row['fpr']):.4f}, "
        f"p={row['p']}, k={row['k']}"
    )


def write_markdown(packet: dict[str, Any], path: Path) -> None:
    """Write a compact internal Markdown report for humans."""
    calibration = packet["calibration"]
    lines = [
        "# RAGTruth False-Positive Audit and Calibration",
        "",
        "## Candidate Results",
        "",
    ]
    for row in packet["result_summaries"]:
        lines.append(
            "- "
            f"`{Path(row['path']).name}`: F1 `{row['f1']:.4f}`, "
            f"precision `{row['precision']:.4f}`, recall `{row['recall']:.4f}`, "
            f"FPR `{row['fpr']:.4f}`, p `{row['p']}`, k `{row['k']}`"
        )
    lines.extend(
        [
            "",
            "## Worst False-Positive Groups",
            "",
        ]
    )
    for name, rows in packet["worst_false_positive_groups"].items():
        lines.append(f"### {name}")
        for row in rows:
            lines.append(
                "- "
                f"`{row.get('group')}`: n `{row.get('n')}`, fp `{row.get('fp')}`, "
                f"FPR `{float(row.get('fpr', 0.0)):.4f}`, "
                f"F1 `{float(row.get('f1', 0.0)):.4f}`"
            )
        lines.append("")
    lines.extend(
        [
            "## Calibration Sweep",
            "",
            f"- Best F1 rule: {_metric_line(calibration['selected_overall'])}",
            f"- Best rule under FPR gate: {_metric_line(calibration['selected_low_fpr'])}",
            f"- Decision: `{calibration['decision']}`",
            "",
            "Interpretation: post-hoc calibration can force FPR under the gate, "
            "but the best low-FPR rule does not meet the F1 promotion gate. "
            "Do not spend JarvisLabs credit on a smoke run unless a new method "
            "first clears both gates on held-out evidence.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_FP_A_CACHE)
    parser.add_argument("--result", type=Path, action="append", dest="results")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--max-fpr", type=float, default=0.08)
    parser.add_argument("--min-f1", type=float, default=0.763)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_paths = args.results or [
        DEFAULT_FP_A_RESULT,
        DEFAULT_HARDNEG_A_RESULT,
        DEFAULT_HARDNEG_B_RESULT,
    ]
    packet = build_audit(
        cache_path=args.cache,
        result_paths=result_paths,
        max_fpr=args.max_fpr,
        min_f1=args.min_f1,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(packet, indent=2, sort_keys=True))
    write_markdown(packet, args.output_md)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    print(f"decision: {packet['calibration']['decision']}")
    print(f"best low-FPR: {_metric_line(packet['calibration']['selected_low_fpr'])}")


if __name__ == "__main__":
    main()
