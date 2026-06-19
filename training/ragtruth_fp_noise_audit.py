# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RAGTruth grounded false-positive noise audit

"""Categorise high-confidence grounded RAGTruth false positives."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    from datasets import load_dataset
except ModuleNotFoundError:  # pragma: no cover - optional in unit tests
    load_dataset = None  # type: ignore[assignment]

try:
    from training.eval_ragtruth_token import _as_text, _enrich_cached_records
except ModuleNotFoundError:  # pragma: no cover - flat script mode
    from eval_ragtruth_token import _as_text, _enrich_cached_records


DEFAULT_RESULT = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "fp_a/diagnostics/example_eval_result_with_diagnostics.json"
)
DEFAULT_CACHE = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "fp_a/diagnostics/token_eval_probs.json"
)
DEFAULT_OUTPUT_JSON = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "fp_a/diagnostics/ragtruth_fp_noise_audit.json"
)
DEFAULT_OUTPUT_MD = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "fp_a/diagnostics/ragtruth_fp_noise_audit.md"
)


def _load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _snippet(value: object, limit: int) -> str:
    text = " ".join(_as_text(value).split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _probability_features(
    record: dict[str, Any], *, threshold: float
) -> dict[str, Any]:
    probs = [float(value) for value in record.get("resp_probs", [])]
    sorted_probs = sorted(probs, reverse=True)
    token_count = max(1, len(probs))
    tokens_at_threshold = sum(1 for value in probs if value >= threshold)
    return {
        "response_tokens": int(record.get("response_tokens", len(probs))),
        "tokens_at_threshold": tokens_at_threshold,
        "threshold_density": tokens_at_threshold / token_count,
        "max_token_probability": sorted_probs[0] if sorted_probs else 0.0,
        "mean_top5_token_probability": (
            float(np.mean(sorted_probs[:5])) if sorted_probs else 0.0
        ),
    }


def categorise_false_positive(item: dict[str, Any]) -> dict[str, Any]:
    """Assign deterministic audit factors to one grounded false positive."""
    factors: list[str] = []
    task_type = str(item.get("task_type", "unknown"))
    context_tokens = int(item.get("context_tokens") or 0)
    context_chars = int(item.get("context_chars") or 0)
    response_tokens = int(item.get("response_tokens") or 0)
    response_chars = int(item.get("response_chars") or 0)
    tokens_at_threshold = int(item.get("tokens_at_threshold") or 0)
    max_probability = float(item.get("max_token_probability") or 0.0)

    if task_type == "Data2txt":
        factors.append("data2txt_structural")
    if context_tokens >= 900 or context_chars >= 8000:
        factors.append("likely_truncation_or_context_loss")
    if response_tokens >= 257 or response_chars >= 1200 or tokens_at_threshold >= 40:
        factors.append("long_response_activation")
    if max_probability >= 0.95 and tokens_at_threshold >= 10:
        factors.append("possible_annotation_noise")
    if not factors:
        factors.append("model_false_positive")

    priority = [
        "data2txt_structural",
        "likely_truncation_or_context_loss",
        "long_response_activation",
        "possible_annotation_noise",
        "model_false_positive",
    ]
    primary = next(factor for factor in priority if factor in factors)
    return {"primary_category": primary, "factors": factors}


def _false_positives(
    records: Sequence[dict[str, Any]], *, threshold: float, min_tokens: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        features = _probability_features(record, threshold=threshold)
        is_flagged = features["tokens_at_threshold"] >= min_tokens
        if int(record["label"]) != 0 or not is_flagged:
            continue
        item = {
            "row_index": int(record.get("row_index", -1)),
            "task_type": str(record.get("task_type", "unknown")),
            "context_tokens": int(record.get("context_tokens") or 0),
            "context_chars": int(record.get("context_chars") or 0),
            "response_chars": int(record.get("response_chars") or 0),
            **features,
        }
        item.update(categorise_false_positive(item))
        rows.append(item)
    return sorted(
        rows,
        key=lambda row: (
            int(row["tokens_at_threshold"]),
            float(row["mean_top5_token_probability"]),
            float(row["max_token_probability"]),
        ),
        reverse=True,
    )


def _load_dataset_rows(dataset: str, split: str) -> list[dict[str, Any]]:
    if load_dataset is None:
        return []
    try:
        return list(load_dataset(dataset, split=split))
    except Exception:
        return []


def _attach_text(
    rows: list[dict[str, Any]],
    dataset_rows: Sequence[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for item in rows:
        row_index = int(item.get("row_index", -1))
        source = dataset_rows[row_index] if 0 <= row_index < len(dataset_rows) else {}
        enriched.append(
            {
                **item,
                "query_snippet": _snippet(source.get("query"), limit),
                "context_snippet": _snippet(source.get("context"), limit),
                "output_snippet": _snippet(source.get("output"), limit),
                "hallucination_labels": source.get("hallucination_labels", "[]"),
            }
        )
    return enriched


def _count(items: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(item.get(key, "unknown")) for item in items))


def _factor_counts(items: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in items:
        counts.update(str(factor) for factor in item.get("factors", []))
    return dict(counts)


def _decision(
    primary_counts: dict[str, int],
    factor_counts: dict[str, int],
    total: int,
) -> dict[str, Any]:
    structural_primary = sum(
        primary_counts.get(key, 0)
        for key in (
            "data2txt_structural",
            "likely_truncation_or_context_loss",
            "long_response_activation",
        )
    )
    structural_factor = sum(
        factor_counts.get(key, 0)
        for key in (
            "data2txt_structural",
            "likely_truncation_or_context_loss",
            "long_response_activation",
        )
    )
    annotation = primary_counts.get("possible_annotation_noise", 0)
    if total and max(structural_primary, structural_factor) / total >= 0.5:
        recommendation = (
            "do_not_launch_jarvis_from_fp_noise_audit; next intervention should be "
            "task/length/context-aware data handling, then rerun eval-only calibration"
        )
    elif total and annotation / total >= 0.25:
        recommendation = (
            "manual_label_review_before_training; do not spend Jarvis credit until "
            "reviewed rows are either excluded or relabelled"
        )
    else:
        recommendation = (
            "do_not_launch_jarvis_from_fp_noise_audit; design a new objective or "
            "router before paid training"
        )
    return {
        "jarvis_decision": "do_not_launch_jarvis",
        "recommendation": recommendation,
        "structural_primary_fraction": structural_primary / total if total else 0.0,
        "structural_factor_fraction": structural_factor / total if total else 0.0,
        "annotation_primary_fraction": annotation / total if total else 0.0,
    }


def build_noise_audit(
    *,
    cache_path: Path,
    result_path: Path,
    dataset: str = "wandb/RAGTruth-processed",
    split: str = "test",
    top_n: int = 50,
    snippet_chars: int = 420,
) -> dict[str, Any]:
    """Build a reproducible audit packet for grounded false positives."""
    result = _load_json(result_path)
    best = result["best"]
    threshold = float(best["p"])
    min_tokens = int(best["k"])
    records = _enrich_cached_records(_load_json(cache_path))
    false_positives = _false_positives(
        records,
        threshold=threshold,
        min_tokens=min_tokens,
    )
    primary_counts = _count(false_positives, "primary_category")
    factor_counts = _factor_counts(false_positives)
    dataset_rows = _load_dataset_rows(dataset, split)
    top_examples = _attach_text(
        false_positives[:top_n],
        dataset_rows,
        limit=snippet_chars,
    )
    task_type_counts = _count(false_positives, "task_type")
    packet = {
        "cache_path": str(cache_path),
        "result_path": str(result_path),
        "dataset": dataset,
        "dataset_split": split,
        "decision_rule": {"p": threshold, "k": min_tokens},
        "baseline_metrics": {
            "f1": float(best["f1"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "fpr": float(best["fpr"]),
            "tp": int(best["tp"]),
            "fp": int(best["fp"]),
            "tn": int(best["tn"]),
            "fn": int(best["fn"]),
        },
        "false_positive_count": len(false_positives),
        "primary_category_counts": primary_counts,
        "factor_counts": factor_counts,
        "task_type_counts": task_type_counts,
        "top_examples": top_examples,
    }
    packet["decision"] = _decision(primary_counts, factor_counts, len(false_positives))
    return packet


def write_markdown(packet: dict[str, Any], path: Path) -> None:
    """Write the false-positive noise audit as a compact internal report."""
    lines = [
        "# RAGTruth Grounded False-Positive Noise Audit",
        "",
        "## Decision",
        "",
        f"- Jarvis decision: `{packet['decision']['jarvis_decision']}`",
        f"- Recommendation: {packet['decision']['recommendation']}",
        f"- Structural primary fraction: "
        f"`{packet['decision']['structural_primary_fraction']:.3f}`",
        f"- Structural factor fraction: "
        f"`{packet['decision'].get('structural_factor_fraction', 0.0):.3f}`",
        f"- Annotation-review primary fraction: "
        f"`{packet['decision']['annotation_primary_fraction']:.3f}`",
        "",
        "## Baseline",
        "",
        f"- Rule: `p={packet['decision_rule']['p']}`, `k={packet['decision_rule']['k']}`",
        f"- F1 `{packet['baseline_metrics']['f1']:.4f}`, precision "
        f"`{packet['baseline_metrics']['precision']:.4f}`, recall "
        f"`{packet['baseline_metrics']['recall']:.4f}`, FPR "
        f"`{packet['baseline_metrics']['fpr']:.4f}`",
        f"- Grounded false positives: `{packet['false_positive_count']}`",
        "",
        "## Counts",
        "",
        f"- Primary categories: `{packet['primary_category_counts']}`",
        f"- Factors: `{packet['factor_counts']}`",
        f"- Task types: `{packet['task_type_counts']}`",
        "",
        "## Highest-Confidence Grounded False Positives",
        "",
    ]
    for item in packet["top_examples"][:15]:
        lines.extend(
            [
                f"### Row {item['row_index']} — {item['primary_category']}",
                "",
                f"- Task: `{item['task_type']}`; factors: `{item['factors']}`",
                f"- Context tokens/chars: `{item['context_tokens']}` / "
                f"`{item['context_chars']}`; response tokens/chars: "
                f"`{item['response_tokens']}` / `{item['response_chars']}`",
                f"- Tokens at threshold: `{item['tokens_at_threshold']}`; max token "
                f"probability: `{item['max_token_probability']:.4f}`; mean top5: "
                f"`{item['mean_top5_token_probability']:.4f}`",
                f"- Query: {item['query_snippet']}",
                f"- Output: {item['output_snippet']}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--snippet-chars", type=int, default=420)
    args = parser.parse_args()

    packet = build_noise_audit(
        cache_path=args.cache,
        result_path=args.result,
        top_n=args.top_n,
        snippet_chars=args.snippet_chars,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(packet, indent=2, sort_keys=True))
    write_markdown(packet, args.output_md)
    print(json.dumps(packet["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
