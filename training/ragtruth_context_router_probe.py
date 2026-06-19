# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RAGTruth task/length/context router probe

"""Evaluate task/length/context-aware threshold routers for RAGTruth."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from training.eval_ragtruth_token import _confusion_metrics, _enrich_cached_records
except ModuleNotFoundError:  # pragma: no cover - flat script mode
    from eval_ragtruth_token import _confusion_metrics, _enrich_cached_records


DEFAULT_CALIBRATION_CACHE = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "calibration/remote_eval/extracted/ragtruth-calibration/"
    "train_stride3_2400_token_eval_probs.json"
)
DEFAULT_TEST_CACHE = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "fp_a/diagnostics/token_eval_probs.json"
)
DEFAULT_NOISE_AUDIT = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "fp_a/diagnostics/ragtruth_fp_noise_audit.json"
)
DEFAULT_OUTPUT_JSON = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "fp_a/diagnostics/ragtruth_context_router_probe.json"
)
DEFAULT_OUTPUT_MD = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "fp_a/diagnostics/ragtruth_context_router_probe.md"
)
DEFAULT_REVIEW_JSONL = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "fp_a/diagnostics/ragtruth_annotation_review_candidates.jsonl"
)
DEFAULT_REVIEW_MD = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "fp_a/diagnostics/ragtruth_annotation_review_candidates.md"
)


@dataclass(frozen=True)
class Rule:
    """Example-level token rule for one segment."""

    p: float
    k: int
    max_density: float = 1.0


def _load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _bucket(value: object, bounds: Sequence[int]) -> str:
    numeric = int(value or 0)
    lower = 0
    for upper in bounds:
        if numeric <= upper:
            return f"{lower}-{upper}"
        lower = upper + 1
    return f">{bounds[-1]}"


def segment_key(record: dict[str, Any], mode: str) -> str:
    """Return a deterministic router segment key for one cached record."""
    task_type = str(record.get("task_type", "unknown"))
    context_bucket = _bucket(record.get("context_tokens"), (512, 1024))
    response_bucket = _bucket(
        record.get("response_tokens", len(record.get("resp_probs", []))),
        (128, 256),
    )
    if mode == "global":
        return "all"
    if mode == "task":
        return task_type
    if mode == "task_context":
        return f"{task_type}|ctx={context_bucket}"
    if mode == "task_response":
        return f"{task_type}|resp={response_bucket}"
    if mode == "task_context_response":
        return f"{task_type}|ctx={context_bucket}|resp={response_bucket}"
    raise ValueError(f"unknown router mode: {mode}")


def _flags(records: Sequence[dict[str, Any]], rule: Rule) -> np.ndarray:
    flags = []
    for record in records:
        probs = [float(value) for value in record.get("resp_probs", [])]
        token_count = max(1, len(probs))
        count = sum(value >= rule.p for value in probs)
        flags.append(count >= rule.k and count / token_count <= rule.max_density)
    return np.array(flags, dtype=bool)


def _metrics(records: Sequence[dict[str, Any]], flags: np.ndarray) -> dict[str, Any]:
    labels = np.array([int(record["label"]) for record in records], dtype=int)
    return _confusion_metrics(labels, flags)


def _candidate_rules() -> list[Rule]:
    return [
        Rule(p=p, k=k, max_density=max_density)
        for p in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.99)
        for k in (1, 2, 3, 5, 8, 13, 21, 34)
        for max_density in (1.0, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05)
    ]


def select_rule(
    records: Sequence[dict[str, Any]],
    *,
    max_fpr: float,
    min_recall: float,
) -> dict[str, Any]:
    """Select a threshold rule from calibration records only."""
    scored = []
    for rule in _candidate_rules():
        metrics = _metrics(records, _flags(records, rule))
        scored.append({"rule": asdict(rule), "metrics": metrics})
    gate = [
        row
        for row in scored
        if float(row["metrics"]["fpr"]) <= max_fpr
        and float(row["metrics"]["recall"]) >= min_recall
    ]
    pool = gate or scored
    selected = max(
        pool,
        key=lambda row: (
            float(row["metrics"]["f1"]),
            -float(row["metrics"]["fpr"]),
            float(row["metrics"]["precision"]),
            float(row["metrics"]["recall"]),
        ),
    )
    selected["selected_from_gate_pool"] = bool(gate)
    return selected


def _group_records(
    records: Sequence[dict[str, Any]], mode: str
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[segment_key(record, mode)].append(record)
    return dict(groups)


def build_router(
    records: Sequence[dict[str, Any]],
    *,
    mode: str,
    min_segment_size: int,
    max_fpr: float,
    min_recall: float,
) -> dict[str, Any]:
    """Build a segment router from calibration records."""
    default = select_rule(records, max_fpr=max_fpr, min_recall=min_recall)
    rules: dict[str, dict[str, Any]] = {}
    for key, group in _group_records(records, mode).items():
        positives = sum(int(record["label"]) for record in group)
        negatives = len(group) - positives
        if len(group) < min_segment_size or positives < 10 or negatives < 10:
            continue
        rules[key] = select_rule(group, max_fpr=max_fpr, min_recall=min_recall)
    return {"mode": mode, "default": default, "segments": rules}


def evaluate_router(
    records: Sequence[dict[str, Any]],
    router: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a router on a record sequence."""
    mode = str(router["mode"])
    default_rule = Rule(**router["default"]["rule"])
    segment_rules = {
        key: Rule(**value["rule"]) for key, value in router["segments"].items()
    }
    flags = []
    segment_counts: dict[str, int] = defaultdict(int)
    for record in records:
        key = segment_key(record, mode)
        rule = segment_rules.get(key, default_rule)
        segment_counts[key if key in segment_rules else "<default>"] += 1
        flags.append(bool(_flags([record], rule)[0]))
    return {
        **_metrics(records, np.array(flags, dtype=bool)),
        "routed_segments": dict(segment_counts),
    }


def _fixed_router(rule: Rule) -> dict[str, Any]:
    return {
        "mode": "global",
        "default": {
            "rule": asdict(rule),
            "metrics": {},
            "selected_from_gate_pool": False,
        },
        "segments": {},
    }


def _rank_variants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["test_metrics"]["f1"]),
            -float(row["test_metrics"]["fpr"]),
            float(row["test_metrics"]["precision"]),
            float(row["test_metrics"]["recall"]),
        ),
        reverse=True,
    )


def _review_candidates(noise_audit_path: Path, limit: int) -> list[dict[str, Any]]:
    if not noise_audit_path.is_file():
        return []
    packet = _load_json(noise_audit_path)
    candidates = [
        item
        for item in packet.get("top_examples", [])
        if "possible_annotation_noise" in item.get("factors", [])
    ]
    return candidates[:limit]


def build_context_router_probe(
    *,
    calibration_cache_path: Path,
    test_cache_path: Path,
    noise_audit_path: Path = DEFAULT_NOISE_AUDIT,
    max_fpr: float = 0.08,
    min_recall: float = 0.70,
    min_f1: float = 0.763,
    min_segment_size: int = 80,
    review_limit: int = 25,
) -> dict[str, Any]:
    """Build a calibration-to-test task/length/context router report."""
    calibration_records = _load_json(calibration_cache_path)
    test_records = _enrich_cached_records(_load_json(test_cache_path))
    variants: list[dict[str, Any]] = []

    baseline_router = _fixed_router(Rule(p=0.7, k=2))
    variants.append(
        {
            "name": "fixed_fp_a_rule",
            "router": baseline_router,
            "calibration_metrics": evaluate_router(
                calibration_records, baseline_router
            ),
            "test_metrics": evaluate_router(test_records, baseline_router),
        }
    )
    for mode in (
        "global",
        "task",
        "task_context",
        "task_response",
        "task_context_response",
    ):
        router = build_router(
            calibration_records,
            mode=mode,
            min_segment_size=min_segment_size,
            max_fpr=max_fpr,
            min_recall=min_recall,
        )
        variants.append(
            {
                "name": f"{mode}_router",
                "router": router,
                "calibration_metrics": evaluate_router(calibration_records, router),
                "test_metrics": evaluate_router(test_records, router),
            }
        )
    ranked = _rank_variants(variants)
    gate_candidates = [
        row
        for row in ranked
        if float(row["test_metrics"]["f1"]) > min_f1
        and float(row["test_metrics"]["fpr"]) <= max_fpr
        and float(row["test_metrics"]["recall"]) >= min_recall
    ]
    return {
        "calibration_cache_path": str(calibration_cache_path),
        "test_cache_path": str(test_cache_path),
        "noise_audit_path": str(noise_audit_path),
        "method": "calibration_cache_select_then_test_cache_evaluate",
        "gates": {"max_fpr": max_fpr, "min_recall": min_recall, "min_f1": min_f1},
        "min_segment_size": min_segment_size,
        "best": ranked[0],
        "best_gate_candidate": gate_candidates[0] if gate_candidates else None,
        "variants": ranked,
        "annotation_review_candidates": _review_candidates(
            noise_audit_path, review_limit
        ),
        "decision": (
            "eligible_for_eval_only_confirmation"
            if gate_candidates
            else "do_not_launch_jarvis_from_context_router_probe"
        ),
    }


def _metric_line(row: dict[str, Any] | None) -> str:
    if not row:
        return "none"
    metrics = row["test_metrics"]
    return (
        f"{row['name']}: F1 {metrics['f1']:.4f}, precision "
        f"{metrics['precision']:.4f}, recall {metrics['recall']:.4f}, "
        f"FPR {metrics['fpr']:.4f}"
    )


def write_review_candidates(
    candidates: Sequence[dict[str, Any]], jsonl: Path, md: Path
) -> None:
    """Write manual annotation-review candidates without changing labels."""
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("w") as handle:
        for item in candidates:
            payload = {
                "row_index": item.get("row_index"),
                "task_type": item.get("task_type"),
                "review_status": "needs_manual_review",
                "current_label": "grounded",
                "reason": "high-confidence grounded false positive with annotation-noise factor",
                "factors": item.get("factors", []),
                "query_snippet": item.get("query_snippet", ""),
                "output_snippet": item.get("output_snippet", ""),
                "context_snippet": item.get("context_snippet", ""),
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    lines = [
        "# RAGTruth Annotation Review Candidates",
        "",
        "These rows are not relabelled. They are candidates for human review before "
        "any further paid training or dataset filtering.",
        "",
    ]
    for item in candidates:
        lines.extend(
            [
                f"## Row {item.get('row_index')} — {item.get('task_type')}",
                "",
                f"- Factors: `{item.get('factors', [])}`",
                f"- Query: {item.get('query_snippet', '')}",
                f"- Output: {item.get('output_snippet', '')}",
                "",
            ]
        )
    md.write_text("\n".join(lines) + "\n")


def write_markdown(packet: dict[str, Any], path: Path) -> None:
    """Write a compact Markdown report for router results."""
    lines = [
        "# RAGTruth Task/Length/Context Router Probe",
        "",
        f"Method: `{packet['method']}`",
        f"Decision: `{packet['decision']}`",
        f"Calibration cache: `{packet['calibration_cache_path']}`",
        f"Test cache: `{packet['test_cache_path']}`",
        "",
        "## Best Held-Out Test Variant",
        "",
        f"- {_metric_line(packet['best'])}",
        f"- Gate-passing candidate: {_metric_line(packet['best_gate_candidate'])}",
        "",
        "## Ranked Held-Out Test Variants",
        "",
    ]
    for row in packet["variants"]:
        lines.append(f"- {_metric_line(row)}")
    lines.extend(
        [
            "",
            "Interpretation: segment-aware thresholding can reduce false positives, "
            "but it is acceptable for the next paid step only if the held-out test "
            "candidate clears F1, recall, and FPR together.",
            "",
            "## Manual Review Queue",
            "",
            f"- Candidates written: `{len(packet['annotation_review_candidates'])}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration-cache", type=Path, default=DEFAULT_CALIBRATION_CACHE
    )
    parser.add_argument("--test-cache", type=Path, default=DEFAULT_TEST_CACHE)
    parser.add_argument("--noise-audit", type=Path, default=DEFAULT_NOISE_AUDIT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--review-jsonl", type=Path, default=DEFAULT_REVIEW_JSONL)
    parser.add_argument("--review-md", type=Path, default=DEFAULT_REVIEW_MD)
    parser.add_argument("--max-fpr", type=float, default=0.08)
    parser.add_argument("--min-recall", type=float, default=0.70)
    parser.add_argument("--min-f1", type=float, default=0.763)
    parser.add_argument("--min-segment-size", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    packet = build_context_router_probe(
        calibration_cache_path=args.calibration_cache,
        test_cache_path=args.test_cache,
        noise_audit_path=args.noise_audit,
        max_fpr=args.max_fpr,
        min_recall=args.min_recall,
        min_f1=args.min_f1,
        min_segment_size=args.min_segment_size,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(packet, indent=2, sort_keys=True))
    write_markdown(packet, args.output_md)
    write_review_candidates(
        packet["annotation_review_candidates"],
        args.review_jsonl,
        args.review_md,
    )
    print(f"decision: {packet['decision']}")
    print(f"best: {_metric_line(packet['best'])}")
    print(f"best gate candidate: {_metric_line(packet['best_gate_candidate'])}")
    print(f"review candidates: {len(packet['annotation_review_candidates'])}")


if __name__ == "__main__":
    main()
