# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RAGTruth annotation review workflow

"""Prepare and apply manual review decisions for RAGTruth FP candidates."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CANDIDATES = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "fp_a/diagnostics/ragtruth_annotation_review_candidates.jsonl"
)
DEFAULT_CACHE = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "fp_a/diagnostics/token_eval_probs.json"
)
DEFAULT_RESULT = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "fp_a/diagnostics/example_eval_result_with_diagnostics.json"
)
DEFAULT_TEMPLATE = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "fp_a/diagnostics/ragtruth_annotation_review_template.jsonl"
)
DEFAULT_OUTPUT_JSON = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "fp_a/diagnostics/ragtruth_annotation_review_packet.json"
)
DEFAULT_OUTPUT_MD = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "fp_a/diagnostics/ragtruth_annotation_review_packet.md"
)

VALID_DECISIONS = frozenset(
    {
        "confirmed_grounded",
        "confirmed_hallucinated",
        "exclude_uncertain",
    }
)


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file as a list of objects."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_review_template(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return reviewer-editable rows without changing labels."""
    template: list[dict[str, Any]] = []
    for candidate in candidates:
        row_index = int(candidate["row_index"])
        template.append(
            {
                "row_index": row_index,
                "task_type": str(candidate.get("task_type", "unknown")),
                "current_label": str(candidate.get("current_label", "grounded")),
                "primary_category": str(candidate.get("primary_category", "unknown")),
                "factors": list(candidate.get("factors", [])),
                "tokens_at_threshold": int(candidate.get("tokens_at_threshold", 0)),
                "max_token_probability": float(
                    candidate.get("max_token_probability", 0.0)
                ),
                "reviewer_decision": "",
                "reviewer_rationale": "",
                "allowed_reviewer_decisions": sorted(VALID_DECISIONS),
                "query_snippet": str(candidate.get("query_snippet", "")),
                "output_snippet": str(candidate.get("output_snippet", "")),
                "context_snippet": str(candidate.get("context_snippet", "")),
            }
        )
    return template


def _normalise_decisions(rows: Sequence[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    decisions: dict[int, dict[str, Any]] = {}
    for row in rows:
        row_index = int(row["row_index"])
        if row_index in decisions:
            raise ValueError(f"duplicate review decision for row_index={row_index}")
        decision = str(row.get("reviewer_decision", "")).strip()
        if decision not in VALID_DECISIONS:
            raise ValueError(
                f"invalid reviewer_decision for row_index={row_index}: {decision!r}"
            )
        rationale = str(row.get("reviewer_rationale", "")).strip()
        if decision != "confirmed_grounded" and not rationale:
            raise ValueError(
                f"reviewer_rationale is required for row_index={row_index}"
            )
        decisions[row_index] = {**row, "reviewer_decision": decision}
    return decisions


def _flags(
    records: Sequence[dict[str, Any]], *, threshold: float, min_tokens: int
) -> np.ndarray:
    return np.array(
        [
            sum(
                1 for value in record.get("resp_probs", []) if float(value) >= threshold
            )
            >= min_tokens
            for record in records
        ],
        dtype=bool,
    )


def _metrics(labels: np.ndarray, flagged: np.ndarray) -> dict[str, Any]:
    tp = int((flagged & (labels == 1)).sum())
    fp = int((flagged & (labels == 0)).sum())
    fn = int((~flagged & (labels == 1)).sum())
    tn = int((~flagged & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "balanced_accuracy": (tpr + tnr) / 2,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _row_index_map(records: Sequence[dict[str, Any]]) -> dict[int, int]:
    index: dict[int, int] = {}
    for position, record in enumerate(records):
        row_index = int(record.get("row_index", position))
        if row_index in index:
            raise ValueError(f"duplicate cache row_index={row_index}")
        index[row_index] = position
    return index


def estimate_review_effect(
    records: Sequence[dict[str, Any]],
    decisions: Sequence[dict[str, Any]],
    *,
    threshold: float,
    min_tokens: int,
) -> dict[str, Any]:
    """Estimate metric sensitivity from explicit manual review decisions."""
    by_row_index = _row_index_map(records)
    normalised = _normalise_decisions(decisions)
    labels = np.array([int(record["label"]) for record in records], dtype=int)
    included = np.ones(len(records), dtype=bool)
    flags = _flags(records, threshold=threshold, min_tokens=min_tokens)

    counts = {
        "confirmed_grounded": 0,
        "confirmed_hallucinated": 0,
        "exclude_uncertain": 0,
    }
    for row_index, decision_row in normalised.items():
        if row_index not in by_row_index:
            raise ValueError(f"review decision row_index={row_index} is not in cache")
        position = by_row_index[row_index]
        decision = str(decision_row["reviewer_decision"])
        counts[decision] += 1
        if decision == "confirmed_hallucinated":
            labels[position] = 1
        elif decision == "exclude_uncertain":
            included[position] = False

    if not included.any():
        raise ValueError("review decisions exclude every cached record")

    baseline_labels = np.array([int(record["label"]) for record in records], dtype=int)
    return {
        "decision_rule": {"p": threshold, "k": min_tokens},
        "reviewed_count": len(normalised),
        "decision_counts": counts,
        "baseline_metrics": _metrics(baseline_labels, flags),
        "review_adjusted_metrics": _metrics(labels[included], flags[included]),
        "excluded_count": int((~included).sum()),
        "note": (
            "Review-adjusted metrics are sensitivity analysis only; do not treat "
            "them as public benchmark evidence without a frozen review protocol."
        ),
    }


def build_review_packet(
    *,
    candidates_path: Path,
    cache_path: Path,
    result_path: Path,
    decisions_path: Path | None = None,
) -> dict[str, Any]:
    """Build a review packet and optional metric-sensitivity report."""
    candidates = load_jsonl(candidates_path)
    result = _load_json(result_path)
    best = result["best"]
    packet: dict[str, Any] = {
        "candidates_path": str(candidates_path),
        "cache_path": str(cache_path),
        "result_path": str(result_path),
        "candidate_count": len(candidates),
        "decision_rule": {"p": float(best["p"]), "k": int(best["k"])},
        "status": "manual_review_pending",
        "next_action": (
            "complete reviewer_decision and reviewer_rationale in the template; "
            "do not train from these rows before the review is frozen"
        ),
    }
    if decisions_path is None:
        return packet

    records = _load_json(cache_path)
    decisions = load_jsonl(decisions_path)
    packet["decisions_path"] = str(decisions_path)
    packet["review_effect"] = estimate_review_effect(
        records,
        decisions,
        threshold=float(best["p"]),
        min_tokens=int(best["k"]),
    )
    adjusted = packet["review_effect"]["review_adjusted_metrics"]
    packet["status"] = (
        "review_adjustment_meets_internal_gate"
        if adjusted["f1"] > 0.763 and adjusted["fpr"] <= 0.08
        else "review_adjustment_does_not_meet_internal_gate"
    )
    packet["next_action"] = (
        "use this only to decide whether a new data strategy is worth eval-only "
        "confirmation; do not claim production improvement from manual adjustment"
    )
    return packet


def write_markdown(packet: dict[str, Any], path: Path) -> None:
    """Write a compact Markdown report for the manual-review packet."""
    lines = [
        "# RAGTruth Annotation Review Packet",
        "",
        f"Status: `{packet['status']}`",
        f"Candidates: `{packet['candidate_count']}`",
        f"Rule: `p={packet['decision_rule']['p']}`, `k={packet['decision_rule']['k']}`",
        "",
        f"Next action: {packet['next_action']}",
        "",
    ]
    effect = packet.get("review_effect")
    if effect:
        baseline = effect["baseline_metrics"]
        adjusted = effect["review_adjusted_metrics"]
        lines.extend(
            [
                "## Review-Adjusted Sensitivity",
                "",
                f"- Reviewed rows: `{effect['reviewed_count']}`",
                f"- Decision counts: `{effect['decision_counts']}`",
                f"- Excluded rows: `{effect['excluded_count']}`",
                f"- Baseline: F1 `{baseline['f1']:.4f}`, precision "
                f"`{baseline['precision']:.4f}`, recall `{baseline['recall']:.4f}`, "
                f"FPR `{baseline['fpr']:.4f}`",
                f"- Adjusted: F1 `{adjusted['f1']:.4f}`, precision "
                f"`{adjusted['precision']:.4f}`, recall `{adjusted['recall']:.4f}`, "
                f"FPR `{adjusted['fpr']:.4f}`",
                "",
                effect["note"],
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    candidates = load_jsonl(args.candidates)
    _write_jsonl(args.template, build_review_template(candidates))
    packet = build_review_packet(
        candidates_path=args.candidates,
        cache_path=args.cache,
        result_path=args.result,
        decisions_path=args.decisions,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(packet, indent=2, sort_keys=True))
    write_markdown(packet, args.output_md)
    print(json.dumps({"status": packet["status"], "template": str(args.template)}))


if __name__ == "__main__":
    main()
