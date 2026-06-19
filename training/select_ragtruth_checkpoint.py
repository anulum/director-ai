# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — select RAGTruth checkpoint by example metrics

"""Select a RAGTruth checkpoint from example-level evaluator JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _candidate(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    best = data["best"]
    return {
        "path": str(path),
        "model_dir": data.get("model_dir"),
        "model_sha256": data.get("model_sha256"),
        "f1": float(best["f1"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "balanced_accuracy": float(best["balanced_accuracy"]),
        "fpr": float(best["fpr"]),
        "p": best["p"],
        "k": best["k"],
        "tp": int(best["tp"]),
        "fp": int(best["fp"]),
        "tn": int(best["tn"]),
        "fn": int(best["fn"]),
    }


def select_checkpoint(
    paths: list[Path],
    *,
    min_f1: float = 0.763,
    max_fpr: float = 0.08,
    min_recall: float = 0.70,
    f1_tie_delta: float = 0.002,
) -> dict[str, Any]:
    """Rank checkpoint results by promotion gate, F1, then lower FPR."""
    candidates = [_candidate(path) for path in paths]
    if not candidates:
        raise ValueError("at least one result JSON is required")

    for item in candidates:
        item["passes_gate"] = (
            item["f1"] >= min_f1
            and item["fpr"] <= max_fpr
            and item["recall"] >= min_recall
        )

    best_f1 = max(item["f1"] for item in candidates)
    shortlist = [
        item for item in candidates if item["f1"] >= best_f1 - max(0.0, f1_tie_delta)
    ]
    selected = sorted(
        shortlist,
        key=lambda item: (
            item["passes_gate"],
            -item["fpr"],
            item["precision"],
            item["recall"],
            item["balanced_accuracy"],
            item["f1"],
        ),
        reverse=True,
    )[0]

    return {
        "selected": selected,
        "candidates": sorted(
            candidates,
            key=lambda item: (
                item["passes_gate"],
                item["f1"],
                -item["fpr"],
                item["precision"],
                item["recall"],
            ),
            reverse=True,
        ),
        "policy": {
            "min_f1": min_f1,
            "max_fpr": max_fpr,
            "min_recall": min_recall,
            "f1_tie_delta": f1_tie_delta,
            "tie_break": "within f1_tie_delta of best F1, choose lower FPR",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--min-f1", type=float, default=0.763)
    parser.add_argument("--max-fpr", type=float, default=0.08)
    parser.add_argument("--min-recall", type=float, default=0.70)
    parser.add_argument("--f1-tie-delta", type=float, default=0.002)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    selection = select_checkpoint(
        args.results,
        min_f1=args.min_f1,
        max_fpr=args.max_fpr,
        min_recall=args.min_recall,
        f1_tie_delta=args.f1_tie_delta,
    )
    text = json.dumps(selection, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
