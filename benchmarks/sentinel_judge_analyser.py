# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Sentinel-Judge Ensemble Analyser
"""CLI wrapper for the Sentinel-Judge ensemble analyser.

Usage
-----
Run the analyser against two or more judge result files::

    python benchmarks/sentinel_judge_analyser.py \
        --judges \
            benchmarks/results/gemma_e4b_q6_with_preds.json \
            benchmarks/results/factcg_with_scores.json \
        --output benchmarks/results/sentinel_judge_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from _sentinel_judge_metrics import (
    balanced_accuracy,
    build_report,
    lr_fusion_ensemble,
    oracle_upper_bound,
    per_dataset_ba,
    routed_ensemble,
    voting_ensemble,
)
from _sentinel_judge_schema import (
    JudgeRecord,
    SentinelReport,
    align_judges,
    load_judge,
)

__all__ = [
    "JudgeRecord",
    "SentinelReport",
    "align_judges",
    "balanced_accuracy",
    "build_report",
    "load_judge",
    "lr_fusion_ensemble",
    "main",
    "oracle_upper_bound",
    "per_dataset_ba",
    "routed_ensemble",
    "voting_ensemble",
]


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judges",
        nargs="+",
        required=True,
        help="Paths to judge JSON files (v2 ensemble schema)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/sentinel_judge_report.json",
    )
    return parser


def _write_report(report: SentinelReport, output: Path) -> None:
    """Write a Sentinel report to disk as deterministic JSON."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _print_summary(report: SentinelReport, output: Path) -> None:
    """Print the human-readable Sentinel report summary."""
    print()
    print("=" * 60)
    print(f"  SENTINEL-JUDGE ENSEMBLE REPORT — {report['samples']} samples")
    print("=" * 60)
    for name in report["judges"]:
        ba = report["individual"][name]["global_balanced_accuracy"]
        print(f"  {name:50s} {ba:.4f}")
    print("-" * 60)
    print(
        "  voting (majority)                               "
        f"{report['voting']['global_balanced_accuracy']:.4f}",
    )
    print(
        "  routed (per-dataset best, 50/50 split)          "
        f"{report['routed']['global_balanced_accuracy']:.4f}",
    )
    if report["lr_fusion"] is not None:
        print(
            "  LR fusion (5-fold CV)                           "
            f"{report['lr_fusion']['global_balanced_accuracy']:.4f}",
        )
    print(
        "  oracle upper bound                              "
        f"{report['oracle_upper_bound']['global_balanced_accuracy']:.4f}",
    )
    print("=" * 60)
    print(f"  Saved: {output}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Sentinel-Judge analyser CLI."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    args = _build_parser().parse_args(argv)
    judge_paths = cast(list[str], args.judges)
    output = Path(cast(str, args.output))
    try:
        report = build_report(judge_paths)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    _write_report(report, output)
    _print_summary(report, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
