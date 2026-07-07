# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# Director-Class AI - routed HiSS prompting for Gemma LLM-as-judge
"""CLI wrapper for the routed Gemma AggreFact HiSS evaluator."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from _gemma_aggrefact_hiss_routed_core import (
    HiSSRoutedEvaluation,
    build_llama,
    evaluate_dataset,
    family_distribution,
    load_aggrefact,
)
from _gemma_aggrefact_hiss_routed_report import (
    HiSSRoutedReport,
    build_report,
    log_summary,
    write_report,
)
from _judge_common import (
    DATASET_TO_FAMILY,
    DECOMPOSE_PROMPT,
    PROMPTS,
    compute_balanced_accuracy,
    parse_response,
    parse_subclaims,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

__all__ = [
    "DATASET_TO_FAMILY",
    "DECOMPOSE_PROMPT",
    "PROMPTS",
    "HiSSRoutedEvaluation",
    "HiSSRoutedReport",
    "build_llama",
    "build_report",
    "compute_balanced_accuracy",
    "evaluate_dataset",
    "family_distribution",
    "load_aggrefact",
    "log_summary",
    "main",
    "parse_response",
    "parse_subclaims",
    "write_report",
]


def _positive_int(value: str) -> int:
    """Parse a positive integer command-line argument."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _support_fraction(value: str) -> float:
    """Parse a support-fraction threshold in the interval ``(0, 1]``."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a float") from exc
    if parsed <= 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("must be > 0 and <= 1")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for routed HiSS evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Gemma LLM-as-judge with routed HiSS prompts.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-samples", type=_positive_int, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/gemma_hiss_routed.json",
    )
    parser.add_argument("--n-ctx", type=_positive_int, default=4096)
    parser.add_argument("--n-threads", type=_positive_int, default=2)
    parser.add_argument("--log-every", type=_positive_int, default=500)
    parser.add_argument(
        "--min-decompose-words",
        type=_positive_int,
        default=12,
        help="Claims shorter than this skip decomposition and use the routed K=1 path.",
    )
    parser.add_argument(
        "--support-frac",
        type=_support_fraction,
        default=0.75,
        help="Verdict is SUPPORTED when the subclaim support fraction reaches this threshold.",
    )
    parser.add_argument(
        "--max-subclaims",
        type=_positive_int,
        default=4,
        help="Maximum subclaims per decomposition.",
    )
    parser.add_argument("--max-decompose-tokens", type=_positive_int, default=160)
    parser.add_argument("--max-verify-tokens", type=_positive_int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the routed HiSS AggreFact evaluator CLI."""
    args = _build_parser().parse_args(argv)
    output_path = Path(cast(str, args.output))
    model_path = cast(str, args.model)

    try:
        dataset = load_aggrefact(cast(int | None, args.max_samples))
        logger.info(
            "Samples: %d  min_decompose_words=%d  support_frac=%.2f  max_sub=%d",
            len(dataset),
            cast(int, args.min_decompose_words),
            cast(float, args.support_frac),
            cast(int, args.max_subclaims),
        )
        logger.info("Family distribution: %s", family_distribution(dataset))
        llm = build_llama(
            model_path,
            n_ctx=cast(int, args.n_ctx),
            n_threads=cast(int, args.n_threads),
        )
        evaluation = evaluate_dataset(
            dataset,
            llm,
            min_decompose_words=cast(int, args.min_decompose_words),
            support_frac=cast(float, args.support_frac),
            max_subclaims=cast(int, args.max_subclaims),
            max_decompose_tokens=cast(int, args.max_decompose_tokens),
            max_verify_tokens=cast(int, args.max_verify_tokens),
            log_every=cast(int, args.log_every),
        )
        report = build_report(
            model_path=model_path,
            sample_count=len(dataset),
            min_decompose_words=cast(int, args.min_decompose_words),
            support_frac=cast(float, args.support_frac),
            max_subclaims=cast(int, args.max_subclaims),
            skipped_decompose=evaluation["skipped_decompose"],
            preds=evaluation["preds"],
            support_fractions=evaluation["support_fractions"],
            labels=evaluation["labels"],
            datasets_per_sample=evaluation["datasets_per_sample"],
            families_per_sample=evaluation["families_per_sample"],
            subclaim_counts=evaluation["subclaim_counts"],
            decomposed_flags=evaluation["decomposed_flags"],
            latencies=evaluation["latencies"],
            unknown_predictions=evaluation["unknown_predictions"],
            total_time=time.time() - evaluation["started_at"],
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    write_report(output_path, report)
    log_summary(report, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
