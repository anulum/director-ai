# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# Director-Class AI - Gemma self-consistency scoring CLI
"""Run routed Gemma prompts with self-consistency voting on AggreFact."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from _gemma_aggrefact_self_consistency_core import (
    SelfConsistencyLlamaModel,
    build_llama,
    evaluate_dataset,
    family_distribution,
    load_aggrefact,
    vote_support_fraction,
)
from _gemma_aggrefact_self_consistency_report import (
    SelfConsistencyReport,
    build_report,
    log_summary,
    write_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

__all__ = [
    "SelfConsistencyLlamaModel",
    "SelfConsistencyReport",
    "build_llama",
    "build_report",
    "evaluate_dataset",
    "family_distribution",
    "load_aggrefact",
    "log_summary",
    "main",
    "vote_support_fraction",
    "write_report",
]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the routed Gemma AggreFact self-consistency benchmark CLI."""
    args = _build_parser().parse_args(argv)
    output_path = Path(cast(str, args.output))
    model_path = cast(str, args.model)

    try:
        dataset = load_aggrefact(cast(int | None, args.max_samples))
        logger.info(
            "Samples: %d  K=%d  T=%.2f",
            len(dataset),
            cast(int, args.k),
            cast(float, args.temperature),
        )
        logger.info("Family distribution: %s", family_distribution(dataset))
        llm = build_llama(
            model_path,
            n_ctx=cast(int, args.n_ctx),
            n_threads=cast(int, args.n_threads),
        )
        (
            preds,
            support_fractions,
            labels,
            datasets_per_sample,
            families_per_sample,
            latencies,
            unknown,
            started_at,
        ) = evaluate_dataset(
            dataset,
            llm,
            k=cast(int, args.k),
            temperature=cast(float, args.temperature),
            top_p=cast(float, args.top_p),
            log_every=cast(int, args.log_every),
        )
        report = build_report(
            model_path=model_path,
            sample_count=len(dataset),
            k=cast(int, args.k),
            temperature=cast(float, args.temperature),
            top_p=cast(float, args.top_p),
            preds=preds,
            support_fractions=support_fractions,
            labels=labels,
            datasets_per_sample=datasets_per_sample,
            families_per_sample=families_per_sample,
            latencies=latencies,
            unknown=unknown,
            total_time=time.time() - started_at,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    write_report(output_path, report)
    log_summary(report, output_path)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the self-consistency evaluator."""
    parser = argparse.ArgumentParser(
        description="Evaluate Gemma LLM-as-judge with routed self-consistency.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-samples", type=_positive_int, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/gemma_routed_self_consistency.json",
    )
    parser.add_argument("--n-ctx", type=_positive_int, default=4096)
    parser.add_argument("--n-threads", type=_positive_int, default=2)
    parser.add_argument("--log-every", type=_positive_int, default=500)
    parser.add_argument(
        "--k",
        type=_positive_int,
        default=3,
        help="Number of samples per claim for self-consistency",
    )
    parser.add_argument(
        "--temperature",
        type=_non_negative_float,
        default=0.4,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=_probability,
        default=0.95,
        help="Nucleus sampling cumulative probability",
    )
    return parser


def _positive_int(value: str) -> int:
    """Parse a positive integer command-line argument."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _non_negative_float(value: str) -> float:
    """Parse a non-negative floating-point command-line argument."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a float") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _probability(value: str) -> float:
    """Parse a probability in the open-closed interval ``(0, 1]``."""
    parsed = _non_negative_float(value)
    if parsed <= 0 or parsed > 1:
        raise argparse.ArgumentTypeError("must be > 0 and <= 1")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
