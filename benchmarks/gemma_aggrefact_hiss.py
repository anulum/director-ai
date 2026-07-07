# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# Director-Class AI - HiSS prompting for Gemma LLM-as-judge
"""CLI wrapper for the Gemma AggreFact HiSS evaluator."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from _gemma_aggrefact_hiss_core import (
    VERIFY_PROMPT,
    HiSSSampleTrace,
    build_llama,
    evaluate_dataset,
    load_aggrefact,
)
from _gemma_aggrefact_hiss_report import (
    HISS_METHOD,
    HiSSReport,
    build_report,
    log_summary,
    write_report,
)
from _judge_common import DECOMPOSE_PROMPT, compute_balanced_accuracy, parse_subclaims

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

__all__ = [
    "DECOMPOSE_PROMPT",
    "HISS_METHOD",
    "VERIFY_PROMPT",
    "HiSSReport",
    "HiSSSampleTrace",
    "build_llama",
    "build_report",
    "compute_balanced_accuracy",
    "evaluate_dataset",
    "load_aggrefact",
    "log_summary",
    "main",
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


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the HiSS evaluator."""
    parser = argparse.ArgumentParser(
        description="Evaluate Gemma LLM-as-judge with HiSS AggreFact prompts.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-samples", type=_positive_int, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/gemma_hiss.json",
    )
    parser.add_argument("--n-ctx", type=_positive_int, default=4096)
    parser.add_argument("--n-threads", type=_positive_int, default=2)
    parser.add_argument("--max-decompose-tokens", type=_positive_int, default=128)
    parser.add_argument("--max-verify-tokens", type=_positive_int, default=8)
    parser.add_argument("--log-every", type=_positive_int, default=200)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the HiSS AggreFact evaluator CLI."""
    args = _build_parser().parse_args(argv)
    output_path = Path(cast(str, args.output))
    model_path = cast(str, args.model)
    try:
        dataset = load_aggrefact(cast(int | None, args.max_samples))
        logger.info("Samples: %d", len(dataset))
        llm = build_llama(
            model_path,
            n_ctx=cast(int, args.n_ctx),
            n_threads=cast(int, args.n_threads),
        )
        (
            preds,
            labels,
            datasets_per_sample,
            subclaim_counts,
            latencies,
            traces,
            unknown,
            started_at,
        ) = evaluate_dataset(
            dataset,
            llm,
            max_decompose_tokens=cast(int, args.max_decompose_tokens),
            max_verify_tokens=cast(int, args.max_verify_tokens),
            log_every=cast(int, args.log_every),
        )
        report = build_report(
            model_path=model_path,
            sample_count=len(dataset),
            preds=preds,
            labels=labels,
            datasets_per_sample=datasets_per_sample,
            subclaim_counts=subclaim_counts,
            latencies=latencies,
            traces=traces,
            unknown=unknown,
            total_time=time.time() - started_at,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    write_report(output_path, report)
    log_summary(report, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
