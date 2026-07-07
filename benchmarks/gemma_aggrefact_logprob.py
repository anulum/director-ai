# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# Director-Class AI - Gemma LLM-as-judge logprob scoring CLI
"""Run Gemma as a hallucination judge with continuous logprob scores."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from _gemma_aggrefact_logprob_core import (
    LlamaCppLogprobBackend,
    compute_balanced_accuracy,
    evaluate_dataset,
    load_aggrefact,
    per_dataset_sweep,
    sweep_threshold,
)
from _gemma_aggrefact_logprob_report import build_report, log_summary, write_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

__all__ = [
    "LlamaCppLogprobBackend",
    "compute_balanced_accuracy",
    "evaluate_dataset",
    "load_aggrefact",
    "main",
    "per_dataset_sweep",
    "sweep_threshold",
]


def main() -> None:
    """Run the Gemma AggreFact logprob benchmark CLI."""
    parser = _build_parser()
    args = parser.parse_args()
    output_path = Path(args.output)

    try:
        dataset = load_aggrefact(args.max_samples)
        logger.info("Samples: %d", len(dataset))
        backend = LlamaCppLogprobBackend(args.model, args.n_ctx, args.n_threads)
        scores, labels, datasets, _raw_responses, latencies, started_at = (
            evaluate_dataset(
                dataset,
                backend,
                log_every=args.log_every,
            )
        )
        report = build_report(
            model_path=args.model,
            sample_count=len(dataset),
            scores=scores,
            labels=labels,
            datasets=datasets,
            latencies=latencies,
            total_time=time.time() - started_at,
        )
        write_report(output_path, report)
        log_summary(report, output_path)
    except ValueError as exc:
        parser.exit(status=1, message=f"{parser.prog}: error: {exc}\n")


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the logprob evaluator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-samples", type=_positive_int, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/gemma_logprob.json",
    )
    parser.add_argument("--n-ctx", type=_positive_int, default=4096)
    parser.add_argument("--n-threads", type=_positive_int, default=2)
    parser.add_argument("--log-every", type=_positive_int, default=500)
    return parser


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer argument."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


if __name__ == "__main__":
    main()
