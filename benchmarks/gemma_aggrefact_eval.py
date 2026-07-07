# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Gemma LLM-as-Judge AggreFact Benchmark
"""CLI wrapper for the Gemma AggreFact evaluator."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from _gemma_aggrefact_eval_core import (
    LlamaCppBackend,
    TransformersBackend,
    build_backend,
    evaluate_dataset,
    load_aggrefact,
)
from _gemma_aggrefact_eval_schema import (
    JUDGE_PROMPT,
    AggreFactDataset,
    DatasetMetric,
    EvalReport,
    JudgeBackend,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

__all__ = [
    "AggreFactDataset",
    "DatasetMetric",
    "EvalReport",
    "JUDGE_PROMPT",
    "JudgeBackend",
    "LlamaCppBackend",
    "TransformersBackend",
    "build_backend",
    "evaluate_dataset",
    "load_aggrefact",
    "main",
]


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer CLI argument."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["llama-cpp", "transformers"], required=True
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-samples", type=_positive_int, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/gemma_aggrefact.json",
    )
    parser.add_argument("--n-ctx", type=_positive_int, default=4096)
    parser.add_argument("--n-threads", type=_positive_int, default=2)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--quantize", type=str, default=None, choices=[None, "4bit"])
    parser.add_argument("--log-every", type=_positive_int, default=100)
    return parser


def _write_report(report: EvalReport, output: Path) -> None:
    """Write an evaluator report as deterministic JSON."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _log_summary(report: EvalReport, output: Path) -> None:
    """Log a human-readable evaluator summary."""
    logger.info("=" * 60)
    logger.info("Global BA: %.4f", report["global_balanced_accuracy"])
    logger.info(
        "Unknown:   %d (%.1f%%)",
        report["unknown_predictions"],
        100 * report["unknown_predictions"] / report["samples"],
    )
    logger.info(
        "Time:      %.1fmin (%.0fms/sample)",
        report["total_time_seconds"] / 60,
        report["mean_latency_ms"],
    )
    logger.info("=" * 60)
    for name, metrics in sorted(report["per_dataset"].items()):
        logger.info(
            "  %-20s %5d  %.4f",
            name,
            metrics["samples"],
            metrics["balanced_accuracy"],
        )
    logger.info("Saved: %s", output)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluator CLI."""
    args = _build_parser().parse_args(argv)
    output = Path(cast(str, args.output))
    try:
        dataset = load_aggrefact(cast(int | None, args.max_samples))
        backend = build_backend(
            cast(str, args.backend),
            model=cast(str, args.model),
            n_ctx=cast(int, args.n_ctx),
            n_threads=cast(int, args.n_threads),
            dtype=cast(str, args.dtype),
            quantize=cast(str | None, args.quantize),
        )
        report = evaluate_dataset(
            dataset,
            backend,
            model=cast(str, args.model),
            backend_name=cast(str, args.backend),
            log_every=cast(int, args.log_every),
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    _write_report(report, output)
    _log_summary(report, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
