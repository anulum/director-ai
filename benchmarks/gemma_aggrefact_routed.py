# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — per-dataset prompt routing
"""Per-dataset prompt routing for Gemma LLM-as-judge.

The 11 AggreFact subsets fall into three task families with distinct
linguistic conventions:

- summ: AggreFact-CNN, AggreFact-XSum, TofuEval-MediaS, TofuEval-MeetB
  (extractive/abstractive summarisation grounding)
- rag: RAGTruth, ClaimVerify, FactCheck-GPT, ExpertQA
  (RAG outputs, GPT-generated text, multi-hop QA)
- claim: Reveal, Lfqa, Wice
  (atomic factual claims, long-form QA)

A single uniform prompt under-fits all three. This script uses the same
underlying judge model but switches the prompt template based on
``sample['dataset']``. Expected gain per the audit: +1-3% global BA,
mostly recovering loss on summarisation tasks.

Usage::

    GGML_VK_VISIBLE_DEVICES=6 python benchmarks/gemma_aggrefact_routed.py \\
        --model /tmp/gemma-models/google_gemma-4-E4B-it-Q6_K.gguf \\
        --max-samples 29320 \\
        --output benchmarks/results/gemma_e4b_q6_routed.json
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from _gemma_aggrefact_routed_core import (
    build_llama,
    build_report,
    evaluate_dataset,
    family_distribution,
    load_aggrefact,
    log_summary,
    write_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer command-line value."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    """Build the routed evaluator command-line parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate Gemma LLM-as-judge with per-family AggreFact prompts.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-samples", type=_positive_int, default=None)
    parser.add_argument(
        "--output", type=str, default="benchmarks/results/gemma_routed.json"
    )
    parser.add_argument("--n-ctx", type=_positive_int, default=4096)
    parser.add_argument("--n-threads", type=_positive_int, default=2)
    parser.add_argument("--log-every", type=_positive_int, default=500)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the routed AggreFact benchmark CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    model_path = str(args.model)
    output_path = Path(str(args.output))
    max_samples = cast(int | None, args.max_samples)
    n_ctx = int(args.n_ctx)
    n_threads = int(args.n_threads)
    log_every = int(args.log_every)

    try:
        dataset = load_aggrefact(max_samples)
        logger.info("Samples: %d", len(dataset))
        logger.info("Family distribution: %s", family_distribution(dataset))

        logger.info("Loading: %s", model_path)
        llm = build_llama(model_path, n_ctx=n_ctx, n_threads=n_threads)
        logger.info("Loaded")

        preds, labels, datasets_list, families, latencies, unknown, t_start = (
            evaluate_dataset(dataset, llm, log_every=log_every)
        )
        total = time.time() - t_start
        report = build_report(
            model_path=model_path,
            sample_count=len(dataset),
            preds=preds,
            labels=labels,
            datasets_list=datasets_list,
            families=families,
            latencies=latencies,
            unknown=unknown,
            total=total,
        )
        write_report(output_path, report)
        log_summary(
            report=report,
            unknown=unknown,
            total=total,
            output_path=output_path,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
