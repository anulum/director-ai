# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — HiSS prompting for Gemma LLM-as-judge
"""HiSS (Hierarchical Step-by-Step) AggreFact benchmark.

Two-pass judging:

1. Decomposition pass: ask Gemma to break the claim into 1-5 atomic
   sub-claims. We parse the bullet list out of the model output.
2. Verification pass: for each sub-claim, run an independent supported /
   not-supported judgement against the same context. The aggregate
   verdict is "supported" iff every sub-claim is supported.

The audit (`gemini_research_report_2026-04-11.md` section 4.2) cites a
+5-8% AggreFact BA gain over single-shot judging. Cost: ~3-5x more tokens
generated per sample (decomposition + N sub-claim verifications).

Usage::

    GGML_VK_VISIBLE_DEVICES=5 python benchmarks/gemma_aggrefact_hiss.py \\
        --model /tmp/gemma-models/google_gemma-4-E4B-it-Q6_K.gguf \\
        --max-samples 29320 \\
        --output benchmarks/results/gemma_e4b_q6_hiss.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from _judge_common import (
    DECOMPOSE_PROMPT,
    aggregate_per_dataset,
    compute_balanced_accuracy,
    parse_subclaims,
)
from _judge_common import (
    parse_response as parse_verdict,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

VERIFY_PROMPT = """You are a fact-checking assistant. Decide if the CLAIM is fully supported by the CONTEXT.

CONTEXT:
{premise}

CLAIM:
{hypothesis}

Answer with exactly one word: SUPPORTED or NOT_SUPPORTED."""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output", type=str, default="benchmarks/results/gemma_hiss.json")
    p.add_argument("--n-ctx", type=int, default=4096)
    p.add_argument("--n-threads", type=int, default=2)
    p.add_argument("--max-decompose-tokens", type=int, default=128)
    p.add_argument("--max-verify-tokens", type=int, default=8)
    p.add_argument("--log-every", type=int, default=200)
    args = p.parse_args()

    from datasets import load_dataset
    from llama_cpp import Llama

    ds = load_dataset("lytang/LLM-AggreFact", split="test")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    logger.info("Samples: %d", len(ds))

    logger.info("Loading: %s", args.model)
    llm = Llama(
        model_path=args.model,
        n_gpu_layers=-1,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_batch=512,
        verbose=False,
        logits_all=False,
    )
    logger.info("Loaded")

    preds: list[int] = []
    labels: list[int] = []
    datasets_list: list[str] = []
    subclaim_counts: list[int] = []
    latencies: list[float] = []
    raw_samples: list[dict] = []
    unknown = 0
    t_start = time.time()

    for i, sample in enumerate(ds):
        premise = sample["doc"]
        hypothesis = sample["claim"]
        label = int(sample["label"])
        dataset_name = sample["dataset"]

        t0 = time.time()
        try:
            # Pass 1 — decomposition
            d_out = llm.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": DECOMPOSE_PROMPT.format(claim=hypothesis),
                    },
                ],
                max_tokens=args.max_decompose_tokens,
                temperature=0.0,
            )
            d_text = d_out["choices"][0]["message"]["content"]
            subclaims = parse_subclaims(d_text, hypothesis)

            # Pass 2 — verify each sub-claim
            sub_verdicts: list[int] = []
            for sub in subclaims:
                v_out = llm.create_chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": VERIFY_PROMPT.format(
                                premise=premise, hypothesis=sub
                            ),
                        },
                    ],
                    max_tokens=args.max_verify_tokens,
                    temperature=0.0,
                )
                v_text = v_out["choices"][0]["message"]["content"]
                sub_verdicts.append(parse_verdict(v_text))

            # Aggregate: supported iff every sub-claim is supported
            if any(v < 0 for v in sub_verdicts):
                pred = -1
            else:
                pred = 1 if all(v == 1 for v in sub_verdicts) else 0
            n_sub = len(subclaims)
        except Exception as e:
            logger.warning("Sample %d failed: %s", i, e)
            pred = -1
            n_sub = 0
            sub_verdicts = []

        latencies.append(time.time() - t0)
        if pred < 0:
            unknown += 1
        preds.append(pred)
        labels.append(label)
        datasets_list.append(dataset_name)
        subclaim_counts.append(n_sub)

        if i < 10:
            raw_samples.append(
                {
                    "claim": hypothesis[:120],
                    "n_sub": n_sub,
                    "sub_verdicts": sub_verdicts,
                    "pred": pred,
                    "label": label,
                }
            )

        if (i + 1) % args.log_every == 0:
            elapsed = time.time() - t_start
            ba = compute_balanced_accuracy(preds, labels)
            mean_sub = sum(subclaim_counts) / len(subclaim_counts)
            eta = (len(ds) - i - 1) * elapsed / (i + 1) / 60
            logger.info(
                "[%d/%d] BA=%.4f unk=%d mean_sub=%.2f %.0fms/sample ETA=%.1fmin",
                i + 1,
                len(ds),
                ba,
                unknown,
                mean_sub,
                1000 * elapsed / (i + 1),
                eta,
            )

    per_ds_metrics = aggregate_per_dataset(preds, labels, datasets_list)

    total = time.time() - t_start
    results = {
        "model": args.model,
        "method": "HiSS (decompose + per-subclaim verify)",
        "samples": len(ds),
        "global_balanced_accuracy": compute_balanced_accuracy(preds, labels),
        "per_dataset": per_ds_metrics,
        "unknown_predictions": unknown,
        "mean_subclaims_per_sample": sum(subclaim_counts)
        / max(len(subclaim_counts), 1),
        "total_time_seconds": total,
        "p50_latency_ms": 1000 * sorted(latencies)[len(latencies) // 2]
        if latencies
        else 0,
        "p99_latency_ms": (
            1000 * sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0
        ),
        "predictions": preds,
        "labels": labels,
        "datasets_per_sample": datasets_list,
        "subclaim_counts": subclaim_counts,
        "first_10_samples": raw_samples,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))

    logger.info("=" * 60)
    logger.info("Global BA:           %.4f", results["global_balanced_accuracy"])
    logger.info("Mean sub-claims:     %.2f", results["mean_subclaims_per_sample"])
    logger.info("Unknown:             %d (%.1f%%)", unknown, 100 * unknown / len(ds))
    logger.info("Time:                %.1fmin", total / 60)
    logger.info("=" * 60)
    for n, m in sorted(per_ds_metrics.items()):
        logger.info("  %-20s %5d  %.4f", n, m["samples"], m["balanced_accuracy"])
    logger.info("Saved: %s", args.output)


if __name__ == "__main__":
    main()
