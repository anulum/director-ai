# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Gemma judge with chain-of-thought prompting
"""Like gemma_aggrefact_eval.py but with a step-by-step (CoT) prompt.

We test whether forcing the model to reason briefly before answering
improves balanced accuracy on AggreFact. Cost: ~3-5x more tokens generated.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

COT_PROMPT = """You are a careful fact-checking assistant. Decide if the CLAIM is fully supported by the CONTEXT.

CONTEXT:
{premise}

CLAIM:
{hypothesis}

Think step by step in 1-2 short sentences, then on the last line write exactly one of:
ANSWER: SUPPORTED
ANSWER: NOT_SUPPORTED"""


def parse_cot(text: str) -> int:
    """Extract verdict from CoT response."""
    t = text.upper()
    # Look for explicit ANSWER line
    m = re.search(r"ANSWER\s*:\s*(NOT[_\s-]?SUPPORTED|SUPPORTED)", t)
    if m:
        verdict = m.group(1)
        return 0 if "NOT" in verdict else 1
    # Fallback
    if "NOT_SUPPORTED" in t or "NOT SUPPORTED" in t:
        return 0
    if "SUPPORTED" in t:
        return 1
    return -1


def compute_ba(preds, labels):
    pos = neg = tp = tn = 0
    for p, l in zip(preds, labels, strict=True):
        if p < 0:
            continue
        if l == 1:
            pos += 1
            if p == 1:
                tp += 1
        else:
            neg += 1
            if p == 0:
                tn += 1
    if pos == 0 or neg == 0:
        return 0.0
    return (tp / pos + tn / neg) / 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--output", type=str,
                   default="benchmarks/results/gemma_cot.json")
    p.add_argument("--n-ctx", type=int, default=4096)
    p.add_argument("--n-threads", type=int, default=2)
    p.add_argument("--log-every", type=int, default=500)
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
        n_gpu_layers=-1, n_ctx=args.n_ctx,
        n_threads=args.n_threads, n_batch=512, verbose=False,
    )
    logger.info("Loaded")

    preds, labels, datasets_list, latencies, raw = [], [], [], [], []
    unknown = 0
    t_start = time.time()

    for i, sample in enumerate(ds):
        prompt = COT_PROMPT.format(premise=sample["doc"], hypothesis=sample["claim"])
        t0 = time.time()
        try:
            out = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_tokens,
                temperature=0.0,
            )
            text = out["choices"][0]["message"]["content"]
        except Exception as e:
            text = f"ERROR: {e}"
        latencies.append(time.time() - t0)

        pred = parse_cot(text)
        if pred < 0:
            unknown += 1
        preds.append(pred)
        labels.append(int(sample["label"]))
        datasets_list.append(sample["dataset"])
        raw.append(text[:80])

        if (i + 1) % args.log_every == 0:
            elapsed = time.time() - t_start
            ba = compute_ba(preds, labels)
            eta = (len(ds) - i - 1) * elapsed / (i + 1) / 60
            logger.info("[%d/%d] BA=%.4f unk=%d %.0fms/sample ETA=%.1fmin",
                        i + 1, len(ds), ba, unknown,
                        1000 * elapsed / (i + 1), eta)

    # Per-dataset
    by_ds = defaultdict(lambda: ([], []))
    for p_, l_, d_ in zip(preds, labels, datasets_list, strict=True):
        by_ds[d_][0].append(p_)
        by_ds[d_][1].append(l_)
    per_ds_metrics = {
        ds: {"samples": len(l_), "balanced_accuracy": compute_ba(p_, l_)}
        for ds, (p_, l_) in by_ds.items()
    }

    total = time.time() - t_start
    results = {
        "model": args.model,
        "prompt_style": "chain-of-thought (1-2 sentences + ANSWER)",
        "samples": len(ds),
        "global_balanced_accuracy": compute_ba(preds, labels),
        "per_dataset": per_ds_metrics,
        "unknown_predictions": unknown,
        "total_time_seconds": total,
        "p50_latency_ms": 1000 * sorted(latencies)[len(latencies)//2],
        "p99_latency_ms": 1000 * sorted(latencies)[int(len(latencies)*0.99)],
        "sample_responses": raw[:20],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))

    logger.info("=" * 60)
    logger.info("Global BA: %.4f", results["global_balanced_accuracy"])
    logger.info("Unknown:   %d (%.1f%%)", unknown, 100*unknown/len(ds))
    logger.info("Time:      %.1fmin (%.0fms/sample)",
                total/60, 1000*total/len(ds))
    logger.info("=" * 60)
    for n, m in sorted(per_ds_metrics.items()):
        logger.info("  %-20s %5d  %.4f", n, m["samples"], m["balanced_accuracy"])


if __name__ == "__main__":
    main()
