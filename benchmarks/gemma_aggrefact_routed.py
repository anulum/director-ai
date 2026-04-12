# SPDX-License-Identifier: AGPL-3.0-or-later
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
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

from _judge_common import (
    DATASET_TO_FAMILY,
    PROMPTS,
    compute_balanced_accuracy,
    parse_response,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output", type=str,
                   default="benchmarks/results/gemma_routed.json")
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

    # Distribution sanity check
    family_counts = defaultdict(int)
    for s in ds:
        family_counts[DATASET_TO_FAMILY.get(s["dataset"], "claim")] += 1
    logger.info("Family distribution: %s", dict(family_counts))

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
    families: list[str] = []
    latencies: list[float] = []
    unknown = 0
    t_start = time.time()

    for i, sample in enumerate(ds):
        premise = sample["doc"]
        hypothesis = sample["claim"]
        label = int(sample["label"])
        dataset_name = sample["dataset"]
        family = DATASET_TO_FAMILY.get(dataset_name, "claim")
        prompt = PROMPTS[family].format(premise=premise, hypothesis=hypothesis)

        t0 = time.time()
        try:
            out = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8,
                temperature=0.0,
            )
            text = out["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning("Sample %d failed: %s", i, e)
            text = "ERROR"
        latencies.append(time.time() - t0)

        pred = parse_response(text)
        if pred < 0:
            unknown += 1
        preds.append(pred)
        labels.append(label)
        datasets_list.append(dataset_name)
        families.append(family)

        if (i + 1) % args.log_every == 0:
            elapsed = time.time() - t_start
            ba = compute_balanced_accuracy(preds, labels)
            eta = (len(ds) - i - 1) * elapsed / (i + 1) / 60
            logger.info(
                "[%d/%d] BA=%.4f unk=%d %.0fms/sample ETA=%.1fmin",
                i + 1, len(ds), ba, unknown,
                1000 * elapsed / (i + 1), eta,
            )

    by_ds: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for p_, l_, d_ in zip(preds, labels, datasets_list, strict=True):
        by_ds[d_][0].append(p_)
        by_ds[d_][1].append(l_)
    per_ds_metrics = {
        ds: {"samples": len(l_), "balanced_accuracy": compute_balanced_accuracy(p_, l_)}
        for ds, (p_, l_) in by_ds.items()
    }

    # Per-family aggregate
    by_fam: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for p_, l_, f_ in zip(preds, labels, families, strict=True):
        by_fam[f_][0].append(p_)
        by_fam[f_][1].append(l_)
    per_family_metrics = {
        f: {"samples": len(l_), "balanced_accuracy": compute_balanced_accuracy(p_, l_)}
        for f, (p_, l_) in by_fam.items()
    }

    total = time.time() - t_start
    results = {
        "model": args.model,
        "method": "per-dataset prompt routing (summ/rag/claim families)",
        "samples": len(ds),
        "global_balanced_accuracy": compute_balanced_accuracy(preds, labels),
        "per_dataset": per_ds_metrics,
        "per_family": per_family_metrics,
        "dataset_to_family": DATASET_TO_FAMILY,
        "unknown_predictions": unknown,
        "total_time_seconds": total,
        "p50_latency_ms": 1000 * sorted(latencies)[len(latencies) // 2] if latencies else 0,
        "p99_latency_ms": (
            1000 * sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0
        ),
        "predictions": preds,
        "labels": labels,
        "datasets_per_sample": datasets_list,
        "families_per_sample": families,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))

    logger.info("=" * 60)
    logger.info("Global BA:    %.4f", results["global_balanced_accuracy"])
    logger.info("Unknown:      %d (%.1f%%)", unknown, 100 * unknown / len(ds))
    logger.info("Time:         %.1fmin", total / 60)
    logger.info("=" * 60)
    logger.info("Per-family:")
    for fam, m in sorted(per_family_metrics.items()):
        logger.info("  %-8s %5d  %.4f", fam, m["samples"], m["balanced_accuracy"])
    logger.info("Per-dataset:")
    for n, m in sorted(per_ds_metrics.items()):
        logger.info("  %-20s %5d  %.4f", n, m["samples"], m["balanced_accuracy"])
    logger.info("Saved: %s", args.output)


if __name__ == "__main__":
    main()
