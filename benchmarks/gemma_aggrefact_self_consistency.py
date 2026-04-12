# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Self-consistency variant of the routed Gemma judge
"""Self-consistency (Wang et al., 2022) on the routed Gemma judge.

Each sample is queried K times at sampling temperature T > 0; the final
verdict is the majority vote across the K samples. Wang et al. report
+5 to +10 pp accuracy gain on reasoning benchmarks from K=5 self-consistency.

This script tests whether test-time compute scaling pushes the routed
champion (currently 82.11 % global BA at K=1, T=0) by sampling at
non-zero temperature and voting.

Soft score: ``support_fraction = votes_supported / K``. Threshold-able.
Hard prediction: ``support_fraction >= 0.5``.

Usage::

    GGML_VK_VISIBLE_DEVICES=4 python benchmarks/gemma_aggrefact_self_consistency.py \\
        --model /tmp/gemma-models/google_gemma-4-E4B-it-Q6_K.gguf \\
        --max-samples 29320 \\
        --k 3 \\
        --temperature 0.4 \\
        --output benchmarks/results/gemma_e4b_q6_routed_sc3.json
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/gemma_routed_sc.json",
    )
    p.add_argument("--n-ctx", type=int, default=4096)
    p.add_argument("--n-threads", type=int, default=2)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of samples per claim for self-consistency",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature; 0.3-0.7 typical for self-consistency",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling cumulative probability",
    )
    args = p.parse_args()

    from datasets import load_dataset
    from llama_cpp import Llama

    ds = load_dataset("lytang/LLM-AggreFact", split="test")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    logger.info(
        "Samples: %d  K=%d  T=%.2f", len(ds), args.k, args.temperature
    )

    family_counts: dict[str, int] = defaultdict(int)
    for sample in ds:
        family_counts[
            DATASET_TO_FAMILY.get(sample["dataset"], "claim")
        ] += 1
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
    support_fractions: list[float | None] = []
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
        prompt = PROMPTS[family].format(
            premise=premise, hypothesis=hypothesis
        )

        votes_sup = 0
        votes_not = 0
        votes_unk = 0
        t0 = time.time()
        for _k in range(args.k):
            try:
                out = llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=8,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                text = out["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Sample %d k=%d failed: %s", i, _k, exc)
                text = "ERROR"
            v = parse_response(text)
            if v == 1:
                votes_sup += 1
            elif v == 0:
                votes_not += 1
            else:
                votes_unk += 1
        latencies.append(time.time() - t0)

        denom = votes_sup + votes_not
        if denom == 0:
            pred = -1
            support_fraction: float | None = None
            unknown += 1
        else:
            support_fraction = votes_sup / denom
            pred = 1 if support_fraction >= 0.5 else 0

        preds.append(pred)
        support_fractions.append(support_fraction)
        labels.append(label)
        datasets_list.append(dataset_name)
        families.append(family)

        if (i + 1) % args.log_every == 0:
            elapsed = time.time() - t_start
            ba = compute_balanced_accuracy(preds, labels)
            eta_min = (len(ds) - i - 1) * elapsed / (i + 1) / 60
            logger.info(
                "[%d/%d] BA=%.4f unk=%d %.0fms/sample ETA=%.1fmin",
                i + 1,
                len(ds),
                ba,
                unknown,
                1000 * elapsed / (i + 1),
                eta_min,
            )

    by_ds: dict[str, tuple[list[int], list[int]]] = defaultdict(
        lambda: ([], [])
    )
    for p_, l_, d_ in zip(preds, labels, datasets_list, strict=True):
        by_ds[d_][0].append(p_)
        by_ds[d_][1].append(l_)
    per_ds_metrics = {
        ds: {
            "samples": len(l_),
            "balanced_accuracy": compute_balanced_accuracy(p_, l_),
        }
        for ds, (p_, l_) in by_ds.items()
    }

    by_fam: dict[str, tuple[list[int], list[int]]] = defaultdict(
        lambda: ([], [])
    )
    for p_, l_, f_ in zip(preds, labels, families, strict=True):
        by_fam[f_][0].append(p_)
        by_fam[f_][1].append(l_)
    per_family_metrics = {
        f: {
            "samples": len(l_),
            "balanced_accuracy": compute_balanced_accuracy(p_, l_),
        }
        for f, (p_, l_) in by_fam.items()
    }

    total = time.time() - t_start
    results = {
        "schema_version": 2,
        "model": args.model,
        "method": (
            f"self-consistency K={args.k} T={args.temperature} on routed prompts"
        ),
        "samples": len(ds),
        "k": args.k,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "global_balanced_accuracy": compute_balanced_accuracy(
            preds, labels
        ),
        "per_dataset": per_ds_metrics,
        "per_family": per_family_metrics,
        "dataset_to_family": DATASET_TO_FAMILY,
        "unknown_predictions": unknown,
        "total_time_seconds": total,
        "p50_latency_ms": 1000
        * sorted(latencies)[len(latencies) // 2]
        if latencies
        else 0,
        "p99_latency_ms": 1000
        * sorted(latencies)[int(len(latencies) * 0.99)]
        if latencies
        else 0,
        "predictions": preds,
        "support_fractions": support_fractions,
        "labels": labels,
        "datasets_per_sample": datasets_list,
        "families_per_sample": families,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    logger.info("=" * 60)
    logger.info("K=%d T=%.2f", args.k, args.temperature)
    logger.info(
        "Global BA:    %.4f", results["global_balanced_accuracy"]
    )
    logger.info(
        "Unknown:      %d (%.1f%%)", unknown, 100 * unknown / len(ds)
    )
    logger.info("Time:         %.1fmin", total / 60)
    logger.info("=" * 60)
    for fam, m in sorted(per_family_metrics.items()):
        logger.info(
            "  %-8s %5d  %.4f",
            fam,
            m["samples"],
            m["balanced_accuracy"],
        )
    for n, m in sorted(per_ds_metrics.items()):
        logger.info(
            "  %-20s %5d  %.4f",
            n,
            m["samples"],
            m["balanced_accuracy"],
        )
    logger.info("Saved: %s", args.output)


if __name__ == "__main__":
    main()
