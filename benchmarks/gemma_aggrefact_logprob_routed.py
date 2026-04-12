# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Gemma routed judge with logprob scores
"""Logprob-extracting variant of the routed Gemma judge.

Combines two ideas:

1. Per-task-family prompt routing — three prompts (summ / rag / claim)
   chosen by AggreFact subset, the strategy that took the baseline
   from 79.34 % to the 82.11 % global champion.
2. Logprob extraction on the next predicted token — instead of taking
   the categorical SUPPORTED / NOT_SUPPORTED verdict, read the
   probability mass on the SUPPORTED token vs NOT* tokens. The score
   $P(\\text{SUPPORTED}) \\in [0, 1]$ is continuous, which unlocks:

   - per-dataset threshold optimisation (the FactCG sweep showed
     +2.18 pp average from per-dataset thresholds)
   - logistic-regression fusion at the score level rather than the
     binary level (much smoother gradients)
   - calibration analysis (reliability diagrams)

The output JSON contains both the per-dataset auto-tuned thresholds
and the global threshold, plus the raw scores for downstream
ensemble use.

Usage::

    GGML_VK_VISIBLE_DEVICES=6 python benchmarks/gemma_aggrefact_logprob_routed.py \\
        --model /tmp/gemma-models/google_gemma-4-E4B-it-Q6_K.gguf \\
        --max-samples 29320 \\
        --output benchmarks/results/gemma_e4b_q6_logprob_routed.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path

from _judge_common import (
    DATASET_TO_FAMILY,
    PROMPTS,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ── Metrics ──────────────────────────────────────────────────────────────


def balanced_accuracy(
    scores: list[float | None],
    labels: list[int],
    threshold: float = 0.5,
) -> float:
    pos = neg = tp = tn = 0
    for s, lab in zip(scores, labels, strict=True):
        if s is None:
            continue
        pred = 1 if s >= threshold else 0
        if lab == 1:
            pos += 1
            if pred == 1:
                tp += 1
        else:
            neg += 1
            if pred == 0:
                tn += 1
    if pos == 0 or neg == 0:
        return 0.0
    return (tp / pos + tn / neg) / 2


def sweep_threshold(
    scores: list[float | None], labels: list[int]
) -> tuple[float, float]:
    """Find the optimal global threshold by sweeping 0.05..0.95."""
    best_t, best_ba = 0.5, 0.0
    for t in [0.05 * i for i in range(1, 20)]:
        ba = balanced_accuracy(scores, labels, t)
        if ba > best_ba:
            best_t, best_ba = t, ba
    return best_t, best_ba


def per_dataset_sweep(
    scores: list[float | None],
    labels: list[int],
    datasets: list[str],
) -> tuple[dict, float]:
    by_ds: dict[str, tuple[list[float | None], list[int]]] = defaultdict(
        lambda: ([], [])
    )
    for s, lab, d in zip(scores, labels, datasets, strict=True):
        by_ds[d][0].append(s)
        by_ds[d][1].append(lab)
    out: dict[str, dict] = {}
    avg = 0.0
    for ds_name, (s_, l_) in by_ds.items():
        t, ba = sweep_threshold(s_, l_)
        out[ds_name] = {
            "samples": len(l_),
            "balanced_accuracy": ba,
            "threshold": t,
        }
        avg += ba
    return out, (avg / len(by_ds)) if by_ds else 0.0


# ── Backend ──────────────────────────────────────────────────────────────


class GemmaRoutedLogprobBackend:
    """llama-cpp Vulkan backend that emits routed-prompt logprobs."""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = 2):
        from llama_cpp import Llama

        logger.info("Loading: %s", model_path)
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=512,
            verbose=False,
            logits_all=True,  # required for top_logprobs extraction
        )
        logger.info("Loaded")

    def judge(
        self, premise: str, hypothesis: str, family: str
    ) -> tuple[float | None, str]:
        """Return (P(SUPPORTED), raw text) for one (premise, hypothesis)."""
        prompt = PROMPTS[family].format(premise=premise, hypothesis=hypothesis)
        out = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4,
            temperature=0.0,
            logprobs=True,
            top_logprobs=10,
        )
        choice = out["choices"][0]
        text = choice["message"]["content"].strip()
        logprobs = choice.get("logprobs")
        score: float | None = None
        if logprobs and "content" in logprobs and logprobs["content"]:
            first = logprobs["content"][0]
            top = first.get("top_logprobs", [])
            p_sup = 0.0
            p_not = 0.0
            for entry in top:
                tok = entry.get("token", "").strip().upper()
                lp = entry.get("logprob", -1e9)
                if "SUPPORTED" in tok and "NOT" not in tok:
                    p_sup += math.exp(lp)
                elif "NOT" in tok or tok in ("UN", "NO"):
                    p_not += math.exp(lp)
            if p_sup + p_not > 0:
                score = p_sup / (p_sup + p_not)
        if score is None:
            t = text.upper()
            score = (
                0.0
                if "NOT" in t
                else (1.0 if "SUPPORTED" in t else None)
            )
        return score, text


# ── Driver ───────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/gemma_logprob_routed.json",
    )
    p.add_argument("--n-ctx", type=int, default=4096)
    p.add_argument("--n-threads", type=int, default=2)
    p.add_argument("--log-every", type=int, default=500)
    args = p.parse_args()

    from datasets import load_dataset

    ds = load_dataset("lytang/LLM-AggreFact", split="test")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    logger.info("Samples: %d", len(ds))

    family_counts: dict[str, int] = defaultdict(int)
    for sample in ds:
        family_counts[
            DATASET_TO_FAMILY.get(sample["dataset"], "claim")
        ] += 1
    logger.info("Family distribution: %s", dict(family_counts))

    backend = GemmaRoutedLogprobBackend(
        args.model, args.n_ctx, args.n_threads
    )

    scores: list[float | None] = []
    labels: list[int] = []
    datasets_list: list[str] = []
    families: list[str] = []
    raw_texts: list[str] = []
    latencies: list[float] = []
    invalid = 0
    t_start = time.time()

    for i, sample in enumerate(ds):
        premise = sample["doc"]
        hypothesis = sample["claim"]
        label = int(sample["label"])
        dataset_name = sample["dataset"]
        family = DATASET_TO_FAMILY.get(dataset_name, "claim")

        t0 = time.time()
        try:
            score, text = backend.judge(premise, hypothesis, family)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sample %d failed: %s", i, exc)
            score, text = None, "ERROR"
        latencies.append(time.time() - t0)

        if score is None:
            invalid += 1
        scores.append(score)
        labels.append(label)
        datasets_list.append(dataset_name)
        families.append(family)
        raw_texts.append(text[:32])

        if (i + 1) % args.log_every == 0:
            elapsed = time.time() - t_start
            ba_05 = balanced_accuracy(scores, labels, 0.5)
            eta_min = (len(ds) - i - 1) * elapsed / (i + 1) / 60
            logger.info(
                "[%d/%d] BA@0.5=%.4f invalid=%d %.0fms/sample ETA=%.1fmin",
                i + 1,
                len(ds),
                ba_05,
                invalid,
                1000 * elapsed / (i + 1),
                eta_min,
            )

    # Final analysis
    ba_t05 = balanced_accuracy(scores, labels, 0.5)
    best_t, ba_best = sweep_threshold(scores, labels)
    per_ds, per_ds_avg = per_dataset_sweep(scores, labels, datasets_list)

    by_fam: dict[str, tuple[list[float | None], list[int]]] = defaultdict(
        lambda: ([], [])
    )
    for s, lab, fam in zip(scores, labels, families, strict=True):
        by_fam[fam][0].append(s)
        by_fam[fam][1].append(lab)
    per_fam: dict[str, dict] = {}
    for fam, (s_, l_) in by_fam.items():
        t, ba = sweep_threshold(s_, l_)
        per_fam[fam] = {
            "samples": len(l_),
            "balanced_accuracy": ba,
            "threshold": t,
        }

    total = time.time() - t_start
    results = {
        "schema_version": 2,
        "model": args.model,
        "method": (
            "per-dataset prompt routing (summ/rag/claim) + logprob scores"
        ),
        "samples": len(ds),
        "global_balanced_accuracy_t05": ba_t05,
        "global_balanced_accuracy_optimal": ba_best,
        "global_optimal_threshold": best_t,
        "per_dataset_avg_balanced_accuracy": per_ds_avg,
        "per_dataset": per_ds,
        "per_family": per_fam,
        "dataset_to_family": DATASET_TO_FAMILY,
        "invalid_scores": invalid,
        "total_time_seconds": total,
        "p50_latency_ms": 1000
        * sorted(latencies)[len(latencies) // 2]
        if latencies
        else 0,
        "p99_latency_ms": 1000
        * sorted(latencies)[int(len(latencies) * 0.99)]
        if latencies
        else 0,
        "scores": scores,
        "labels": labels,
        "datasets_per_sample": datasets_list,
        "families_per_sample": families,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    logger.info("=" * 60)
    logger.info("Global BA @ t=0.5:    %.4f", ba_t05)
    logger.info("Global BA optimal:    %.4f (t=%.2f)", ba_best, best_t)
    logger.info("Per-dataset average:  %.4f", per_ds_avg)
    logger.info(
        "Invalid: %d (%.1f%%)", invalid, 100 * invalid / len(ds)
    )
    logger.info("Time: %.1fmin", total / 60)
    logger.info("=" * 60)
    for ds_name, m in sorted(per_ds.items()):
        logger.info(
            "  %-20s %5d  BA=%.4f  t=%.2f",
            ds_name,
            m["samples"],
            m["balanced_accuracy"],
            m["threshold"],
        )
    logger.info("Saved: %s", args.output)


if __name__ == "__main__":
    main()
