# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Gemma LLM-as-Judge with Logprob Scoring
"""Run Gemma 4 as a hallucination judge with continuous logprob scores.

Instead of taking just the argmax token (SUPPORTED/NOT_SUPPORTED), this
extracts the log-probability of each option from the next-token distribution.
The continuous score enables:

- Per-dataset threshold optimisation (like FactCG)
- Calibration analysis
- Ensemble with FactCG NLI scores

Score formula:
    P(SUPPORTED) = softmax(logits)[token_id_supported]
    score = P(SUPPORTED)  # in [0, 1], 0=hallucination, 1=supported
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are a fact-checking assistant. Decide if the CLAIM is fully supported by the CONTEXT.

CONTEXT:
{premise}

CLAIM:
{hypothesis}

Answer with exactly one word: SUPPORTED or NOT_SUPPORTED."""


def compute_balanced_accuracy(
    scores: list[float], labels: list[int], threshold: float = 0.5
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


def sweep_threshold(scores: list[float], labels: list[int]) -> tuple[float, float]:
    """Find optimal global threshold by sweeping 0..1."""
    best_t, best_ba = 0.5, 0.0
    for t in [0.05 * i for i in range(1, 20)]:
        ba = compute_balanced_accuracy(scores, labels, t)
        if ba > best_ba:
            best_ba, best_t = ba, t
    return best_t, best_ba


def per_dataset_sweep(
    scores: list[float], labels: list[int], datasets: list[str]
) -> dict:
    by_ds = defaultdict(lambda: ([], []))
    for s, lab, d in zip(scores, labels, datasets, strict=True):
        by_ds[d][0].append(s)
        by_ds[d][1].append(lab)

    out = {}
    avg_ba = 0.0
    n = 0
    for ds, (s_, l_) in by_ds.items():
        t, ba = sweep_threshold(s_, l_)
        out[ds] = {"samples": len(l_), "balanced_accuracy": ba, "threshold": t}
        avg_ba += ba
        n += 1
    return out, avg_ba / n if n else 0.0


class LlamaCppLogprobBackend:
    """llama-cpp backend with logprob extraction."""

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
            logits_all=True,  # required for logprobs
        )
        # Find token IDs for "SUPPORTED" and "NOT" (NOT_SUPPORTED starts with NOT)
        toks_supported = self.llm.tokenize(b" SUPPORTED", add_bos=False)
        toks_not = self.llm.tokenize(b" NOT", add_bos=False)
        logger.info("token ids: SUPPORTED=%s, NOT=%s", toks_supported, toks_not)
        self.tok_supported = toks_supported[0] if toks_supported else None
        self.tok_not = toks_not[0] if toks_not else None
        logger.info("Loaded")

    def judge(self, premise: str, hypothesis: str) -> tuple[float, str]:
        """Returns (score in [0,1], raw text). Score = P(SUPPORTED)."""
        prompt = JUDGE_PROMPT.format(premise=premise, hypothesis=hypothesis)
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
        score = None
        if logprobs and "content" in logprobs and len(logprobs["content"]) > 0:
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
            # Fallback: parse text
            t = text.upper()
            score = 0.0 if "NOT" in t else (1.0 if "SUPPORTED" in t else 0.5)
        return score, text


def load_aggrefact(max_samples: int | None = None):
    from datasets import load_dataset

    ds = load_dataset("lytang/LLM-AggreFact", split="test")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--output", type=str, default="benchmarks/results/gemma_logprob.json"
    )
    p.add_argument("--n-ctx", type=int, default=4096)
    p.add_argument("--n-threads", type=int, default=2)
    p.add_argument("--log-every", type=int, default=500)
    args = p.parse_args()

    ds = load_aggrefact(args.max_samples)
    logger.info("Samples: %d", len(ds))

    backend = LlamaCppLogprobBackend(args.model, args.n_ctx, args.n_threads)

    scores = []
    labels = []
    datasets_list = []
    raw = []
    latencies = []
    t_start = time.time()

    for i, sample in enumerate(ds):
        premise = sample["doc"]
        hypothesis = sample["claim"]
        label = int(sample["label"])
        dataset_name = sample["dataset"]

        t0 = time.time()
        try:
            score, text = backend.judge(premise, hypothesis)
        except Exception as e:
            logger.warning("Sample %d failed: %s", i, e)
            score, text = None, "ERROR"
        latencies.append(time.time() - t0)

        scores.append(score)
        labels.append(label)
        datasets_list.append(dataset_name)
        raw.append(text[:32])

        if (i + 1) % args.log_every == 0:
            elapsed = time.time() - t_start
            ba_05 = compute_balanced_accuracy(scores, labels, 0.5)
            eta = (len(ds) - i - 1) * elapsed / (i + 1) / 60
            logger.info(
                "[%d/%d] BA@0.5=%.4f %.0fms/sample ETA=%.1fmin",
                i + 1,
                len(ds),
                ba_05,
                1000 * elapsed / (i + 1),
                eta,
            )

    # Final analysis
    invalid = sum(1 for s in scores if s is None)

    ba_default = compute_balanced_accuracy(scores, labels, 0.5)
    best_t, ba_best = sweep_threshold(scores, labels)
    per_ds, avg_per_ds = per_dataset_sweep(scores, labels, datasets_list)

    total = time.time() - t_start
    results = {
        "model": args.model,
        "samples": len(ds),
        "global_balanced_accuracy_t05": ba_default,
        "global_balanced_accuracy_optimal": ba_best,
        "global_optimal_threshold": best_t,
        "per_dataset_avg_balanced_accuracy": avg_per_ds,
        "per_dataset": per_ds,
        "invalid_scores": invalid,
        "total_time_seconds": total,
        "p50_latency_ms": 1000 * sorted(latencies)[len(latencies) // 2],
        "p99_latency_ms": 1000 * sorted(latencies)[int(len(latencies) * 0.99)],
        "scores": scores,  # save for ensemble analysis
        "labels": labels,
        "datasets": datasets_list,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))

    logger.info("=" * 60)
    logger.info("Global BA @ t=0.5:    %.4f", ba_default)
    logger.info("Global BA optimal:    %.4f (t=%.2f)", ba_best, best_t)
    logger.info("Per-dataset average:  %.4f", avg_per_ds)
    logger.info("Invalid: %d (%.1f%%)", invalid, 100 * invalid / len(ds))
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
