# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Public hallucination detection competitor benchmarks
"""Run public hallucination detection classifiers on AggreFact 29 K.

Currently supports:

- ``vectara/hallucination_evaluation_model`` (HHEM-2.1, 184 M, classifier head)
- ``lytang/MiniCheck-Roberta-Large`` (1.4 GB, T5-style entailment classifier)

These are NLI / entailment classifiers, not generative judges. Each model
has its own input format. We adapt the prompt per backend so that the
output is comparable to our other AggreFact JSONs (same schema).

Output schema matches benchmarks/gemma_aggrefact_eval.py so the
sentinel_judge_analyzer.py can ensemble them with the LLM judges.

Usage::

    HIP_VISIBLE_DEVICES=4 python benchmarks/competitor_aggrefact.py \\
        --model vectara/hallucination_evaluation_model \\
        --output benchmarks/results/competitor_hhem_21.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def balanced_accuracy(preds: list[int], labels: list[int]) -> float:
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


# ── HHEM-2.1 (Vectara) ───────────────────────────────────────────────────


class HHEMBackend:
    """vectara/hallucination_evaluation_model — sequence classifier."""

    def __init__(self, model_id: str, max_length: int):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, torch_dtype=torch.float32, trust_remote_code=True
        )
        self.model.to("cuda")
        self.model.eval()
        self.max_length = max_length

    def score(self, premise: str, hypothesis: str) -> float:
        # HHEM expects ``premise<sep>hypothesis`` and returns P(consistent)
        text = f"{premise}<eos>{hypothesis}"
        with self.torch.no_grad():
            inputs = self.tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=self.max_length,
            ).to(self.model.device)
            logits = self.model(**inputs).logits
            return float(self.torch.sigmoid(logits[0, 0]).item())


# ── MiniCheck-Roberta-Large ──────────────────────────────────────────────


class MiniCheckBackend:
    """lytang/MiniCheck-Roberta-Large — Roberta entailment classifier."""

    def __init__(self, model_id: str, max_length: int):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to("cuda")
        self.model.eval()
        self.max_length = max_length

    def score(self, premise: str, hypothesis: str) -> float:
        with self.torch.no_grad():
            inputs = self.tokenizer(
                premise, hypothesis,
                return_tensors="pt", truncation=True, max_length=self.max_length,
            ).to(self.model.device)
            logits = self.model(**inputs).logits
            probs = self.torch.softmax(logits, dim=-1)
            # Roberta NLI label 0 = entailment / supported
            return float(probs[0, 0].item())


BACKENDS = {
    "vectara/hallucination_evaluation_model": HHEMBackend,
    "lytang/MiniCheck-Roberta-Large": MiniCheckBackend,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(BACKENDS.keys()))
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--output", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--log-every", type=int, default=500)
    args = p.parse_args()

    from datasets import load_dataset

    ds = load_dataset("lytang/LLM-AggreFact", split="test")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    logger.info("Samples: %d", len(ds))

    logger.info("Loading: %s", args.model)
    backend = BACKENDS[args.model](args.model, args.max_length)
    logger.info("Loaded")

    scores: list[float] = []
    preds: list[int] = []
    labels: list[int] = []
    datasets_list: list[str] = []
    latencies: list[float] = []
    t_start = time.time()

    for i, sample in enumerate(ds):
        premise = sample["doc"]
        hypothesis = sample["claim"]
        label = int(sample["label"])
        dataset_name = sample["dataset"]

        t0 = time.time()
        try:
            score = backend.score(premise, hypothesis)
            pred = 1 if score >= args.threshold else 0
        except Exception as e:
            logger.warning("Sample %d failed: %s", i, e)
            score, pred = -1.0, -1
        latencies.append(time.time() - t0)

        scores.append(score)
        preds.append(pred)
        labels.append(label)
        datasets_list.append(dataset_name)

        if (i + 1) % args.log_every == 0:
            elapsed = time.time() - t_start
            ba = balanced_accuracy(preds, labels)
            eta = (len(ds) - i - 1) * elapsed / (i + 1) / 60
            logger.info("[%d/%d] BA=%.4f %.0fms/sample ETA=%.1fmin",
                        i + 1, len(ds), ba, 1000 * elapsed / (i + 1), eta)

    by_ds: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for p_, l_, d_ in zip(preds, labels, datasets_list, strict=True):
        by_ds[d_][0].append(p_)
        by_ds[d_][1].append(l_)
    per_ds = {
        d: {"samples": len(l_), "balanced_accuracy": balanced_accuracy(p_, l_)}
        for d, (p_, l_) in by_ds.items()
    }

    total = time.time() - t_start
    results = {
        "model": args.model,
        "backend": "transformers-classifier",
        "samples": len(ds),
        "global_balanced_accuracy": balanced_accuracy(preds, labels),
        "per_dataset": per_ds,
        "scores": scores,
        "predictions": preds,
        "labels": labels,
        "datasets_per_sample": datasets_list,
        "threshold": args.threshold,
        "unknown_predictions": sum(1 for p in preds if p < 0),
        "total_time_seconds": total,
        "p50_latency_ms": 1000 * sorted(latencies)[len(latencies) // 2] if latencies else 0,
        "p99_latency_ms": (
            1000 * sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))

    logger.info("=" * 60)
    logger.info("Global BA: %.4f", results["global_balanced_accuracy"])
    logger.info("Time:      %.1fmin", total / 60)
    logger.info("=" * 60)
    for n, m in sorted(per_ds.items()):
        logger.info("  %-20s %5d  %.4f", n, m["samples"], m["balanced_accuracy"])


if __name__ == "__main__":
    main()
