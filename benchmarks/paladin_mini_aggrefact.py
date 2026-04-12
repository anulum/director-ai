# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Paladin-mini (Qualifire) AggreFact baseline
"""Run Qualifire/context-grounding-paladin-mini as zero-shot judge on AggreFact 29 K.

Paladin-mini is a Phi-3 3.8 B grounding-tuned classifier that the Qualifire team
claims hits 79.31 % BA on AggreFact (per the 2026-04-11 Gemini Deep Research
audit). We reproduce it on the exact 29 320-sample lytang/LLM-AggreFact split
we use for everything else, with the same uniform prompt our other LLM-as-judge
benchmarks use, so the result is directly comparable.

Backend: transformers + ROCm (HIP_VISIBLE_DEVICES). Phi-3 has no architecture
quirks on RDNA2 — unlike gemma4 with bitsandbytes, this works in bf16 directly.

Output schema matches benchmarks/gemma_aggrefact_eval.py so the
sentinel_judge_analyser.py can load it as a third judge.

Usage::

    HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=2 \\
        python benchmarks/paladin_mini_aggrefact.py \\
        --max-samples 29320 \\
        --output benchmarks/results/paladin_mini.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

from _judge_common import compute_balanced_accuracy as balanced_accuracy
from _judge_common import parse_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are a fact-checking assistant. Decide if the CLAIM is fully supported by the CONTEXT.

CONTEXT:
{premise}

CLAIM:
{hypothesis}

Answer with exactly one word: SUPPORTED or NOT_SUPPORTED."""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qualifire/context-grounding-paladin-mini")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-input-tokens", type=int, default=4096)
    p.add_argument("--output", type=str, default="benchmarks/results/paladin_mini.json")
    p.add_argument("--log-every", type=int, default=200)
    args = p.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ds = load_dataset("lytang/LLM-AggreFact", split="test")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    logger.info("Samples: %d", len(ds))

    logger.info("Loading: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Loaded on %s", model.device)

    preds, labels, datasets_list, latencies = [], [], [], []
    unknown = 0
    t_start = time.time()

    for i, sample in enumerate(ds):
        premise = sample["doc"]
        hypothesis = sample["claim"]
        label = int(sample["label"])
        dataset_name = sample["dataset"]
        prompt = JUDGE_PROMPT.format(premise=premise, hypothesis=hypothesis)

        t0 = time.time()
        try:
            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
                truncation=True,
                max_length=args.max_input_tokens,
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    inputs,
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(out[0][inputs.shape[1] :], skip_special_tokens=True)
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

        if (i + 1) % args.log_every == 0:
            elapsed = time.time() - t_start
            ba = balanced_accuracy(preds, labels)
            eta = (len(ds) - i - 1) * elapsed / (i + 1) / 60
            logger.info(
                "[%d/%d] BA=%.4f unk=%d %.0fms/sample ETA=%.1fmin",
                i + 1,
                len(ds),
                ba,
                unknown,
                1000 * elapsed / (i + 1),
                eta,
            )

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
        "backend": "transformers",
        "samples": len(ds),
        "global_balanced_accuracy": balanced_accuracy(preds, labels),
        "per_dataset": per_ds,
        "predictions": preds,
        "labels": labels,
        "datasets_per_sample": datasets_list,
        "unknown_predictions": unknown,
        "total_time_seconds": total,
        "p50_latency_ms": 1000 * sorted(latencies)[len(latencies) // 2]
        if latencies
        else 0,
        "p99_latency_ms": (
            1000 * sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))

    logger.info("=" * 60)
    logger.info("Global BA: %.4f", results["global_balanced_accuracy"])
    logger.info("Unknown:   %d (%.1f%%)", unknown, 100 * unknown / len(ds))
    logger.info("Time:      %.1fmin", total / 60)
    logger.info("=" * 60)
    for n, m in sorted(per_ds.items()):
        logger.info("  %-20s %5d  %.4f", n, m["samples"], m["balanced_accuracy"])


if __name__ == "__main__":
    main()
