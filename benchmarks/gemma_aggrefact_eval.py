# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Gemma LLM-as-Judge AggreFact Benchmark
"""Run Gemma 4 as a hallucination judge on the AggreFact 29K benchmark.

Supports two backends:
- llama-cpp (local, GGUF quantized, Vulkan/HIP)
- transformers (cloud, full precision or HF-quantized)

Computes balanced accuracy globally and per-dataset, matching the
FactCG NLI benchmark protocol so results are directly comparable.

Usage::

    # Local (llama-cpp Vulkan)
    GGML_VK_VISIBLE_DEVICES=3 python benchmarks/gemma_aggrefact_eval.py \\
        --backend llama-cpp \\
        --model /tmp/gemma-models/google_gemma-4-E4B-it-Q4_K_M.gguf \\
        --max-samples 29320

    # Cloud (transformers)
    python benchmarks/gemma_aggrefact_eval.py \\
        --backend transformers \\
        --model google/gemma-4-E4B-it
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

from _judge_common import (
    compute_balanced_accuracy,
    parse_response,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are a fact-checking assistant. Decide if the CLAIM is fully supported by the CONTEXT.

CONTEXT:
{premise}

CLAIM:
{hypothesis}

Answer with exactly one word: SUPPORTED or NOT_SUPPORTED."""


class LlamaCppBackend:
    """Local llama-cpp-python backend (Vulkan/HIP)."""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = 2):
        from llama_cpp import Llama
        logger.info("Loading llama-cpp model: %s", model_path)
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=512,
            verbose=False,
            logits_all=False,
        )
        logger.info("Loaded")

    def judge(self, premise: str, hypothesis: str) -> str:
        prompt = JUDGE_PROMPT.format(premise=premise, hypothesis=hypothesis)
        out = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8,
            temperature=0.0,
        )
        return out["choices"][0]["message"]["content"]


class TransformersBackend:
    """HuggingFace transformers backend (cloud GPU)."""

    def __init__(self, model_id: str, dtype: str = "bfloat16", device: str = "cuda",
                 quantize: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info("Loading transformers model: %s (dtype=%s, quant=%s)",
                    model_id, dtype, quantize)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        kwargs = {"device_map": device}
        if quantize == "4bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = getattr(torch, dtype)

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        logger.info("Loaded on %s", self.model.device)

    def judge(self, premise: str, hypothesis: str) -> str:
        import torch
        prompt = JUDGE_PROMPT.format(premise=premise, hypothesis=hypothesis)
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
            tokenize=True,
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                inputs, max_new_tokens=8, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        return text


def load_aggrefact(max_samples: int | None = None):
    """Load LLM-AggreFact dataset (gated, needs HF_TOKEN)."""
    from datasets import load_dataset
    logger.info("Loading LLM-AggreFact...")
    ds = load_dataset("lytang/LLM-AggreFact", split="test")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    logger.info("Loaded %d samples", len(ds))
    return ds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["llama-cpp", "transformers"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output", type=str, default="benchmarks/results/gemma_aggrefact.json")
    p.add_argument("--n-ctx", type=int, default=4096)
    p.add_argument("--n-threads", type=int, default=2)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--quantize", type=str, default=None, choices=[None, "4bit"])
    p.add_argument("--log-every", type=int, default=100)
    args = p.parse_args()

    # Load dataset
    ds = load_aggrefact(args.max_samples)

    # Load model
    if args.backend == "llama-cpp":
        backend = LlamaCppBackend(args.model, args.n_ctx, args.n_threads)
    else:
        backend = TransformersBackend(args.model, args.dtype, "cuda", args.quantize)

    # Run inference
    preds = []
    labels = []
    datasets = []
    raw_responses = []
    latencies = []
    unknown_count = 0
    t_start = time.time()

    for i, sample in enumerate(ds):
        premise = sample.get("doc", sample.get("document", ""))
        hypothesis = sample.get("claim", sample.get("hypothesis", ""))
        # AggreFact label: 1 = supported, 0 = hallucination
        label = int(sample.get("label", sample.get("annotations", 0)))
        dataset_name = sample.get("dataset", "unknown")

        t0 = time.time()
        try:
            response = backend.judge(premise, hypothesis)
        except Exception as e:
            logger.warning("Sample %d failed: %s", i, e)
            response = "ERROR"
        elapsed = time.time() - t0

        pred = parse_response(response)
        if pred < 0:
            unknown_count += 1

        preds.append(pred)
        labels.append(label)
        datasets.append(dataset_name)
        raw_responses.append(response[:64])
        latencies.append(elapsed)

        if (i + 1) % args.log_every == 0:
            elapsed_total = time.time() - t_start
            current_ba = compute_balanced_accuracy(preds, labels)
            eta_s = (len(ds) - i - 1) * (elapsed_total / (i + 1))
            logger.info(
                "[%d/%d] BA=%.4f unk=%d %.0fms/sample ETA=%.1fmin",
                i + 1, len(ds), current_ba, unknown_count,
                1000 * elapsed_total / (i + 1), eta_s / 60,
            )

    # Final metrics
    global_ba = compute_balanced_accuracy(preds, labels)

    per_dataset = defaultdict(lambda: {"preds": [], "labels": []})
    for p_, l_, d_ in zip(preds, labels, datasets, strict=True):
        per_dataset[d_]["preds"].append(p_)
        per_dataset[d_]["labels"].append(l_)

    per_dataset_metrics = {}
    for ds_name, data in per_dataset.items():
        ba = compute_balanced_accuracy(data["preds"], data["labels"])
        per_dataset_metrics[ds_name] = {
            "samples": len(data["labels"]),
            "balanced_accuracy": ba,
        }

    total_time = time.time() - t_start
    results = {
        "model": args.model,
        "backend": args.backend,
        "samples": len(ds),
        "global_balanced_accuracy": global_ba,
        "per_dataset": per_dataset_metrics,
        "unknown_predictions": unknown_count,
        "total_time_seconds": total_time,
        "mean_latency_ms": 1000 * sum(latencies) / len(latencies) if latencies else 0,
        "p50_latency_ms": 1000 * sorted(latencies)[len(latencies)//2] if latencies else 0,
        "p99_latency_ms": 1000 * sorted(latencies)[int(len(latencies)*0.99)] if latencies else 0,
    }

    # Include per-sample predictions for ensemble analysis
    results["predictions"] = preds
    results["labels"] = labels
    results["datasets_per_sample"] = datasets

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    logger.info("=" * 60)
    logger.info("Global BA: %.4f", global_ba)
    logger.info("Unknown:   %d (%.1f%%)", unknown_count, 100*unknown_count/len(ds))
    logger.info("Time:      %.1fmin (%.0fms/sample)",
                total_time/60, 1000*total_time/len(ds))
    logger.info("=" * 60)
    for name, m in sorted(per_dataset_metrics.items()):
        logger.info("  %-20s %5d  %.4f", name, m["samples"], m["balanced_accuracy"])
    logger.info("Saved: %s", args.output)


if __name__ == "__main__":
    main()
