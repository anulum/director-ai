# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Long-Context Model Evaluation

"""Evaluate long-context NLI models on AggreFact WITHOUT chunking.

The current pipeline chunks documents at 512 tokens and scores
cross-products. A model with native >512 context could skip chunking
entirely, eliminating chunk-boundary artifacts.

Candidate models (sorted by max_length):

    Model                                   max_len  params  notes
    ────────────────────────────────────────────────────────────────
    yaxili96/FactCG-DeBERTa-v3-Large        512      0.4B    current baseline
    MoritzLaurer/deberta-v3-large-zeroshot   512      0.4B    3-class NLI
    lytang/MiniCheck-DeBERTa-L              512      0.4B    2-class grounding
    MoritzLaurer/DeBERTa-v3-large-mnli      512      0.4B    3-class NLI
    allenai/longformer-large-4096           4096     0.4B    sparse attention
    google/bigbird-roberta-large            4096     0.4B    sparse + random
    MoritzLaurer/ModernBERT-large-zeroshot  8192     0.4B    failed in v3.8
    google/long-t5-tglobal-large            16384    0.8B    encoder-decoder

Usage::

    # Score all samples with a long-context model (no chunking)
    python -m benchmarks.long_context_eval \\
        --model allenai/longformer-large-4096 \\
        --max-length 4096 \\
        --save-scores long_context_scores.json

    # Analyze cached scores
    python -m benchmarks.long_context_eval \\
        --load-scores long_context_scores.json \\
        --sweep

Requires HF_TOKEN for the gated AggreFact dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

logger = logging.getLogger("DirectorAI.Benchmark.LongContext")

CANDIDATES = {
    "allenai/longformer-large-4096": {
        "max_len": 4096,
        "params": "0.4B",
        "attn": "sparse",
    },
    "google/bigbird-roberta-large": {
        "max_len": 4096,
        "params": "0.4B",
        "attn": "sparse+random",
    },
    "MoritzLaurer/ModernBERT-large-zeroshot-v2.0": {
        "max_len": 8192,
        "params": "0.4B",
        "attn": "full",
    },
}

BASELINE_BA = 75.86  # FactCG-DeBERTa-L with SummaC chunking


def _load_aggrefact():
    """Load LLM-AggreFact test split from HuggingFace."""
    try:
        from datasets import load_dataset

        ds = load_dataset("lytang/LLM-AggreFact", split="test")
        return list(ds)
    except Exception as e:
        jsonl = Path(__file__).parent / "aggrefact_test.jsonl"
        if jsonl.exists():
            with open(jsonl) as f:
                return [json.loads(line) for line in f if line.strip()]
        raise RuntimeError(f"Cannot load AggreFact: {e}") from e


def _score_no_chunking(
    model_name: str, max_length: int, device: str | None, samples: list
) -> list[dict]:
    """Score all samples with a single forward pass per (doc, claim) — no chunking."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Loading %s (max_length=%d)", model_name, max_length)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    num_labels = model.config.num_labels
    logger.info("Model loaded: %d labels, device=%s", num_labels, device)

    results = []
    for i, sample in enumerate(samples):
        doc = sample.get("doc", "")
        claim = sample.get("claim", "")
        label = sample.get("label", 0)
        dataset = sample.get("dataset", "unknown")

        t0 = time.monotonic()
        inputs = tokenizer(
            doc,
            claim,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # 2-class: P(supported) = probs[1]; 3-class: P(entailment) = probs[-1]
        score = float(probs[1]) if num_labels == 2 else float(probs[-1])

        elapsed = time.monotonic() - t0
        results.append(
            {
                "dataset": dataset,
                "label": label,
                "score": round(score, 6),
                "latency_ms": round(elapsed * 1000, 1),
            }
        )

        if (i + 1) % 500 == 0:
            logger.info("Scored %d/%d samples", i + 1, len(samples))

    return results


def _compute_ba(scores: list[dict], threshold: float) -> dict:
    """Compute per-dataset and macro BA at a given threshold."""
    from collections import defaultdict

    by_ds: dict[str, list] = defaultdict(list)
    for s in scores:
        by_ds[s["dataset"]].append(s)

    per_ds = {}
    for ds, items in sorted(by_ds.items()):
        y_true = [it["label"] for it in items]
        y_pred = [int(it["score"] >= threshold) for it in items]
        ba = balanced_accuracy_score(y_true, y_pred)
        per_ds[ds] = {"ba": round(ba * 100, 1), "n": len(items), "threshold": threshold}

    macro_ba = np.mean([v["ba"] for v in per_ds.values()])
    return {"macro_ba": round(float(macro_ba), 2), "per_dataset": per_ds}


def _sweep(scores: list[dict]) -> dict:
    """Sweep thresholds 0.10-0.90 and find optimal."""
    best_ba = 0.0
    best_t = 0.5
    for t_int in range(10, 91):
        t = t_int / 100.0
        result = _compute_ba(scores, t)
        if result["macro_ba"] > best_ba:
            best_ba = result["macro_ba"]
            best_t = t

    best_result = _compute_ba(scores, best_t)
    return {
        "best_threshold": best_t,
        "best_macro_ba": best_ba,
        "baseline_ba": BASELINE_BA,
        "delta": round(best_ba - BASELINE_BA, 2),
        "per_dataset": best_result["per_dataset"],
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Long-context NLI model evaluation")
    parser.add_argument("--model", type=str, help="HuggingFace model ID")
    parser.add_argument(
        "--max-length", type=int, default=4096, help="Max token length (default: 4096)"
    )
    parser.add_argument("--device", type=str, default=None, help="torch device")
    parser.add_argument("--save-scores", type=str, help="Save raw scores to JSON")
    parser.add_argument(
        "--load-scores", type=str, help="Load cached scores (skip inference)"
    )
    parser.add_argument("--sweep", action="store_true", help="Threshold sweep")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Single threshold (default: 0.5)"
    )
    args = parser.parse_args()

    if args.load_scores:
        with open(args.load_scores) as f:
            scores = json.load(f)
        logger.info("Loaded %d cached scores from %s", len(scores), args.load_scores)
    elif args.model:
        samples = _load_aggrefact()
        logger.info("Loaded %d AggreFact samples", len(samples))
        scores = _score_no_chunking(args.model, args.max_length, args.device, samples)
        if args.save_scores:
            Path(args.save_scores).write_text(json.dumps(scores, indent=2) + "\n")
            logger.info("Saved scores to %s", args.save_scores)
    else:
        print("Specify --model or --load-scores. Candidates:")
        print()
        for name, info in CANDIDATES.items():
            print(
                f"  {name:50s}  max={info['max_len']:>5d}  params={info['params']}  attn={info['attn']}"
            )
        print(f"\nBaseline (FactCG + chunking): {BASELINE_BA}% BA")
        sys.exit(0)

    if args.sweep:
        result = _sweep(scores)
        print(f"\nBest: {result['best_macro_ba']}% BA at t={result['best_threshold']}")
        print(f"Baseline: {BASELINE_BA}% BA (FactCG + chunking)")
        print(f"Delta: {result['delta']:+.2f}pp")
        print("\nPer-dataset:")
        for ds, info in sorted(result["per_dataset"].items()):
            print(f"  {ds:25s}: {info['ba']}% (n={info['n']})")

        out = Path("benchmarks/results/long_context_sweep.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
    else:
        result = _compute_ba(scores, args.threshold)
        print(f"\nMacro BA: {result['macro_ba']}% at t={args.threshold}")
        print(f"Baseline: {BASELINE_BA}% BA (FactCG + chunking)")
        print(f"Delta: {result['macro_ba'] - BASELINE_BA:+.2f}pp")

    avg_lat = np.mean([s.get("latency_ms", 0) for s in scores])
    print(f"Avg latency: {avg_lat:.1f} ms/sample")


if __name__ == "__main__":
    main()
