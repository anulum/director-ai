# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RAGTruth example-level benchmark for the token detector

"""Measure the token span detector at the example level on RAGTruth.

Runs :class:`~director_ai.core.scoring.span_detector.HallucinationSpanDetector`
over the balanced ``wandb/RAGTruth-processed`` test split and reports the
example-level confusion matrix (a response is flagged when the detector returns
any hallucinated span), so the headline F1 / balanced-accuracy / FPR are grounded
in a runnable script rather than a quoted number. This is the production path —
the same class the guard loads — not the training-time evaluator.

    python benchmarks/ragtruth_token_detector_bench.py --max-samples 600
    DIRECTOR_SPAN_MODEL=/path/to/local python benchmarks/ragtruth_token_detector_bench.py
"""

from __future__ import annotations

import argparse
import logging
import os
import time

from benchmarks._common import save_results

logger = logging.getLogger("ragtruth_token_bench")


def _row_label(item: dict) -> bool:
    from benchmarks.ragtruth_eval import _row_label as rl

    return rl(item)


def run(max_samples: int | None = None) -> dict:
    from datasets import load_dataset

    from director_ai.core.scoring.span_detector import HallucinationSpanDetector

    model_id = os.environ.get(
        "DIRECTOR_SPAN_MODEL", "anulum/director-ragtruth-token-modernbert"
    )
    device = int(os.environ.get("DIRECTOR_SPAN_DEVICE", "-1"))
    threshold = float(os.environ.get("DIRECTOR_SPAN_THRESHOLD", "0.95"))
    min_tokens = int(os.environ.get("DIRECTOR_SPAN_MIN_TOKENS", "1"))
    max_length = int(os.environ.get("DIRECTOR_SPAN_MAX_LENGTH", "1024"))

    detector = HallucinationSpanDetector.from_pretrained(
        model_id,
        device=device,
        token_threshold=threshold,
        min_tokens=min_tokens,
        max_length=max_length,
    )

    ds = load_dataset("wandb/RAGTruth-processed", split="test")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    tp = fp = tn = fn = 0
    latencies: list[float] = []
    for item in ds:
        gold = _row_label(item)
        t0 = time.perf_counter()
        det = detector.detect(item.get("context", ""), item.get("output", ""))
        latencies.append((time.perf_counter() - t0) * 1000)
        flagged = det.hallucinated
        if gold and flagged:
            tp += 1
        elif gold:
            fn += 1
        elif flagged:
            fp += 1
        else:
            tn += 1

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    latencies.sort()
    result = {
        "benchmark": "ragtruth_token_detector",
        "model": model_id,
        "token_threshold": threshold,
        "min_tokens": min_tokens,
        "max_length": max_length,
        "n": tp + fp + tn + fn,
        "n_hallucinated": tp + fn,
        "n_grounded": tn + fp,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "balanced_accuracy": (tpr + tnr) / 2,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "latency_ms_p50": latencies[len(latencies) // 2] if latencies else 0.0,
        "latency_ms_mean": sum(latencies) / len(latencies) if latencies else 0.0,
        "reference": {"lettucedetect_example_f1": 0.7922, "nli_decompose_f1": 0.366},
    }
    return result


def _print(r: dict) -> None:
    print(f"\n=== RAGTruth token detector — example-level (n={r['n']}) ===")
    print(f"  model        : {r['model']}")
    print(f"  F1           : {r['f1']:.4f}")
    print(f"  precision    : {r['precision']:.4f}")
    print(f"  recall       : {r['recall']:.4f}")
    print(f"  balanced acc : {r['balanced_accuracy']:.4f}")
    print(f"  FPR          : {r['false_positive_rate']:.4f}")
    print(f"  latency p50  : {r['latency_ms_p50']:.1f} ms")
    print(
        f"  (LettuceDetect example F1 = {r['reference']['lettucedetect_example_f1']})"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="RAGTruth token-detector benchmark")
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()
    res = run(max_samples=args.max_samples)
    _print(res)
    save_results(res, "ragtruth_token_detector_results.json")
