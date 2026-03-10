# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Legal Domain Benchmark
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Evaluate Director-AI on legal domain factual consistency.

Datasets:
  - ContractNLI: NLI on contract clauses — determines if a hypothesis
    about a contract clause is entailed/contradicted/neutral.
  - CUAD (via RAGBench): Contract Understanding Atticus Dataset — RAG
    pipeline outputs with hallucination labels on legal documents.

Uses the legal profile (threshold=0.68, w_logic=0.6, w_fact=0.4).

Usage::

    python -m benchmarks.legal_eval
    python -m benchmarks.legal_eval --dataset contractnli --max-samples 500
    python -m benchmarks.legal_eval --dataset cuad --nli
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from benchmarks._common import save_results
from benchmarks.medical_eval import DomainMetrics, _print_domain_results

logger = logging.getLogger("DirectorAI.Benchmark.Legal")

CONTRACTNLI_HF_ID = "kiddothe2b/contract-nli"


def _load_contractnli(split: str = "test", max_samples: int | None = None) -> list[dict]:
    """Load ContractNLI dataset."""
    from datasets import load_dataset

    for hf_id in [CONTRACTNLI_HF_ID, "law-ai/contract-nli"]:
        try:
            ds = load_dataset(hf_id, split=split, trust_remote_code=True)
            rows = list(ds)
            if max_samples:
                rows = rows[:max_samples]
            logger.info("Loaded ContractNLI from %s: %d samples", hf_id, len(rows))
            return rows
        except Exception as exc:
            logger.debug("Could not load %s: %s", hf_id, exc)
    raise RuntimeError(
        "ContractNLI not available. "
        "Try: pip install datasets && export HF_TOKEN=..."
    )


def _load_cuad_ragbench(max_samples: int | None = None) -> list[dict]:
    """Load CUAD subset from RAGBench (rungalileo/ragbench)."""
    from datasets import load_dataset

    ds = load_dataset("rungalileo/ragbench", "cuad", split="test", trust_remote_code=True)
    rows = list(ds)
    if max_samples:
        rows = rows[:max_samples]
    logger.info("Loaded RAGBench CUAD: %d samples", len(rows))
    return rows


def run_contractnli(
    max_samples: int | None = None,
    threshold: float = 0.5,
    model_name: str | None = None,
) -> DomainMetrics:
    """Evaluate NLI accuracy on ContractNLI (binary: entailment vs contradiction)."""
    from benchmarks.aggrefact_eval import _BinaryNLIPredictor

    predictor = _BinaryNLIPredictor(model_name=model_name)
    rows = _load_contractnli(max_samples=max_samples)
    metrics = DomainMetrics(domain="legal", dataset="ContractNLI")

    y_true, y_pred = [], []
    for row in rows:
        premise = row.get("premise", "") or row.get("context", "")
        hypothesis = row.get("hypothesis", "") or row.get("statement", "")
        label_raw = row.get("label")
        if not premise or not hypothesis or label_raw is None:
            continue

        # Map to binary: entailment → 1, contradiction → 0, skip neutral
        if isinstance(label_raw, str):
            if label_raw.lower() == "neutral":
                continue
            is_supported = label_raw.lower() == "entailment"
        else:
            if int(label_raw) == 1:
                continue
            is_supported = int(label_raw) == 0

        t0 = time.perf_counter()
        ent_prob = predictor.score(premise, hypothesis)
        metrics.inference_times.append(time.perf_counter() - t0)

        predicted_supported = ent_prob >= threshold
        y_true.append(1 if is_supported else 0)
        y_pred.append(1 if predicted_supported else 0)
        metrics.total += 1

    for yt, yp in zip(y_true, y_pred, strict=False):
        is_hallucination = yt == 0
        flagged = yp == 0
        if is_hallucination and flagged:
            metrics.true_positives += 1
        elif not is_hallucination and flagged:
            metrics.false_positives += 1
        elif not is_hallucination and not flagged:
            metrics.true_negatives += 1
        else:
            metrics.false_negatives += 1

    if y_true:
        metrics.scores = [balanced_accuracy_score(y_true, y_pred)]
    return metrics


def run_cuad_guardrail(
    max_samples: int | None = None,
    threshold: float = 0.68,
    use_nli: bool = False,
    nli_model: str | None = None,
) -> DomainMetrics:
    """Evaluate Director-AI guardrail on CUAD via RAGBench."""
    from director_ai.core.scorer import CoherenceScorer

    scorer = CoherenceScorer(
        threshold=threshold,
        use_nli=use_nli,
        nli_model=nli_model,
        w_logic=0.6,
        w_fact=0.4,
    )
    rows = _load_cuad_ragbench(max_samples=max_samples)
    metrics = DomainMetrics(domain="legal", dataset="CUAD-RAGBench")

    for row in rows:
        context = row.get("context", "") or ""
        response = row.get("response", "") or ""
        is_hallucinated = bool(row.get("label", 0))

        if not context or not response:
            continue

        t0 = time.perf_counter()
        approved, score = scorer.review(context, response)
        metrics.inference_times.append(time.perf_counter() - t0)
        metrics.total += 1

        flagged = not approved
        if is_hallucinated and flagged:
            metrics.true_positives += 1
        elif not is_hallucinated and flagged:
            metrics.false_positives += 1
        elif not is_hallucinated and not flagged:
            metrics.true_negatives += 1
        else:
            metrics.false_negatives += 1

    return metrics


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Director-AI legal domain benchmark")
    parser.add_argument(
        "--dataset", choices=["contractnli", "cuad", "all"], default="all",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.68)
    parser.add_argument("--nli", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    results = {}

    if args.dataset in ("contractnli", "all"):
        try:
            m = run_contractnli(
                max_samples=args.max_samples,
                threshold=args.threshold,
                model_name=args.model,
            )
            _print_domain_results(m)
            results["contractnli"] = m.to_dict()
        except Exception as exc:
            logger.error("ContractNLI failed: %s", exc)

    if args.dataset in ("cuad", "all"):
        try:
            m = run_cuad_guardrail(
                max_samples=args.max_samples,
                threshold=args.threshold,
                use_nli=args.nli,
                nli_model=args.model,
            )
            _print_domain_results(m)
            results["cuad"] = m.to_dict()
        except Exception as exc:
            logger.error("CUAD failed: %s", exc)

    save_results(
        {"benchmark": "legal_domain", **results},
        "legal_eval.json",
    )
