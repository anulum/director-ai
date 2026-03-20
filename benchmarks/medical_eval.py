# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Medical Domain Benchmark

"""Evaluate Director-AI on medical domain factual consistency.

Datasets:
  - MedNLI (physionet): clinical NLI with expert-annotated entailment/
    neutral/contradiction labels on MIMIC-III clinical notes.
  - PubMedQA: biomedical yes/no/maybe question answering with context.

Uses the medical profile (threshold=0.75, w_logic=0.5, w_fact=0.5).

Usage::

    python -m benchmarks.medical_eval
    python -m benchmarks.medical_eval --dataset mednli --max-samples 500
    python -m benchmarks.medical_eval --dataset pubmedqa --nli
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
)

from benchmarks._common import save_results

logger = logging.getLogger("DirectorAI.Benchmark.Medical")

MEDNLI_HF_ID = "medhalt/mednli"
PUBMEDQA_HF_ID = "qiaojin/PubMedQA"


@dataclass
class DomainMetrics:
    """Binary classification metrics for domain guardrail evaluation."""

    domain: str = ""
    dataset: str = ""
    total: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    scores: list[float] = field(default_factory=list, repr=False)
    inference_times: list[float] = field(default_factory=list, repr=False)

    @property
    def catch_rate(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def fpr(self) -> float:
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.catch_rate
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return (
            float(np.mean(self.inference_times)) * 1000 if self.inference_times else 0.0
        )

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "dataset": self.dataset,
            "total": self.total,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "catch_rate": round(self.catch_rate, 4),
            "precision": round(self.precision, 4),
            "fpr": round(self.fpr, 4),
            "f1": round(self.f1, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


def _load_mednli(split: str = "test", max_samples: int | None = None) -> list[dict]:
    """Load MedNLI dataset. Falls back to alternative HF sources."""
    from datasets import load_dataset

    for hf_id in [MEDNLI_HF_ID, "bigbio/mednli"]:
        try:
            ds = load_dataset(hf_id, split=split, trust_remote_code=False)
            rows = list(ds)
            if max_samples:
                rows = rows[:max_samples]
            logger.info("Loaded MedNLI from %s: %d samples", hf_id, len(rows))
            return rows
        except Exception as exc:
            logger.debug("Could not load %s: %s", hf_id, exc)
    raise RuntimeError("MedNLI not available. Install 'datasets' and check HF access.")


def _load_pubmedqa(max_samples: int | None = None) -> list[dict]:
    """Load PubMedQA labeled subset."""
    from datasets import load_dataset

    ds = load_dataset(
        PUBMEDQA_HF_ID,
        "pqa_labeled",
        split="train",
        trust_remote_code=False,
    )
    rows = list(ds)
    if max_samples:
        rows = rows[:max_samples]
    logger.info("Loaded PubMedQA: %d samples", len(rows))
    return rows


def run_mednli_nli(
    max_samples: int | None = None,
    threshold: float = 0.5,
    model_name: str | None = None,
) -> DomainMetrics:
    """Evaluate raw NLI accuracy on MedNLI using the AggreFact predictor pattern."""
    from benchmarks.aggrefact_eval import _BinaryNLIPredictor

    predictor = _BinaryNLIPredictor(model_name=model_name)
    rows = _load_mednli(max_samples=max_samples)
    metrics = DomainMetrics(domain="medical", dataset="MedNLI")

    y_true, y_pred = [], []
    for row in rows:
        premise = row.get("premise") or row.get("sentence1", "")
        hypothesis = row.get("hypothesis") or row.get("sentence2", "")
        label_raw = row.get("label")
        if not premise or not hypothesis or label_raw is None:
            continue

        # MedNLI labels: entailment=0, neutral=1, contradiction=2
        # Map to binary: entailment → supported (1), contradiction → not-supported (0)
        if isinstance(label_raw, str):
            is_supported = label_raw.lower() == "entailment"
        else:
            is_supported = int(label_raw) == 0
        # Skip neutral for binary evaluation
        if isinstance(label_raw, str) and label_raw.lower() == "neutral":
            continue
        if isinstance(label_raw, int) and label_raw == 1:
            continue

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

    metrics.scores = [balanced_accuracy_score(y_true, y_pred)] if y_true else []
    return metrics


def run_pubmedqa_guardrail(
    max_samples: int | None = None,
    threshold: float = 0.75,
    use_nli: bool = False,
    nli_model: str | None = None,
) -> DomainMetrics:
    """Evaluate Director-AI guardrail on PubMedQA with the medical profile."""
    import torch

    from director_ai.core.scorer import CoherenceScorer

    scorer = CoherenceScorer(
        threshold=threshold,
        use_nli=use_nli,
        nli_model=nli_model,
        nli_device="cuda" if torch.cuda.is_available() else "cpu",
        w_logic=0.5,
        w_fact=0.5,
    )
    rows = _load_pubmedqa(max_samples=max_samples)
    metrics = DomainMetrics(domain="medical", dataset="PubMedQA")

    for row in rows:
        contexts = row.get("context", {})
        if isinstance(contexts, dict):
            context_text = " ".join(contexts.get("contexts", []))
        else:
            context_text = str(contexts)
        long_answer = row.get("long_answer", "")
        final_decision = row.get("final_decision", "")

        if not context_text or not long_answer:
            continue

        # PubMedQA: yes → factually grounded, no/maybe → potentially wrong
        is_hallucination = final_decision.lower() != "yes"

        t0 = time.perf_counter()
        approved, score = scorer.review(context_text, long_answer)
        metrics.inference_times.append(time.perf_counter() - t0)
        metrics.total += 1

        flagged = not approved
        if is_hallucination and flagged:
            metrics.true_positives += 1
        elif not is_hallucination and flagged:
            metrics.false_positives += 1
        elif not is_hallucination and not flagged:
            metrics.true_negatives += 1
        else:
            metrics.false_negatives += 1

    return metrics


def _print_domain_results(m: DomainMetrics) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {m.domain.upper()} Domain — {m.dataset}")
    print(f"{'=' * 60}")
    print(f"  Samples:    {m.total}")
    print(f"  Catch rate: {m.catch_rate:.1%}")
    print(f"  Precision:  {m.precision:.1%}")
    print(f"  FPR:        {m.fpr:.1%}")
    print(f"  F1:         {m.f1:.1%}")
    if m.inference_times:
        print(f"  Latency:    {m.avg_latency_ms:.1f} ms avg")
    print(
        f"  TP={m.true_positives}  FP={m.false_positives}  "
        f"TN={m.true_negatives}  FN={m.false_negatives}",
    )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Director-AI medical domain benchmark")
    parser.add_argument(
        "--dataset",
        choices=["mednli", "pubmedqa", "all"],
        default="all",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--nli", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    results = {}

    if args.dataset in ("mednli", "all"):
        try:
            m = run_mednli_nli(
                max_samples=args.max_samples,
                threshold=args.threshold,
                model_name=args.model,
            )
            _print_domain_results(m)
            results["mednli"] = m.to_dict()
        except Exception as exc:
            logger.error("MedNLI failed: %s", exc)

    if args.dataset in ("pubmedqa", "all"):
        try:
            m = run_pubmedqa_guardrail(
                max_samples=args.max_samples,
                threshold=args.threshold,
                use_nli=args.nli,
                nli_model=args.model,
            )
            _print_domain_results(m)
            results["pubmedqa"] = m.to_dict()
        except Exception as exc:
            logger.error("PubMedQA failed: %s", exc)

    save_results(
        {"benchmark": "medical_domain", **results},
        "medical_eval.json",
    )
