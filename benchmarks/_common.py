# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Shared Benchmark Utilities
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Shared model loading, prediction, and metrics for NLI benchmarks."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger("DirectorAI.Benchmark")

CACHE_DIR = Path(__file__).parent / ".cache"
RESULTS_DIR = Path(__file__).parent / "results"
LABEL_NAMES = ["entailment", "neutral", "contradiction"]


@dataclass
class NLIMetrics:
    """3-class NLI classification metrics."""

    y_true: list[int] = field(default_factory=list, repr=False)
    y_pred: list[int] = field(default_factory=list, repr=False)
    inference_times: list[float] = field(default_factory=list, repr=False)

    @property
    def total(self) -> int:
        return len(self.y_true)

    @property
    def accuracy(self) -> float:
        if not self.y_true:
            return 0.0
        return sum(t == p for t, p in zip(self.y_true, self.y_pred)) / len(self.y_true)

    def f1_per_class(self) -> dict[str, float]:
        from sklearn.metrics import f1_score
        f1s = f1_score(self.y_true, self.y_pred, average=None, labels=[0, 1, 2])
        return {name: float(f1s[i]) for i, name in enumerate(LABEL_NAMES)}

    def macro_f1(self) -> float:
        from sklearn.metrics import f1_score
        return float(f1_score(self.y_true, self.y_pred, average="macro", labels=[0, 1, 2]))

    def precision_recall_per_class(self) -> dict[str, dict[str, float]]:
        from sklearn.metrics import precision_score, recall_score
        prec = precision_score(self.y_true, self.y_pred, average=None, labels=[0, 1, 2])
        rec = recall_score(self.y_true, self.y_pred, average=None, labels=[0, 1, 2])
        return {
            name: {"precision": float(prec[i]), "recall": float(rec[i])}
            for i, name in enumerate(LABEL_NAMES)
        }

    @property
    def avg_latency_ms(self) -> float:
        if not self.inference_times:
            return 0.0
        return float(np.mean(self.inference_times)) * 1000

    @property
    def p95_latency_ms(self) -> float:
        if not self.inference_times:
            return 0.0
        return float(np.percentile(self.inference_times, 95)) * 1000

    def to_dict(self) -> dict:
        per_class = self.f1_per_class()
        pr = self.precision_recall_per_class()
        return {
            "total": self.total,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1(), 4),
            "per_class": {
                name: {
                    "f1": round(per_class[name], 4),
                    "precision": round(pr[name]["precision"], 4),
                    "recall": round(pr[name]["recall"], 4),
                }
                for name in LABEL_NAMES
            },
            "latency_ms_avg": round(self.avg_latency_ms, 2),
            "latency_ms_p95": round(self.p95_latency_ms, 2),
        }


class NLIPredictor:
    """Direct 3-class NLI predictor wrapping a HuggingFace model.

    Unlike NLIScorer (which maps to a [0,1] divergence float), this
    returns raw class predictions for standard NLI benchmark evaluation.
    """

    def __init__(self, model_name: str | None = None, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name or os.environ.get(
            "DIRECTOR_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        )
        logger.info("Loading NLI model: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = max_length
        logger.info("Model loaded on %s", self.device)

    def predict(self, premise: str, hypothesis: str) -> int:
        """Return predicted label: 0=entailment, 1=neutral, 2=contradiction."""
        import torch

        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt", truncation=True, max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return int(logits.argmax(dim=1).item())

    def predict_with_probs(self, premise: str, hypothesis: str) -> tuple[int, np.ndarray]:
        """Return (predicted_label, probability_array[3])."""
        import torch

        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt", truncation=True, max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return int(logits.argmax(dim=1).item()), probs


def print_nli_metrics(metrics: NLIMetrics, benchmark_name: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {benchmark_name}")
    print(f"{'=' * 65}")
    print(f"  Samples:    {metrics.total}")
    print(f"  Accuracy:   {metrics.accuracy:.1%}")
    print(f"  Macro F1:   {metrics.macro_f1():.4f}")
    per_class = metrics.f1_per_class()
    pr = metrics.precision_recall_per_class()
    for name in LABEL_NAMES:
        p, r = pr[name]["precision"], pr[name]["recall"]
        print(f"    {name:15s}  F1={per_class[name]:.4f}  P={p:.4f}  R={r:.4f}")
    if metrics.inference_times:
        print(f"  Latency:    {metrics.avg_latency_ms:.1f} ms avg, {metrics.p95_latency_ms:.1f} ms p95")
    print(f"{'=' * 65}")


def save_results(data: dict, filename: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nResults saved to {path}")
    return path


def add_common_args(parser) -> None:
    """Add --model and --max-samples to an argparse parser."""
    parser.add_argument("max_samples", nargs="?", type=int, default=None,
                        help="Limit evaluation samples (default: all)")
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model ID or local path")
