# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — LLM-AggreFact Benchmark (Factual Consistency)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Evaluate NLI model on LLM-AggreFact — the standard factual consistency
benchmark aggregating 11 datasets across summarization, RAG, and
grounding tasks.

This is the benchmark used by the official leaderboard at
https://llm-aggrefact.github.io/. Published top scores (balanced acc):

    Bespoke-MiniCheck-7B    77.4%
    Claude-3.5 Sonnet       77.2%
    FactCG-DeBERTa-L        75.6%   (0.4B — our weight class)
    MiniCheck-Flan-T5-L     75.0%   (0.8B)
    HHEM-2.1                71.8%

The dataset is gated on HuggingFace. Authenticate first::

    export HF_TOKEN=hf_...
    # or: huggingface-cli login

Usage::

    python -m benchmarks.aggrefact_eval
    python -m benchmarks.aggrefact_eval --model training/output/deberta-v3-base-hallucination
    python -m benchmarks.aggrefact_eval --threshold 0.6
    python -m benchmarks.aggrefact_eval --sweep   # find optimal threshold

Mapping: NLI entailment probability > threshold → supported (1), else → not supported (0).
Metric: balanced accuracy per dataset, then macro-averaged (same as leaderboard).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from benchmarks._common import RESULTS_DIR, add_common_args, save_results

logger = logging.getLogger("DirectorAI.Benchmark.AggreFact")

AGGREFACT_DATASETS = [
    "AggreFact-CNN",
    "AggreFact-XSum",
    "TofuEval-MediaS",
    "TofuEval-MeetB",
    "Wice",
    "Reveal",
    "ClaimVerify",
    "FactCheck-GPT",
    "ExpertQA",
    "Lfqa",
    "RAGTruth",
]

# Published reference scores (balanced accuracy %) from the leaderboard
REFERENCE_SCORES = {
    "Bespoke-MiniCheck-7B": 77.4,
    "Claude-3.5-Sonnet": 77.2,
    "Granite-Guardian-3.3-8B": 76.5,
    "FactCG-DeBERTa-L (0.4B)": 75.6,
    "MiniCheck-Flan-T5-L (0.8B)": 75.0,
    "Llama-3.3-70B": 74.5,
    "HHEM-2.1": 71.8,
}


@dataclass
class AggreFactMetrics:
    """Per-dataset and aggregate balanced accuracy."""

    per_dataset: dict[str, dict] = field(default_factory=dict)
    threshold: float = 0.5
    inference_times: list[float] = field(default_factory=list, repr=False)

    @property
    def avg_balanced_acc(self) -> float:
        accs = [d["balanced_acc"] for d in self.per_dataset.values() if d["total"] > 0]
        return float(np.mean(accs)) if accs else 0.0

    @property
    def total_samples(self) -> int:
        return sum(d["total"] for d in self.per_dataset.values())

    @property
    def avg_latency_ms(self) -> float:
        if not self.inference_times:
            return 0.0
        return float(np.mean(self.inference_times)) * 1000

    def to_dict(self) -> dict:
        return {
            "avg_balanced_accuracy": round(self.avg_balanced_acc, 4),
            "avg_balanced_accuracy_pct": round(self.avg_balanced_acc * 100, 1),
            "threshold": self.threshold,
            "total_samples": self.total_samples,
            "per_dataset": {
                k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                for k, v in self.per_dataset.items()
            },
            "latency_ms_avg": round(self.avg_latency_ms, 2),
        }


class _BinaryNLIPredictor:
    """NLI model wrapped for binary factual consistency scoring.

    Returns entailment probability as the "supported" score.
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
        self._torch = torch
        self._num_labels = self.model.config.num_labels
        logger.info(
            "Model loaded on %s (%s, %d labels)",
            self.device, self.model_name, self._num_labels,
        )

    def score(self, premise: str, hypothesis: str) -> float:
        """Return P(supported) as factual consistency score in [0, 1].

        2-class models (FactCG): label1 = supported.
        3-class models (DeBERTa-mnli): label0 = entailment.
        """
        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt", truncation=True, max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
        probs = self._torch.softmax(logits, dim=1).cpu().numpy()[0]
        if self._num_labels == 2:
            return float(probs[1])
        return float(probs[0])


def _load_aggrefact(max_samples: int | None = None) -> list[dict]:
    """Load LLM-AggreFact test split. Requires HF authentication."""
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    logger.info("Loading LLM-AggreFact (gated dataset)...")
    ds = load_dataset("lytang/LLM-AggreFact", split="test", token=token)
    rows = list(ds)
    if max_samples:
        rows = rows[:max_samples]
    logger.info("Loaded %d samples across %d datasets", len(rows), len(set(r["dataset"] for r in rows)))
    return rows


def run_aggrefact_benchmark(
    threshold: float = 0.5,
    max_samples: int | None = None,
    model_name: str | None = None,
) -> AggreFactMetrics:
    predictor = _BinaryNLIPredictor(model_name=model_name)
    rows = _load_aggrefact(max_samples)

    # Collect predictions grouped by dataset
    by_dataset: dict[str, list[tuple[int, float]]] = {}
    metrics = AggreFactMetrics(threshold=threshold)

    for row in rows:
        doc = row.get("doc", "")
        claim = row.get("claim", "")
        label = row.get("label")
        ds_name = row.get("dataset", "unknown")

        if label is None or not doc or not claim:
            continue

        t0 = time.perf_counter()
        ent_prob = predictor.score(doc, claim)
        metrics.inference_times.append(time.perf_counter() - t0)

        if ds_name not in by_dataset:
            by_dataset[ds_name] = []
        by_dataset[ds_name].append((int(label), ent_prob))

    # Compute balanced accuracy per dataset
    for ds_name in sorted(by_dataset.keys()):
        pairs = by_dataset[ds_name]
        y_true = [p[0] for p in pairs]
        y_scores = [p[1] for p in pairs]
        y_pred = [1 if s >= threshold else 0 for s in y_scores]
        ba = balanced_accuracy_score(y_true, y_pred)
        n_pos = sum(y_true)
        n_neg = len(y_true) - n_pos
        metrics.per_dataset[ds_name] = {
            "total": len(pairs),
            "positive": n_pos,
            "negative": n_neg,
            "balanced_acc": float(ba),
        }

    return metrics


def sweep_thresholds(
    max_samples: int | None = None,
    model_name: str | None = None,
) -> tuple[float, AggreFactMetrics]:
    """Find the threshold that maximises average balanced accuracy."""
    predictor = _BinaryNLIPredictor(model_name=model_name)
    rows = _load_aggrefact(max_samples)

    by_dataset: dict[str, list[tuple[int, float]]] = {}
    inference_times: list[float] = []

    for row in rows:
        doc = row.get("doc", "")
        claim = row.get("claim", "")
        label = row.get("label")
        ds_name = row.get("dataset", "unknown")
        if label is None or not doc or not claim:
            continue
        t0 = time.perf_counter()
        ent_prob = predictor.score(doc, claim)
        inference_times.append(time.perf_counter() - t0)
        if ds_name not in by_dataset:
            by_dataset[ds_name] = []
        by_dataset[ds_name].append((int(label), ent_prob))

    best_thresh, best_avg = 0.5, 0.0
    for thresh_int in range(10, 91):
        thresh = thresh_int / 100.0
        accs = []
        for pairs in by_dataset.values():
            y_true = [p[0] for p in pairs]
            y_pred = [1 if p[1] >= thresh else 0 for p in pairs]
            accs.append(balanced_accuracy_score(y_true, y_pred))
        avg = float(np.mean(accs))
        if avg > best_avg:
            best_avg = avg
            best_thresh = thresh

    # Build final metrics at best threshold
    metrics = AggreFactMetrics(threshold=best_thresh)
    metrics.inference_times = inference_times
    for ds_name in sorted(by_dataset.keys()):
        pairs = by_dataset[ds_name]
        y_true = [p[0] for p in pairs]
        y_pred = [1 if p[1] >= best_thresh else 0 for p in pairs]
        ba = balanced_accuracy_score(y_true, y_pred)
        metrics.per_dataset[ds_name] = {
            "total": len(pairs),
            "positive": sum(y_true),
            "negative": len(y_true) - sum(y_true),
            "balanced_acc": float(ba),
        }

    return best_thresh, metrics


def _print_aggrefact_results(m: AggreFactMetrics, model_label: str = "") -> None:
    title = "LLM-AggreFact — Factual Consistency Benchmark"
    if model_label:
        title += f" ({model_label})"
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")
    print(f"  Threshold:  {m.threshold:.2f}")
    print(f"  Samples:    {m.total_samples}")
    print(f"  Avg Bal Acc: {m.avg_balanced_acc:.1%}")
    if m.inference_times:
        print(f"  Latency:    {m.avg_latency_ms:.1f} ms avg")
    print()

    print(f"  {'Dataset':<20} {'N':>5} {'Pos':>5} {'Neg':>5} {'Bal Acc':>9}")
    print(f"  {'-' * 50}")
    for ds_name, d in sorted(m.per_dataset.items()):
        print(f"  {ds_name:<20} {d['total']:>5} {d['positive']:>5} {d['negative']:>5} {d['balanced_acc']:>8.1%}")
    print()

    # Comparison with published scores
    our_pct = m.avg_balanced_acc * 100
    print(f"  {'Model':<30} {'Bal Acc':>8}  {'vs Ours':>8}")
    print(f"  {'-' * 50}")
    inserted = False
    for ref_name, ref_score in sorted(REFERENCE_SCORES.items(), key=lambda x: -x[1]):
        if not inserted and our_pct >= ref_score:
            print(f"  {'>>> OURS <<<':<30} {our_pct:>7.1f}%")
            inserted = True
        print(f"  {ref_name:<30} {ref_score:>7.1f}%  {our_pct - ref_score:>+7.1f}")
    if not inserted:
        print(f"  {'>>> OURS <<<':<30} {our_pct:>7.1f}%")

    print(f"{'=' * 72}")


# ── Pytest ─────────────────────────────────────────────────────────

@pytest.mark.slow
def test_aggrefact_sample():
    """Smoke test on a small sample (requires HF_TOKEN)."""
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("HF_TOKEN required for gated LLM-AggreFact dataset")
    m = run_aggrefact_benchmark(max_samples=100)
    _print_aggrefact_results(m)
    assert m.avg_balanced_acc > 0.50


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="LLM-AggreFact factual consistency benchmark")
    add_common_args(parser)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Entailment probability threshold (default: 0.5)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep thresholds 0.10-0.90 to find optimal")
    args = parser.parse_args()

    if args.sweep:
        best_thresh, m = sweep_thresholds(
            max_samples=args.max_samples, model_name=args.model
        )
        print(f"\nOptimal threshold: {best_thresh:.2f}")
        _print_aggrefact_results(m, args.model or "default")
    else:
        m = run_aggrefact_benchmark(
            threshold=args.threshold,
            max_samples=args.max_samples,
            model_name=args.model,
        )
        _print_aggrefact_results(m, args.model or "default")

    # Derive output filename from model name to avoid overwriting
    model_tag = (args.model or "default").replace("/", "_").replace("\\", "_")
    outfile = f"aggrefact_{model_tag}.json"
    save_results(
        {"benchmark": "LLM-AggreFact", "model": args.model or "default", **m.to_dict()},
        outfile,
    )
