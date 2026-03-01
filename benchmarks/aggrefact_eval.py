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
    python -m benchmarks.aggrefact_eval --model yaxili96/FactCG-DeBERTa-v3-Large
    python -m benchmarks.aggrefact_eval --threshold 0.6
    python -m benchmarks.aggrefact_eval --sweep

NLI entailment prob > threshold → supported (1), else → not supported (0).
Balanced accuracy per dataset, macro-averaged (same as leaderboard).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pytest
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from benchmarks._common import add_common_args, save_results

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
                k: {
                    kk: round(vv, 4) if isinstance(vv, float) else vv
                    for kk, vv in v.items()
                }
                for k, v in self.per_dataset.items()
            },
            "latency_ms_avg": round(self.avg_latency_ms, 2),
        }


_FACTCG_TEMPLATE = (
    '{text_a}\n\nChoose your answer: based on the paragraph above '
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    'I think the answer is '
)


def _chunk_source(text: str, max_tokens: int = 550) -> list[str]:
    """Split source document into sentence-level chunks (SummaC-style)."""
    import nltk
    try:
        sents = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        sents = nltk.sent_tokenize(text)

    chunks: list[str] = []
    chunk, chunk_len = "", 0
    for s in sents:
        s_len = len(s.split())
        if chunk and chunk_len + s_len > max_tokens:
            chunks.append(chunk)
            chunk, chunk_len = s, s_len
        else:
            chunk = f"{chunk}\n{s}".strip("\n") if chunk else s
            chunk_len += s_len
    if chunk:
        chunks.append(chunk)
    return chunks or [text]


class _BinaryNLIPredictor:
    """NLI model wrapped for binary factual consistency scoring.

    Returns entailment probability as the "supported" score.
    FactCG models use instruction template + SummaC-style source chunking.
    """

    def __init__(self, model_name: str | None = None, max_length: int = 2048):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name or os.environ.get(
            "DIRECTOR_NLI_MODEL", "yaxili96/FactCG-DeBERTa-v3-Large"
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
        self._is_factcg = "factcg" in self.model_name.lower()
        logger.info(
            "Model loaded on %s (%s, %d labels, factcg=%s)",
            self.device, self.model_name, self._num_labels, self._is_factcg,
        )

    def _score_single(self, premise: str, hypothesis: str) -> float:
        """Score a single (premise, hypothesis) pair."""
        if self._is_factcg:
            text = _FACTCG_TEMPLATE.format(text_a=premise, text_b=hypothesis)
            inputs = self.tokenizer(
                text,
                return_tensors="pt", truncation=True, max_length=self.max_length,
            )
        else:
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

    def _score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Batched forward pass — all pairs in one call."""
        if self._is_factcg:
            texts = [
                _FACTCG_TEMPLATE.format(text_a=p, text_b=h) for p, h in pairs
            ]
            inputs = self.tokenizer(
                texts, return_tensors="pt", truncation=True,
                padding=True, max_length=self.max_length,
            )
        else:
            premises = [p for p, _ in pairs]
            hypotheses = [h for _, h in pairs]
            inputs = self.tokenizer(
                premises, hypotheses, return_tensors="pt", truncation=True,
                padding=True, max_length=self.max_length,
            )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
        probs = self._torch.softmax(logits, dim=1).cpu().numpy()
        if self._num_labels == 2:
            return [float(row[1]) for row in probs]
        return [float(row[0]) for row in probs]

    def score(self, premise: str, hypothesis: str) -> float:
        """Return P(supported) with SummaC source chunking for FactCG.

        Splits premise into sentence chunks, scores each vs hypothesis,
        returns max (matching official FactCG evaluation).
        Chunks are batched into a single forward pass.
        """
        if not self._is_factcg:
            return self._score_single(premise, hypothesis)
        chunks = _chunk_source(premise)
        if len(chunks) == 1:
            return self._score_single(chunks[0], hypothesis)
        return max(self._score_batch([(c, hypothesis) for c in chunks]))


class _NLIScorerPredictor:
    """Wraps NLIScorer.score_chunked() for bidirectional chunking comparison."""

    def __init__(self, model_name: str | None = None):
        from director_ai.core.nli import NLIScorer

        self.scorer = NLIScorer(
            use_model=True,
            model_name=model_name or os.environ.get(
                "DIRECTOR_NLI_MODEL", "yaxili96/FactCG-DeBERTa-v3-Large"
            ),
        )
        logger.info("NLIScorerPredictor (bidirectional) ready")

    def score(self, premise: str, hypothesis: str) -> float:
        score, _ = self.scorer.score_chunked(premise, hypothesis)
        # score_chunked returns divergence [0,1]; convert to entailment prob
        return 1.0 - score


def _binary_class_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Precision/recall/F1 for both supported (1) and hallucination (0) classes."""
    labels = sorted(set(y_true) | set(y_pred))
    if len(labels) < 2:
        return {}
    prec = precision_score(y_true, y_pred, average=None, labels=[0, 1])
    rec = recall_score(y_true, y_pred, average=None, labels=[0, 1])
    f1 = f1_score(y_true, y_pred, average=None, labels=[0, 1])
    return {
        "hallucination_precision": float(prec[0]),
        "hallucination_recall": float(rec[0]),
        "hallucination_f1": float(f1[0]),
        "supported_precision": float(prec[1]),
        "supported_recall": float(rec[1]),
        "supported_f1": float(f1[1]),
    }


def _load_aggrefact(max_samples: int | None = None) -> list[dict]:
    """Load LLM-AggreFact test split. Requires HF authentication."""
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    logger.info("Loading LLM-AggreFact (gated dataset)...")
    ds = load_dataset("lytang/LLM-AggreFact", split="test", token=token)
    rows = list(ds)
    if max_samples:
        rows = rows[:max_samples]
    n_ds = len(set(r["dataset"] for r in rows))
    logger.info("Loaded %d samples across %d datasets", len(rows), n_ds)
    return rows


def run_aggrefact_benchmark(
    threshold: float = 0.5,
    max_samples: int | None = None,
    model_name: str | None = None,
    bidirectional: bool = False,
) -> AggreFactMetrics:
    if bidirectional:
        predictor = _NLIScorerPredictor(model_name=model_name)
    else:
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
            **_binary_class_metrics(y_true, y_pred),
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
            **_binary_class_metrics(y_true, y_pred),
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

    has_pr = any(
        "hallucination_recall" in d for d in m.per_dataset.values()
    )
    if has_pr:
        hdr = (
            f"  {'Dataset':<20} {'N':>5} {'BalAcc':>7}"
            f" {'H-Prec':>7} {'H-Rec':>7} {'H-F1':>7}"
        )
    else:
        hdr = (
            f"  {'Dataset':<20} {'N':>5} {'Pos':>5}"
            f" {'Neg':>5} {'Bal Acc':>9}"
        )
    print(hdr)
    print(f"  {'-' * len(hdr.strip())}")
    for ds_name, d in sorted(m.per_dataset.items()):
        if has_pr:
            print(
                f"  {ds_name:<20} {d['total']:>5}"
                f" {d['balanced_acc']:>6.1%}"
                f" {d.get('hallucination_precision', 0):>6.1%}"
                f" {d.get('hallucination_recall', 0):>6.1%}"
                f" {d.get('hallucination_f1', 0):>6.1%}"
            )
        else:
            print(
                f"  {ds_name:<20} {d['total']:>5}"
                f" {d['positive']:>5} {d['negative']:>5}"
                f" {d['balanced_acc']:>8.1%}"
            )
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

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="LLM-AggreFact factual consistency benchmark",
    )
    add_common_args(parser)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Entailment probability threshold (default: 0.5)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep thresholds 0.10-0.90 to find optimal")
    parser.add_argument("--bidirectional", action="store_true",
                        help="Use NLIScorer.score_chunked() bidirectional path")
    args = parser.parse_args()

    if args.sweep:
        best_thresh, m = sweep_thresholds(
            max_samples=args.max_samples, model_name=args.model
        )
        print(f"\nOptimal threshold: {best_thresh:.2f}")
        _print_aggrefact_results(m, args.model or "default")
    elif args.bidirectional:
        m_summac = run_aggrefact_benchmark(
            threshold=args.threshold,
            max_samples=args.max_samples,
            model_name=args.model,
            bidirectional=False,
        )
        _print_aggrefact_results(m_summac, "SummaC chunking")
        m_bidir = run_aggrefact_benchmark(
            threshold=args.threshold,
            max_samples=args.max_samples,
            model_name=args.model,
            bidirectional=True,
        )
        _print_aggrefact_results(m_bidir, "Bidirectional chunking")
        print(f"\n{'=' * 55}")
        print("  Delta: Bidirectional vs SummaC")
        print(f"{'=' * 55}")
        for ds in sorted(set(m_summac.per_dataset) | set(m_bidir.per_dataset)):
            s = m_summac.per_dataset.get(ds, {}).get("balanced_acc", 0)
            b = m_bidir.per_dataset.get(ds, {}).get("balanced_acc", 0)
            print(f"  {ds:<20} {s:.1%} → {b:.1%}  ({b - s:+.1%})")
        delta = m_bidir.avg_balanced_acc - m_summac.avg_balanced_acc
        print(f"\n  Overall: {m_summac.avg_balanced_acc:.1%} → "
              f"{m_bidir.avg_balanced_acc:.1%}  ({delta:+.1%})")
        print(f"{'=' * 55}")
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
