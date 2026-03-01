# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — End-to-End Guardrail Benchmark
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
End-to-end benchmark measuring Director-AI as a guardrail on real
hallucination datasets (HaluEval QA/summarization/dialogue, TruthfulQA).

Unlike component-level NLI benchmarks (aggrefact_eval, mnli_eval),
this evaluates the **full pipeline**: scorer + threshold + evidence +
fallback + soft warning zone — the stack a real user deploys.

Metrics:
- Hallucination catch rate (recall): % of hallucinated outputs flagged
- False-positive rate: % of correct outputs wrongly halted
- Precision: of flagged outputs, what % were actual hallucinations
- F1: harmonic mean of precision and recall
- Latency: per-sample wall clock (ms)
- Evidence coverage: % of rejections that include evidence chunks
- Warning rate: % of outputs in the soft warning zone
- Fallback rate: % of halts recovered by fallback mode

Usage::

    python -m benchmarks.e2e_eval
    python -m benchmarks.e2e_eval --max-samples 200 --fallback disclaimer
    python -m benchmarks.e2e_eval --sweep-thresholds
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from benchmarks._common import save_results
from benchmarks.halueval_eval import _download_task_data, _extract_pairs

logger = logging.getLogger("DirectorAI.Benchmark.E2E")


@dataclass
class E2ESample:
    """Single evaluation sample with ground truth and predictions."""

    task: str
    context: str
    response: str
    is_hallucinated: bool
    coherence_score: float = 0.0
    approved: bool = True
    warning: bool = False
    fallback_used: bool = False
    has_evidence: bool = False
    evidence_chunks: int = 0
    latency_ms: float = 0.0


@dataclass
class E2EMetrics:
    """Aggregated end-to-end benchmark metrics."""

    samples: list[E2ESample] = field(default_factory=list, repr=False)
    threshold: float = 0.5
    soft_limit: float = 0.6
    fallback_mode: str | None = None

    @property
    def total(self) -> int:
        return len(self.samples)

    @property
    def tp(self) -> int:
        return sum(1 for s in self.samples if s.is_hallucinated and not s.approved)

    @property
    def fp(self) -> int:
        return sum(1 for s in self.samples if not s.is_hallucinated and not s.approved)

    @property
    def tn(self) -> int:
        return sum(1 for s in self.samples if not s.is_hallucinated and s.approved)

    @property
    def fn(self) -> int:
        return sum(1 for s in self.samples if s.is_hallucinated and s.approved)

    @property
    def catch_rate(self) -> float:
        """Hallucination recall: % of hallucinations caught."""
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def false_positive_rate(self) -> float:
        """% of correct outputs wrongly halted."""
        denom = self.fp + self.tn
        return self.fp / denom if denom > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.catch_rate
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total > 0 else 0.0

    @property
    def warning_rate(self) -> float:
        approved = [s for s in self.samples if s.approved]
        if not approved:
            return 0.0
        return sum(1 for s in approved if s.warning) / len(approved)

    @property
    def fallback_rate(self) -> float:
        halted = [s for s in self.samples if not s.approved or s.fallback_used]
        if not halted:
            return 0.0
        return sum(1 for s in halted if s.fallback_used) / len(halted)

    @property
    def evidence_coverage(self) -> float:
        rejected = [s for s in self.samples if not s.approved]
        if not rejected:
            return 0.0
        return sum(1 for s in rejected if s.has_evidence) / len(rejected)

    @property
    def avg_latency_ms(self) -> float:
        times = [s.latency_ms for s in self.samples]
        return float(np.mean(times)) if times else 0.0

    @property
    def p95_latency_ms(self) -> float:
        times = [s.latency_ms for s in self.samples]
        return float(np.percentile(times, 95)) if times else 0.0

    def per_task(self) -> dict[str, dict]:
        """Breakdown by task (qa, summarization, dialogue)."""
        tasks: dict[str, list[E2ESample]] = {}
        for s in self.samples:
            tasks.setdefault(s.task, []).append(s)

        result = {}
        for task_name, task_samples in sorted(tasks.items()):
            tp = sum(1 for s in task_samples if s.is_hallucinated and not s.approved)
            fp = sum(
                1 for s in task_samples if not s.is_hallucinated and not s.approved
            )
            tn = sum(1 for s in task_samples if not s.is_hallucinated and s.approved)
            fn = sum(1 for s in task_samples if s.is_hallucinated and s.approved)
            catch = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * prec * catch / (prec + catch) if (prec + catch) > 0 else 0.0
            result[task_name] = {
                "total": len(task_samples),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "catch_rate": round(catch, 4),
                "false_positive_rate": round(fpr, 4),
                "precision": round(prec, 4),
                "f1": round(f1, 4),
                "avg_latency_ms": round(
                    float(np.mean([s.latency_ms for s in task_samples])), 2
                ),
            }
        return result

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "threshold": self.threshold,
            "soft_limit": self.soft_limit,
            "fallback_mode": self.fallback_mode,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "catch_rate": round(self.catch_rate, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "precision": round(self.precision, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "warning_rate": round(self.warning_rate, 4),
            "fallback_rate": round(self.fallback_rate, 4),
            "evidence_coverage": round(self.evidence_coverage, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "per_task": self.per_task(),
        }


def run_e2e_benchmark(
    tasks: list[str] | None = None,
    max_samples_per_task: int | None = None,
    threshold: float = 0.5,
    soft_limit: float = 0.6,
    use_nli: bool = False,
    nli_model: str | None = None,
    fallback: str | None = None,
) -> E2EMetrics:
    """Run end-to-end guardrail benchmark on HaluEval.

    Parameters
    ----------
    tasks : which HaluEval tasks to run (default: all three).
    max_samples_per_task : limit per task for quick runs.
    threshold : coherence threshold (hard_limit).
    soft_limit : soft warning zone upper bound.
    use_nli : enable NLI model (requires torch + transformers).
    nli_model : HuggingFace model ID for NLI.
    fallback : None, "retrieval", or "disclaimer".
    """
    from director_ai.core.scorer import CoherenceScorer
    from director_ai.core.vector_store import VectorGroundTruthStore

    if tasks is None:
        tasks = ["qa", "summarization", "dialogue"]

    store = VectorGroundTruthStore(auto_index=True)
    scorer = CoherenceScorer(
        threshold=threshold,
        soft_limit=soft_limit,
        use_nli=use_nli,
        ground_truth_store=store,
        nli_model=nli_model,
    )

    metrics = E2EMetrics(
        threshold=threshold,
        soft_limit=soft_limit,
        fallback_mode=fallback,
    )

    for task in tasks:
        try:
            samples = _download_task_data(task)
        except (ImportError, OSError, ValueError, KeyError) as e:
            logger.warning("Could not load HaluEval %s: %s", task, e)
            continue

        if max_samples_per_task:
            samples = samples[:max_samples_per_task]

        for sample_data in samples:
            pairs = _extract_pairs(task, sample_data)
            for context, response, is_hallucinated in pairs:
                if not context or not response:
                    continue

                # Ingest context into vector store for this sample
                store.ingest([context])

                t0 = time.perf_counter()
                approved, score = scorer.review(context, response)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                has_evidence = score.evidence is not None
                n_chunks = len(score.evidence.chunks) if score.evidence else 0

                # Simulate fallback behavior
                fallback_used = False
                if not approved and fallback:
                    fallback_used = True
                    # In fallback mode, the halt is "recovered"
                    # but we still count it as a detection for metrics

                e2e_sample = E2ESample(
                    task=task,
                    context=context[:200],
                    response=response[:200],
                    is_hallucinated=is_hallucinated,
                    coherence_score=score.score,
                    approved=approved,
                    warning=score.warning,
                    fallback_used=fallback_used,
                    has_evidence=has_evidence,
                    evidence_chunks=n_chunks,
                    latency_ms=elapsed_ms,
                )
                metrics.samples.append(e2e_sample)

    return metrics


def sweep_thresholds(
    tasks: list[str] | None = None,
    max_samples_per_task: int | None = None,
    use_nli: bool = False,
    nli_model: str | None = None,
) -> list[dict]:
    """Sweep threshold from 0.3 to 0.8 and report metrics at each point.

    Returns a list of dicts with threshold, catch_rate, fpr, precision, f1.
    Useful for plotting ROC-like curves.
    """
    from director_ai.core.scorer import CoherenceScorer
    from director_ai.core.vector_store import VectorGroundTruthStore

    if tasks is None:
        tasks = ["qa", "summarization", "dialogue"]

    store = VectorGroundTruthStore(auto_index=True)

    # Pre-score all samples once, then sweep threshold
    raw_scores: list[tuple[str, bool, float]] = []  # (task, is_halluc, score)

    scorer = CoherenceScorer(
        threshold=0.1,
        use_nli=use_nli,
        ground_truth_store=store,
        nli_model=nli_model,
    )

    for task in tasks:
        try:
            samples = _download_task_data(task)
        except (ImportError, OSError, ValueError, KeyError):
            continue

        if max_samples_per_task:
            samples = samples[:max_samples_per_task]

        for sample_data in samples:
            pairs = _extract_pairs(task, sample_data)
            for context, response, is_hallucinated in pairs:
                if not context or not response:
                    continue
                store.ingest([context])
                _, score = scorer.review(context, response)
                raw_scores.append((task, is_hallucinated, score.score))

    results = []
    for thresh_pct in range(30, 81, 5):
        thresh = thresh_pct / 100.0
        tp = sum(1 for _, h, s in raw_scores if h and s < thresh)
        fp = sum(1 for _, h, s in raw_scores if not h and s < thresh)
        tn = sum(1 for _, h, s in raw_scores if not h and s >= thresh)
        fn = sum(1 for _, h, s in raw_scores if h and s >= thresh)
        catch = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * catch / (prec + catch) if (prec + catch) > 0 else 0.0
        results.append(
            {
                "threshold": thresh,
                "catch_rate": round(catch, 4),
                "false_positive_rate": round(fpr, 4),
                "precision": round(prec, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )

    return results


def print_e2e_results(m: E2EMetrics) -> None:
    """Pretty-print end-to-end benchmark results."""
    print(f"\n{'=' * 72}")
    print("  Director-AI End-to-End Guardrail Benchmark")
    print(f"{'=' * 72}")
    print(f"  Samples:           {m.total}")
    print(f"  Threshold:         {m.threshold}")
    print(f"  Soft limit:        {m.soft_limit}")
    print(f"  Fallback:          {m.fallback_mode or 'none'}")
    print()
    caught = f"{m.tp}/{m.tp + m.fn}"
    halted = f"{m.fp}/{m.fp + m.tn}"
    print(f"  Catch rate:        {m.catch_rate:.1%} ({caught} hallucinations caught)")
    print(f"  False positive:    {m.false_positive_rate:.1%} ({halted} correct halted)")
    print(f"  Precision:         {m.precision:.1%}")
    print(f"  F1:                {m.f1:.1%}")
    print(f"  Accuracy:          {m.accuracy:.1%}")
    print()
    print(f"  Warning rate:      {m.warning_rate:.1%}")
    print(f"  Fallback rate:     {m.fallback_rate:.1%}")
    print(f"  Evidence coverage: {m.evidence_coverage:.1%}")
    print()
    print(f"  Latency avg:       {m.avg_latency_ms:.1f} ms")
    print(f"  Latency p95:       {m.p95_latency_ms:.1f} ms")
    print()

    per_task = m.per_task()
    cols = f"{'Task':<15} {'N':>5} {'Catch':>7} {'FPR':>7} {'Prec':>7}"
    hdr = f"  {cols} {'F1':>7} {'Lat':>8}"
    print(hdr)
    print(f"  {'-' * 60}")
    for task_name, d in per_task.items():
        print(
            f"  {task_name:<15} {d['total']:>5} "
            f"{d['catch_rate']:>6.1%} {d['false_positive_rate']:>6.1%} "
            f"{d['precision']:>6.1%} {d['f1']:>6.1%} "
            f"{d['avg_latency_ms']:>7.1f}ms"
        )
    print(f"{'=' * 72}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Director-AI end-to-end guardrail benchmark"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per task (default: all)",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--soft-limit", type=float, default=0.6)
    parser.add_argument("--nli", action="store_true", help="Enable NLI model")
    parser.add_argument("--nli-model", type=str, default=None)
    parser.add_argument(
        "--fallback",
        type=str,
        default=None,
        choices=["retrieval", "disclaimer"],
    )
    parser.add_argument(
        "--sweep-thresholds",
        action="store_true",
        help="Sweep thresholds and print table",
    )
    args = parser.parse_args()

    if args.sweep_thresholds:
        results = sweep_thresholds(
            max_samples_per_task=args.max_samples,
            use_nli=args.nli,
            nli_model=args.nli_model,
        )
        print(f"\n{'Threshold':>10} {'Catch':>8} {'FPR':>8} {'Prec':>8} {'F1':>8}")
        print("-" * 46)
        for r in results:
            print(
                f"{r['threshold']:>10.2f} {r['catch_rate']:>7.1%} "
                f"{r['false_positive_rate']:>7.1%} {r['precision']:>7.1%} "
                f"{r['f1']:>7.1%}"
            )
        save_results(
            {"benchmark": "E2E-Sweep", "results": results},
            "e2e_threshold_sweep.json",
        )
    else:
        m = run_e2e_benchmark(
            max_samples_per_task=args.max_samples,
            threshold=args.threshold,
            soft_limit=args.soft_limit,
            use_nli=args.nli,
            nli_model=args.nli_model,
            fallback=args.fallback,
        )
        print_e2e_results(m)
        save_results(
            {"benchmark": "E2E-Guardrail", **m.to_dict()},
            "e2e_guardrail.json",
        )
