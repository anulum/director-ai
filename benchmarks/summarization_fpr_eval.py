# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Summarization False-Positive Rate Benchmark
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
A/B benchmark comparing aggregation strategies on correct summaries.

HaluEval summarization correct (document, right_summary) pairs are long
documents that split into multiple NLI premise chunks — the exact
scenario where MIN inner aggregation reduces false positives vs MAX.

For each correct summary, score twice through CoherenceScorer:
  1. max-max  (old default): _fact_inner_agg="max", _fact_outer_agg="max"
  2. min-mean (new profile): _fact_inner_agg="min", _fact_outer_agg="mean"

A sample is a false positive if ``approved == False``.

Usage::

    python -m benchmarks.summarization_fpr_eval
    python -m benchmarks.summarization_fpr_eval 200
    python -m benchmarks.summarization_fpr_eval --threshold 0.50
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pytest

from benchmarks._common import add_common_args, save_results
from benchmarks.halueval_eval import _download_task_data, _extract_pairs

logger = logging.getLogger("DirectorAI.Benchmark.SummarizationFPR")


@dataclass
class SummarizationFPRMetrics:
    total: int = 0
    fp_max_max: int = 0
    fp_min_mean: int = 0
    scores_max_max: list[float] = field(default_factory=list, repr=False)
    scores_min_mean: list[float] = field(default_factory=list, repr=False)
    inference_times: list[float] = field(default_factory=list, repr=False)

    @property
    def fpr_max_max(self) -> float:
        return self.fp_max_max / max(self.total, 1)

    @property
    def fpr_min_mean(self) -> float:
        return self.fp_min_mean / max(self.total, 1)

    @property
    def fpr_reduction_pct(self) -> float:
        if self.fpr_max_max == 0:
            return 0.0
        return (self.fpr_max_max - self.fpr_min_mean) / self.fpr_max_max * 100

    @property
    def avg_score_max_max(self) -> float:
        return float(np.mean(self.scores_max_max)) if self.scores_max_max else 0.0

    @property
    def avg_score_min_mean(self) -> float:
        return float(np.mean(self.scores_min_mean)) if self.scores_min_mean else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return (
            float(np.mean(self.inference_times)) * 1000 if self.inference_times else 0.0
        )

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "fp_max_max": self.fp_max_max,
            "fp_min_mean": self.fp_min_mean,
            "fpr_max_max": round(self.fpr_max_max, 4),
            "fpr_min_mean": round(self.fpr_min_mean, 4),
            "fpr_reduction_pct": round(self.fpr_reduction_pct, 2),
            "avg_score_max_max": round(self.avg_score_max_max, 4),
            "avg_score_min_mean": round(self.avg_score_min_mean, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


def run_summarization_fpr_benchmark(
    max_samples: int | None = None,
    threshold: float = 0.55,
    nli_model: str | None = None,
    scorer_backend: str = "deberta",
) -> SummarizationFPRMetrics:
    """Score correct summaries under both aggregation strategies.

    Parameters
    ----------
    max_samples : cap on number of correct summaries to evaluate.
    threshold : coherence hard_limit (samples below this are rejected).
    nli_model : HuggingFace NLI model ID.
    scorer_backend : "deberta", "hybrid", "onnx", or "lite".
    """
    from director_ai.core.scorer import CoherenceScorer
    from director_ai.core.vector_store import VectorGroundTruthStore

    scorer = CoherenceScorer(
        threshold=threshold,
        soft_limit=threshold + 0.05,
        use_nli=True,
        ground_truth_store=VectorGroundTruthStore(),
        nli_model=nli_model,
        scorer_backend=scorer_backend,
    )

    samples = _download_task_data("summarization")
    correct_pairs: list[tuple[str, str]] = []
    for sample_data in samples:
        for context, response, is_hallucinated in _extract_pairs(
            "summarization", sample_data
        ):
            if not is_hallucinated and context and response:
                correct_pairs.append((context, response))

    if max_samples:
        correct_pairs = correct_pairs[:max_samples]

    metrics = SummarizationFPRMetrics()

    for i, (document, summary) in enumerate(correct_pairs):
        if (i + 1) % 50 == 0:
            logger.info("Progress: %d / %d", i + 1, len(correct_pairs))

        store = VectorGroundTruthStore()
        scorer.ground_truth_store = store
        store.ingest([document])

        # max-max (old default)
        scorer._fact_inner_agg = "max"
        scorer._fact_outer_agg = "max"
        t0 = time.perf_counter()
        approved_mm, score_mm = scorer.review(document, summary)
        elapsed = time.perf_counter() - t0

        # min-mean (new profile)
        scorer._fact_inner_agg = "min"
        scorer._fact_outer_agg = "mean"
        approved_mn, score_mn = scorer.review(document, summary)

        metrics.total += 1
        metrics.scores_max_max.append(score_mm.score)
        metrics.scores_min_mean.append(score_mn.score)
        metrics.inference_times.append(elapsed)

        if not approved_mm:
            metrics.fp_max_max += 1
        if not approved_mn:
            metrics.fp_min_mean += 1

    return metrics


def _print_results(m: SummarizationFPRMetrics) -> None:
    print(f"\n{'=' * 65}")
    print("  Summarization FPR: max-max vs min-mean Aggregation")
    print(f"{'=' * 65}")
    print(f"  Correct summaries evaluated: {m.total}")
    print()
    hdr = f"  {'Metric':<28} {'max-max':>10} {'min-mean':>10}"
    print(hdr)
    print(f"  {'-' * 50}")
    print(f"  {'False positives':<28} {m.fp_max_max:>10} {m.fp_min_mean:>10}")
    print(f"  {'FPR':<28} {m.fpr_max_max:>9.1%} {m.fpr_min_mean:>9.1%}")
    print(
        f"  {'Avg coherence score':<28} {m.avg_score_max_max:>10.4f} {m.avg_score_min_mean:>10.4f}"
    )
    print()
    print(f"  FPR reduction: {m.fpr_reduction_pct:.1f}%")
    if m.inference_times:
        print(f"  Avg latency:   {m.avg_latency_ms:.1f} ms")
    print(f"{'=' * 65}")


# ── Pytest ─────────────────────────────────────────────────────────


@pytest.mark.slow
def test_summarization_fpr_reduction():
    m = run_summarization_fpr_benchmark(max_samples=100)
    _print_results(m)
    assert m.fpr_min_mean < m.fpr_max_max, (
        f"min-mean FPR ({m.fpr_min_mean:.1%}) should be lower than "
        f"max-max FPR ({m.fpr_max_max:.1%})"
    )


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Summarization FPR: max-max vs min-mean aggregation"
    )
    add_common_args(parser)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument(
        "--scorer-backend",
        type=str,
        default="deberta",
        choices=["deberta", "hybrid", "onnx", "lite"],
    )
    args = parser.parse_args()

    m = run_summarization_fpr_benchmark(
        max_samples=args.max_samples,
        threshold=args.threshold,
        nli_model=args.model,
        scorer_backend=args.scorer_backend,
    )
    _print_results(m)

    save_results(
        {"benchmark": "Summarization_FPR_AggComparison", **m.to_dict()},
        "summarization_fpr_results.json",
    )
