# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Summarization False-Positive Rate Benchmark

"""A/B benchmark comparing aggregation strategies on correct summaries.

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
    fp_summ_profile: int = 0
    fp_phase3: int = 0
    scores_max_max: list[float] = field(default_factory=list, repr=False)
    scores_min_mean: list[float] = field(default_factory=list, repr=False)
    scores_summ_profile: list[float] = field(default_factory=list, repr=False)
    scores_phase3: list[float] = field(default_factory=list, repr=False)
    inference_times: list[float] = field(default_factory=list, repr=False)

    @property
    def fpr_max_max(self) -> float:
        return self.fp_max_max / max(self.total, 1)

    @property
    def fpr_min_mean(self) -> float:
        return self.fp_min_mean / max(self.total, 1)

    @property
    def fpr_summ_profile(self) -> float:
        return self.fp_summ_profile / max(self.total, 1)

    @property
    def fpr_phase3(self) -> float:
        return self.fp_phase3 / max(self.total, 1)

    @property
    def fpr_reduction_pct(self) -> float:
        if self.fpr_max_max == 0:
            return 0.0
        return (self.fpr_max_max - self.fpr_min_mean) / self.fpr_max_max * 100

    @property
    def fpr_reduction_summ_pct(self) -> float:
        if self.fpr_max_max == 0:
            return 0.0
        return (self.fpr_max_max - self.fpr_summ_profile) / self.fpr_max_max * 100

    @property
    def fpr_reduction_phase3_pct(self) -> float:
        if self.fpr_max_max == 0:
            return 0.0
        return (self.fpr_max_max - self.fpr_phase3) / self.fpr_max_max * 100

    @property
    def avg_score_max_max(self) -> float:
        return float(np.mean(self.scores_max_max)) if self.scores_max_max else 0.0

    @property
    def avg_score_min_mean(self) -> float:
        return float(np.mean(self.scores_min_mean)) if self.scores_min_mean else 0.0

    @property
    def avg_score_summ_profile(self) -> float:
        return (
            float(np.mean(self.scores_summ_profile))
            if self.scores_summ_profile
            else 0.0
        )

    @property
    def avg_score_phase3(self) -> float:
        return float(np.mean(self.scores_phase3)) if self.scores_phase3 else 0.0

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
            "fp_summ_profile": self.fp_summ_profile,
            "fp_phase3": self.fp_phase3,
            "fpr_max_max": round(self.fpr_max_max, 4),
            "fpr_min_mean": round(self.fpr_min_mean, 4),
            "fpr_summ_profile": round(self.fpr_summ_profile, 4),
            "fpr_phase3": round(self.fpr_phase3, 4),
            "fpr_reduction_pct": round(self.fpr_reduction_pct, 2),
            "fpr_reduction_summ_pct": round(self.fpr_reduction_summ_pct, 2),
            "fpr_reduction_phase3_pct": round(self.fpr_reduction_phase3_pct, 2),
            "avg_score_max_max": round(self.avg_score_max_max, 4),
            "avg_score_min_mean": round(self.avg_score_min_mean, 4),
            "avg_score_summ_profile": round(self.avg_score_summ_profile, 4),
            "avg_score_phase3": round(self.avg_score_phase3, 4),
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
            "summarization",
            sample_data,
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

        # min-mean (factual agg only)
        scorer._fact_inner_agg = "min"
        scorer._fact_outer_agg = "mean"
        scorer._logic_inner_agg = "max"
        scorer._logic_outer_agg = "max"
        scorer._premise_ratio = 0.4
        approved_mn, score_mn = scorer.review(document, summary)

        # summ-profile (all three fixes: logic agg + premise ratio + fact agg)
        scorer._fact_inner_agg = "min"
        scorer._fact_outer_agg = "mean"
        scorer._logic_inner_agg = "min"
        scorer._logic_outer_agg = "mean"
        scorer._premise_ratio = 0.85
        approved_sp, score_sp = scorer.review(document, summary)

        # phase3 (w_logic=0, trimmed_mean, top_k=8, threshold=0.20)
        scorer.W_LOGIC = 0.0
        scorer.W_FACT = 1.0
        scorer._fact_inner_agg = "min"
        scorer._fact_outer_agg = "trimmed_mean"
        scorer._logic_inner_agg = "min"
        scorer._logic_outer_agg = "mean"
        scorer._premise_ratio = 0.85
        scorer._fact_retrieval_top_k = 8
        old_threshold = scorer.threshold
        scorer.threshold = 0.20
        approved_p3, score_p3 = scorer.review(document, summary)
        scorer.threshold = old_threshold
        scorer.W_LOGIC = 0.6
        scorer.W_FACT = 0.4
        scorer._fact_retrieval_top_k = 3

        metrics.total += 1
        metrics.scores_max_max.append(score_mm.score)
        metrics.scores_min_mean.append(score_mn.score)
        metrics.scores_summ_profile.append(score_sp.score)
        metrics.scores_phase3.append(score_p3.score)
        metrics.inference_times.append(elapsed)

        if not approved_mm:
            metrics.fp_max_max += 1
        if not approved_mn:
            metrics.fp_min_mean += 1
        if not approved_sp:
            metrics.fp_summ_profile += 1
        if not approved_p3:
            metrics.fp_phase3 += 1

    return metrics


def _print_results(m: SummarizationFPRMetrics) -> None:
    print(f"\n{'=' * 80}")
    print("  Summarization FPR: Four-Way Comparison")
    print(f"{'=' * 80}")
    print(f"  Correct summaries evaluated: {m.total}")
    print()
    hdr = (
        f"  {'Metric':<24} {'max-max':>10} {'min-mean':>10}"
        f" {'summ-prof':>10} {'phase3':>10}"
    )
    print(hdr)
    print(f"  {'-' * 66}")
    print(
        f"  {'False positives':<24} {m.fp_max_max:>10} {m.fp_min_mean:>10}"
        f" {m.fp_summ_profile:>10} {m.fp_phase3:>10}",
    )
    print(
        f"  {'FPR':<24} {m.fpr_max_max:>9.1%} {m.fpr_min_mean:>9.1%}"
        f" {m.fpr_summ_profile:>9.1%} {m.fpr_phase3:>9.1%}",
    )
    print(
        f"  {'Avg coherence score':<24} {m.avg_score_max_max:>10.4f}"
        f" {m.avg_score_min_mean:>10.4f} {m.avg_score_summ_profile:>10.4f}"
        f" {m.avg_score_phase3:>10.4f}",
    )
    print()
    print(f"  FPR reduction (min-mean):    {m.fpr_reduction_pct:.1f}%")
    print(f"  FPR reduction (summ-prof):   {m.fpr_reduction_summ_pct:.1f}%")
    print(f"  FPR reduction (phase3):      {m.fpr_reduction_phase3_pct:.1f}%")
    if m.inference_times:
        print(f"  Avg latency:                 {m.avg_latency_ms:.1f} ms")
    print(f"{'=' * 80}")


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
        description="Summarization FPR: max-max vs min-mean aggregation",
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
