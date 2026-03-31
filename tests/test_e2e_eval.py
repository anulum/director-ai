# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — End-to-End Evaluation Tests
"""Multi-angle tests for benchmarks/e2e_eval.py.

Covers: E2ESample dataclass, E2EMetrics aggregation, metric edge cases,
run_e2e_benchmark parameter wiring (nli_torch_dtype, scorer_backend),
per-task breakdown, and pipeline performance documentation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.e2e_eval import E2EMetrics, E2ESample

# ── E2ESample dataclass ─────────────────────────────────────────────


class TestE2ESample:
    """Test sample construction and field defaults."""

    def test_defaults(self):
        s = E2ESample(task="qa", context="C", response="R", is_hallucinated=True)
        assert s.approved is True
        assert s.coherence_score == 0.0
        assert s.warning is False
        assert s.fallback_used is False

    def test_hallucinated_sample(self):
        s = E2ESample(
            task="qa",
            context="C",
            response="R",
            is_hallucinated=True,
            approved=False,
            coherence_score=0.3,
        )
        assert s.is_hallucinated is True
        assert s.approved is False

    @pytest.mark.parametrize("task", ["qa", "summarization", "dialogue"])
    def test_all_task_types(self, task):
        s = E2ESample(task=task, context="C", response="R", is_hallucinated=False)
        assert s.task == task


# ── E2EMetrics aggregation ──────────────────────────────────────────


class TestE2EMetrics:
    """Multi-angle tests for metric computation."""

    def _make_metrics(self, samples):
        m = E2EMetrics(threshold=0.5, soft_limit=0.6)
        m.samples = samples
        return m

    def test_perfect_catch_rate(self):
        samples = [
            E2ESample("qa", "C", "R", is_hallucinated=True, approved=False),
            E2ESample("qa", "C", "R", is_hallucinated=True, approved=False),
        ]
        m = self._make_metrics(samples)
        assert m.catch_rate == 1.0

    def test_zero_catch_rate(self):
        samples = [
            E2ESample("qa", "C", "R", is_hallucinated=True, approved=True),
        ]
        m = self._make_metrics(samples)
        assert m.catch_rate == 0.0

    def test_false_positive_rate(self):
        samples = [
            E2ESample("qa", "C", "R", is_hallucinated=False, approved=False),
            E2ESample("qa", "C", "R", is_hallucinated=False, approved=True),
        ]
        m = self._make_metrics(samples)
        assert m.false_positive_rate == 0.5

    def test_zero_false_positives(self):
        samples = [
            E2ESample("qa", "C", "R", is_hallucinated=False, approved=True),
        ]
        m = self._make_metrics(samples)
        assert m.false_positive_rate == 0.0

    def test_empty_samples(self):
        m = self._make_metrics([])
        assert m.catch_rate == 0.0
        assert m.false_positive_rate == 0.0

    def test_precision(self):
        samples = [
            E2ESample("qa", "C", "R", is_hallucinated=True, approved=False),  # TP
            E2ESample("qa", "C", "R", is_hallucinated=False, approved=False),  # FP
        ]
        m = self._make_metrics(samples)
        assert m.precision == 0.5

    def test_f1_score(self):
        samples = [
            E2ESample("qa", "C", "R", is_hallucinated=True, approved=False),  # TP
            E2ESample("qa", "C", "R", is_hallucinated=True, approved=True),  # FN
            E2ESample("qa", "C", "R", is_hallucinated=False, approved=True),  # TN
        ]
        m = self._make_metrics(samples)
        # precision = 1/1 = 1.0, recall = 1/2 = 0.5, F1 = 2/3
        assert abs(m.f1 - 2 / 3) < 0.01

    def test_to_dict_contains_all_fields(self):
        samples = [
            E2ESample(
                "qa", "C", "R", is_hallucinated=True, approved=False, latency_ms=10.0
            ),
        ]
        m = self._make_metrics(samples)
        d = m.to_dict()
        required_keys = [
            "total",
            "threshold",
            "tp",
            "fp",
            "tn",
            "fn",
            "catch_rate",
            "false_positive_rate",
            "precision",
            "f1",
            "accuracy",
            "avg_latency_ms",
        ]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"

    def test_per_task_breakdown(self):
        samples = [
            E2ESample("qa", "C", "R", is_hallucinated=True, approved=False),
            E2ESample("dialogue", "C", "R", is_hallucinated=True, approved=True),
            E2ESample("summarization", "C", "R", is_hallucinated=False, approved=True),
        ]
        m = self._make_metrics(samples)
        d = m.to_dict()
        assert "per_task" in d
        assert "qa" in d["per_task"]
        assert "dialogue" in d["per_task"]
        assert "summarization" in d["per_task"]

    def test_latency_aggregation(self):
        samples = [
            E2ESample(
                "qa", "C", "R", is_hallucinated=True, approved=False, latency_ms=10.0
            ),
            E2ESample(
                "qa", "C", "R", is_hallucinated=True, approved=False, latency_ms=20.0
            ),
        ]
        m = self._make_metrics(samples)
        assert m.avg_latency_ms == 15.0

    @pytest.mark.parametrize(
        "n_tp,n_fp,n_tn,n_fn,expected_acc",
        [
            (10, 0, 10, 0, 1.0),  # perfect
            (5, 5, 5, 5, 0.5),  # random
            (0, 10, 0, 10, 0.0),  # worst case
        ],
    )
    def test_accuracy_parametrized(self, n_tp, n_fp, n_tn, n_fn, expected_acc):
        samples = (
            [E2ESample("qa", "C", "R", is_hallucinated=True, approved=False)] * n_tp
            + [E2ESample("qa", "C", "R", is_hallucinated=False, approved=False)] * n_fp
            + [E2ESample("qa", "C", "R", is_hallucinated=False, approved=True)] * n_tn
            + [E2ESample("qa", "C", "R", is_hallucinated=True, approved=True)] * n_fn
        )
        m = self._make_metrics(samples)
        assert abs(m.accuracy - expected_acc) < 0.01


# ── run_e2e_benchmark wiring ───────────────────────────────────────


class TestRunE2EBenchmarkWiring:
    """Verify run_e2e_benchmark passes parameters to CoherenceScorer."""

    def test_nli_torch_dtype_parameter_exists(self):
        """run_e2e_benchmark must accept nli_torch_dtype kwarg."""
        import inspect

        from benchmarks.e2e_eval import run_e2e_benchmark

        sig = inspect.signature(run_e2e_benchmark)
        assert "nli_torch_dtype" in sig.parameters

    def test_scorer_backend_parameter_exists(self):
        import inspect

        from benchmarks.e2e_eval import run_e2e_benchmark

        sig = inspect.signature(run_e2e_benchmark)
        assert "scorer_backend" in sig.parameters

    def test_llm_judge_parameters_exist(self):
        import inspect

        from benchmarks.e2e_eval import run_e2e_benchmark

        sig = inspect.signature(run_e2e_benchmark)
        assert "llm_judge_provider" in sig.parameters
        assert "llm_judge_model" in sig.parameters


# ── Pipeline performance documentation ──────────────────────────────


class TestPipelinePerformanceDoc:
    """Verify e2e evaluation documents pipeline performance."""

    def test_to_dict_includes_latency(self):
        m = E2EMetrics(threshold=0.5, soft_limit=0.6)
        m.samples = [
            E2ESample(
                "qa", "C", "R", is_hallucinated=True, approved=False, latency_ms=15.0
            ),
        ]
        d = m.to_dict()
        assert "avg_latency_ms" in d
        assert "p95_latency_ms" in d

    def test_to_dict_includes_evidence_coverage(self):
        m = E2EMetrics(threshold=0.5, soft_limit=0.6)
        m.samples = [
            E2ESample(
                "qa",
                "C",
                "R",
                is_hallucinated=True,
                approved=False,
                has_evidence=True,
                evidence_chunks=3,
            ),
        ]
        d = m.to_dict()
        assert "evidence_coverage" in d

    def test_to_dict_includes_warning_rate(self):
        m = E2EMetrics(threshold=0.5, soft_limit=0.6)
        m.samples = [
            E2ESample(
                "qa", "C", "R", is_hallucinated=False, approved=True, warning=True
            ),
        ]
        d = m.to_dict()
        assert "warning_rate" in d
