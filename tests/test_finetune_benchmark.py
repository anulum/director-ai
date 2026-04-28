# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Anti-Regression Benchmark Gate Tests
"""Multi-angle tests for fine-tune regression gate pipeline."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from director_ai.core.finetune_benchmark import (
    _BASELINE_ACCURACY,
    _DEPLOY_THRESHOLD_PP,
    _REJECT_THRESHOLD_PP,
    ModelBenchmarkReport,
    ModelBenchmarkResult,
    RegressionReport,
    _load_benchmark_jsonl,
    benchmark_finetuned_model,
    benchmark_model_candidates,
)


def _make_benchmark_file(tmp_path, name, n=100):
    rows = []
    for i in range(n):
        rows.append(
            {
                "premise": f"Evidence {i} is factual.",
                "hypothesis": f"Claim {i} derived from evidence.",
                "label": i % 2,
            },
        )
    f = tmp_path / name
    f.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    return f


class TestRegressionReport:
    def test_defaults(self):
        r = RegressionReport()
        assert r.recommendation == "deploy"
        assert r.regression_acceptable
        assert r.baseline_accuracy == _BASELINE_ACCURACY

    def test_summary_format(self):
        r = RegressionReport(
            domain_accuracy=0.85,
            general_accuracy=0.74,
            regression_pp=-1.8,
            recommendation="deploy",
        )
        s = r.summary()
        assert "85.0%" in s
        assert "74.0%" in s
        assert "deploy" in s


class TestLoadBenchmarkJsonl:
    def test_loads_standard_fields(self, tmp_path):
        f = _make_benchmark_file(tmp_path, "bench.jsonl", 50)
        rows = _load_benchmark_jsonl(f)
        assert len(rows) == 50
        assert all(k in rows[0] for k in ("premise", "hypothesis", "label"))

    def test_loads_alternative_fields(self, tmp_path):
        f = tmp_path / "alt.jsonl"
        f.write_text(
            json.dumps({"doc": "Source.", "claim": "Derived.", "label": 1}) + "\n",
            encoding="utf-8",
        )
        rows = _load_benchmark_jsonl(f)
        assert len(rows) == 1
        assert rows[0]["premise"] == "Source."

    def test_skips_incomplete(self, tmp_path):
        f = tmp_path / "partial.jsonl"
        f.write_text(
            json.dumps({"premise": "a", "hypothesis": "b", "label": 1})
            + "\n"
            + json.dumps({"premise": "a"})
            + "\n",
            encoding="utf-8",
        )
        rows = _load_benchmark_jsonl(f)
        assert len(rows) == 1


class TestBenchmarkDecisionLogic:
    """Test the regression decision logic by mocking _evaluate_model."""

    def _mock_eval(self, bal_acc, f1=0.8):
        return lambda *a, **kw: {"balanced_accuracy": bal_acc, "f1": f1}

    @patch("director_ai.core.finetune_benchmark._evaluate_model")
    def test_deploy_no_regression(self, mock_eval, tmp_path):
        mock_eval.side_effect = self._mock_eval(0.76)
        general = _make_benchmark_file(tmp_path, "general.jsonl")
        report = benchmark_finetuned_model(
            "/fake/model",
            general_path=general,
            baseline_accuracy=0.758,
        )
        assert report.recommendation == "deploy"
        assert report.regression_acceptable

    @patch("director_ai.core.finetune_benchmark._evaluate_model")
    def test_deploy_domain_only_moderate_regression(self, mock_eval, tmp_path):
        mock_eval.side_effect = self._mock_eval(0.71)
        general = _make_benchmark_file(tmp_path, "general.jsonl")
        report = benchmark_finetuned_model(
            "/fake/model",
            general_path=general,
            baseline_accuracy=0.758,
        )
        assert report.recommendation == "deploy_domain_only"
        assert not report.regression_acceptable

    @patch("director_ai.core.finetune_benchmark._evaluate_model")
    def test_reject_catastrophic_regression(self, mock_eval, tmp_path):
        mock_eval.side_effect = self._mock_eval(0.60)
        general = _make_benchmark_file(tmp_path, "general.jsonl")
        report = benchmark_finetuned_model(
            "/fake/model",
            general_path=general,
            baseline_accuracy=0.758,
        )
        assert report.recommendation == "reject"
        assert not report.regression_acceptable

    @patch("director_ai.core.finetune_benchmark._evaluate_model")
    def test_domain_eval_metrics(self, mock_eval, tmp_path):
        mock_eval.side_effect = self._mock_eval(0.88, f1=0.85)
        domain = _make_benchmark_file(tmp_path, "domain.jsonl")
        report = benchmark_finetuned_model(
            "/fake/model",
            eval_path=domain,
            baseline_accuracy=0.758,
        )
        assert report.domain_accuracy == 0.88
        assert report.domain_f1 == 0.85

    def test_no_general_data_defaults_domain_only(self, tmp_path):
        report = benchmark_finetuned_model(
            "/fake/model",
            general_path=None,
            baseline_accuracy=0.758,
        )
        assert report.recommendation == "deploy_domain_only"
        assert report.details.get("reason") == "no general benchmark available"

    @patch("director_ai.core.finetune_benchmark._evaluate_model")
    def test_regression_pp_calculation(self, mock_eval, tmp_path):
        mock_eval.side_effect = self._mock_eval(0.72)
        general = _make_benchmark_file(tmp_path, "general.jsonl")
        report = benchmark_finetuned_model(
            "/fake/model",
            general_path=general,
            baseline_accuracy=0.758,
        )
        expected_pp = (0.72 - 0.758) * 100  # -3.8pp
        assert abs(report.regression_pp - expected_pp) < 0.1

    @patch("director_ai.core.finetune_benchmark._evaluate_model")
    def test_improvement_is_deploy(self, mock_eval, tmp_path):
        mock_eval.side_effect = self._mock_eval(0.80)
        general = _make_benchmark_file(tmp_path, "general.jsonl")
        report = benchmark_finetuned_model(
            "/fake/model",
            general_path=general,
            baseline_accuracy=0.758,
        )
        assert report.recommendation == "deploy"
        assert report.regression_pp > 0

    @patch("director_ai.core.finetune_benchmark._evaluate_model")
    def test_both_domain_and_general(self, mock_eval, tmp_path):
        call_count = [0]

        def side_effect(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"balanced_accuracy": 0.92, "f1": 0.90}
            return {"balanced_accuracy": 0.74, "f1": 0.72}

        mock_eval.side_effect = side_effect
        domain = _make_benchmark_file(tmp_path, "domain.jsonl")
        general = _make_benchmark_file(tmp_path, "general.jsonl")
        report = benchmark_finetuned_model(
            "/fake/model",
            general_path=general,
            eval_path=domain,
            baseline_accuracy=0.758,
        )
        assert report.domain_accuracy == 0.92
        assert report.general_accuracy == 0.74
        assert report.recommendation == "deploy"  # -1.8pp < 3pp threshold


class TestThresholdConstants:
    def test_deploy_threshold(self):
        assert _DEPLOY_THRESHOLD_PP == 3.0

    def test_reject_threshold(self):
        assert _REJECT_THRESHOLD_PP == 8.0

    def test_baseline(self):
        assert pytest.approx(0.758, abs=0.001) == _BASELINE_ACCURACY


class TestModelBenchmarkSweep:
    @patch("director_ai.core.finetune_benchmark._evaluate_model")
    def test_sweep_selects_best_non_rejected_model(self, mock_eval, tmp_path):
        general = _make_benchmark_file(tmp_path, "general.jsonl")
        mock_eval.side_effect = [
            {"balanced_accuracy": 0.74, "f1": 0.72},
            {"balanced_accuracy": 0.80, "f1": 0.78},
        ]
        report = benchmark_model_candidates(
            {
                "factcg-deberta-v3-large": tmp_path / "factcg-model",
                "roberta-large-mnli": tmp_path / "roberta-model",
            },
            general_path=general,
            allow_experimental=True,
        )
        assert isinstance(report, ModelBenchmarkReport)
        assert report.best_model_alias == "roberta-large-mnli"
        assert len(report.results) == 2
        assert all(
            isinstance(result, ModelBenchmarkResult) for result in report.results
        )

    @patch("director_ai.core.finetune_benchmark._evaluate_model")
    def test_sweep_records_rejected_candidates(self, mock_eval, tmp_path):
        general = _make_benchmark_file(tmp_path, "general.jsonl")
        mock_eval.return_value = {"balanced_accuracy": 0.60, "f1": 0.50}
        report = benchmark_model_candidates(
            {"factcg-deberta-v3-large": tmp_path / "factcg-model"},
            general_path=general,
        )
        assert report.best_model_alias == ""
        assert report.results[0].recommendation == "reject"

    def test_sweep_rejects_unknown_without_experimental_flag(self, tmp_path):
        report = benchmark_model_candidates(
            {"org/custom-model": tmp_path / "custom-model"},
        )
        assert report.results[0].recommendation == "reject"
        assert "stable fine-tune registry" in report.results[0].error

    def test_sweep_requires_models(self):
        with pytest.raises(ValueError, match="at least one model"):
            benchmark_model_candidates({})


class TestExports:
    def test_importable_from_core(self):
        from director_ai.core import (
            ModelBenchmarkReport,
            RegressionReport,
            benchmark_finetuned_model,
            benchmark_model_candidates,
        )

        assert callable(benchmark_finetuned_model)
        assert callable(benchmark_model_candidates)
        assert RegressionReport is not None
        assert ModelBenchmarkReport is not None
