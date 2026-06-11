# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Anti-Regression Benchmark Gate Tests
"""Multi-angle tests for fine-tune regression gate pipeline."""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import patch

import pytest

from director_ai.core.finetune_benchmark import (
    _BASELINE_ACCURACY,
    _DEPLOY_THRESHOLD_PP,
    _REJECT_THRESHOLD_PP,
    ModelBenchmarkReport,
    ModelBenchmarkResult,
    RegressionReport,
    _evaluate_model,
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

    def test_rejects_directory_paths(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not a file"):
            _load_benchmark_jsonl(tmp_path)

    def test_skips_invalid_json_and_accepts_response_field(self, tmp_path):
        f = tmp_path / "mixed.jsonl"
        f.write_text(
            "{not-json}\n"
            + json.dumps({"context": "Grounding.", "response": "Claim.", "label": "1"})
            + "\n",
            encoding="utf-8",
        )

        rows = _load_benchmark_jsonl(f)

        assert rows == [{"premise": "Grounding.", "hypothesis": "Claim.", "label": 1}]

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


class _FakeTensor:
    def __init__(self, value):
        self.value = value

    def to(self, device):
        self.device = device
        return self


class _FakePredictions:
    def __init__(self, values):
        self._values = values

    def cpu(self):
        return self

    def numpy(self):
        return self

    def flatten(self):
        return self._values


class _FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTorch(types.SimpleNamespace):
    def __init__(self, predictions):
        super().__init__()
        self.predictions = list(predictions)
        self.devices = []

    def device(self, name):
        self.devices.append(name)
        return name

    def no_grad(self):
        return _FakeNoGrad()

    def argmax(self, logits, dim=-1):
        return _FakePredictions(self.predictions.pop(0))


class _FakeTokenizer:
    sep_token = "[SEP]"
    calls: list[list[str]] = []
    revisions: list[str | None] = []

    @classmethod
    def from_pretrained(cls, model_source, revision=None):
        cls.revisions.append(revision)
        return cls()

    def __call__(self, batch_texts, **kwargs):
        self.calls.append(list(batch_texts))
        assert kwargs["truncation"] is True
        assert kwargs["padding"] is True
        assert kwargs["max_length"] == 512
        assert kwargs["return_tensors"] == "pt"
        return {"input_ids": _FakeTensor(batch_texts)}


class _FakeModel:
    revisions: list[str | None] = []
    moved_to: list[str] = []

    @classmethod
    def from_pretrained(cls, model_source, revision=None):
        cls.revisions.append(revision)
        return cls()

    def eval(self):
        self.evaluated = True

    def to(self, device):
        self.moved_to.append(str(device))
        return self

    def __call__(self, **encodings):
        assert "input_ids" in encodings
        return types.SimpleNamespace(logits=object())


def _install_fake_inference_modules(monkeypatch, *, predictions):
    _FakeTokenizer.calls = []
    _FakeTokenizer.revisions = []
    _FakeModel.revisions = []
    _FakeModel.moved_to = []
    fake_torch = _FakeTorch(predictions)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=_FakeTokenizer,
            AutoModelForSequenceClassification=_FakeModel,
        ),
    )
    monkeypatch.setattr(
        "director_ai.core.finetune_benchmark.resolve_model_revision",
        lambda _model_source: "test-revision",
    )
    monkeypatch.setattr(
        "director_ai.core._device.select_torch_device",
        lambda: "cpu",
    )
    released = []
    monkeypatch.setattr(
        "director_ai.core._device.release_torch_cuda",
        lambda: released.append(True),
    )
    return released


class TestEvaluateModel:
    def test_evaluate_model_batches_plain_nli_inputs_and_releases_device(
        self, monkeypatch
    ):
        released = _install_fake_inference_modules(
            monkeypatch,
            predictions=[[0, 1], [1]],
        )
        samples = [
            {"premise": "A", "hypothesis": "A", "label": 0},
            {"premise": "B", "hypothesis": "B", "label": 1},
            {"premise": "C", "hypothesis": "C", "label": 1},
        ]

        metrics = _evaluate_model("plain-model", samples, batch_size=2)

        assert metrics == {"balanced_accuracy": 1.0, "f1": 1.0}
        assert _FakeTokenizer.revisions == ["test-revision"]
        assert _FakeModel.revisions == ["test-revision"]
        assert _FakeTokenizer.calls == [["A [SEP] A", "B [SEP] B"], ["C [SEP] C"]]
        assert _FakeModel.moved_to == ["cpu", "cpu"]
        assert released == [True]

    def test_evaluate_model_uses_factcg_template(self, monkeypatch):
        released = _install_fake_inference_modules(monkeypatch, predictions=[[1]])

        metrics = _evaluate_model(
            "factcg-model",
            [{"premise": "Source.", "hypothesis": "Claim.", "label": 1}],
            batch_size=4,
        )

        assert metrics["balanced_accuracy"] == 1.0
        assert "Source." in _FakeTokenizer.calls[0][0]
        assert '"Claim."' in _FakeTokenizer.calls[0][0]
        assert "OPTIONS:" in _FakeTokenizer.calls[0][0]
        assert released == [True]

    def test_evaluate_model_reports_missing_finetune_extras(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)

        with pytest.raises(ImportError, match=r"director-ai\[finetune\]"):
            _evaluate_model(
                "plain-model",
                [{"premise": "A", "hypothesis": "A", "label": 0}],
            )


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

    def test_report_without_deployable_candidates_has_no_winner(self):
        report = ModelBenchmarkReport(
            results=[
                ModelBenchmarkResult(
                    requested_model="bad",
                    alias="bad",
                    model_id="bad",
                    model_path="/tmp/bad",
                    status="error",
                    template="unknown",
                    label_count=0,
                    baseline_accuracy=0.0,
                    recommended_batch_size=0,
                    error="failed",
                )
            ]
        )

        assert report.best_model_alias == ""
        assert report.best_model_id == ""

    def test_report_summary_includes_result_rows_and_errors(self):
        report = ModelBenchmarkReport(
            results=[
                ModelBenchmarkResult(
                    requested_model="candidate",
                    alias="candidate",
                    model_id="model-id",
                    model_path="/tmp/model",
                    status="stable",
                    template="nli_pair",
                    label_count=2,
                    baseline_accuracy=0.75,
                    recommended_batch_size=16,
                    general_accuracy=0.8,
                    domain_accuracy=0.9,
                    recommendation="deploy",
                ),
                ModelBenchmarkResult(
                    requested_model="broken",
                    alias="broken",
                    model_id="broken",
                    model_path="/tmp/broken",
                    status="error",
                    template="unknown",
                    label_count=0,
                    baseline_accuracy=0.0,
                    recommended_batch_size=0,
                    error="load failed",
                ),
            ],
            general_path="general.jsonl",
            eval_path="domain.jsonl",
        )

        text = report.summary()

        assert "General data: general.jsonl" in text
        assert "Domain data: domain.jsonl" in text
        assert "Best model: candidate" in text
        assert (
            "- broken: general=0.0%, domain=0.0%, rec=reject error=load failed" in text
        )

    def test_model_result_uses_requested_alias_for_custom_profiles(self):
        from director_ai.core.training.model_registry import TrainingModelProfile

        result = ModelBenchmarkResult.from_report(
            requested_model="org/custom-nli",
            profile=TrainingModelProfile(
                alias="custom-experimental",
                model_id="org/custom-nli",
                template="nli_pair",
                label_count=2,
                status="experimental",
                baseline_accuracy=0.7,
                default_max_length=512,
                recommended_batch_size=12,
                recommended_learning_rate=1e-5,
                hardware_profile="single-l4",
            ),
            model_path="/models/custom",
            report=RegressionReport(
                general_accuracy=0.78,
                domain_accuracy=0.81,
                recommendation="deploy",
                details={"samples": 5},
            ),
            elapsed_seconds=1.25,
        )

        payload = result.to_dict()
        assert result.alias == "org/custom-nli"
        assert payload["details"] == {"samples": 5}
        assert payload["elapsed_seconds"] == 1.25


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


class TestBenchmarkBoundaryDecisions:
    """Fine-tune benchmark gates preserve deployment threshold boundaries."""

    @patch("director_ai.core.finetune_benchmark._evaluate_model")
    def test_deploy_threshold_boundary(self, mock_eval, tmp_path):
        general = _make_benchmark_file(tmp_path, "general-boundary.jsonl")

        mock_eval.return_value = {"balanced_accuracy": 0.729, "f1": 0.75}
        deploy = benchmark_finetuned_model(
            "/fake/model",
            general_path=general,
            baseline_accuracy=0.758,
        )

        mock_eval.return_value = {"balanced_accuracy": 0.727, "f1": 0.75}
        domain_only = benchmark_finetuned_model(
            "/fake/model",
            general_path=general,
            baseline_accuracy=0.758,
        )

        assert deploy.recommendation == "deploy"
        assert domain_only.recommendation == "deploy_domain_only"

    @patch("director_ai.core.finetune_benchmark._evaluate_model")
    def test_reject_threshold_boundary(self, mock_eval, tmp_path):
        general = _make_benchmark_file(tmp_path, "general-reject-boundary.jsonl")

        mock_eval.return_value = {"balanced_accuracy": 0.758 - 0.08, "f1": 0.75}
        domain_only = benchmark_finetuned_model(
            "/fake/model",
            general_path=general,
            baseline_accuracy=0.758,
        )

        mock_eval.return_value = {"balanced_accuracy": 0.758 - 0.0801, "f1": 0.75}
        reject = benchmark_finetuned_model(
            "/fake/model",
            general_path=general,
            baseline_accuracy=0.758,
        )

        assert domain_only.recommendation == "deploy_domain_only"
        assert reject.recommendation == "reject"
        assert reject.regression_acceptable is False

    def test_candidate_sweep_records_model_errors_without_aborting(self, tmp_path):
        report = benchmark_model_candidates(
            {"org/custom-model": tmp_path / "custom-model"},
            allow_experimental=False,
            batch_size=8,
        )

        assert report.best_model_alias == ""
        assert len(report.results) == 1
        result = report.results[0]
        assert result.requested_model == "org/custom-model"
        assert result.status == "error"
        assert result.recommendation == "reject"
        assert result.recommended_batch_size == 8
        assert "stable fine-tune registry" in result.error
