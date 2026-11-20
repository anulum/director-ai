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
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Literal, Self
from unittest.mock import MagicMock, patch

import pytest

from director_ai.core.training.finetune import TrainingRow
from director_ai.core.training.finetune_benchmark import (
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
from director_ai.core.training.training_provenance import (
    PROVENANCE_FILENAME,
    TrainingProvenance,
)


def _make_benchmark_file(tmp_path: Path, name: str, n: int = 100) -> Path:
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


def _trained_model(tmp_path: Path, name: str = "model") -> Path:
    """Create a model directory whose provenance is disjoint from benchmarks."""
    directory = tmp_path / name
    rows: list[TrainingRow] = [
        {
            "premise": f"Training passage {i}.",
            "hypothesis": f"Training claim {i}.",
            "label": i % 2,
            "source_cluster": f"train-doc-{i}",
        }
        for i in range(10)
    ]
    TrainingProvenance.from_rows(rows).write(directory)
    return directory


class TestRegressionReport:
    """RegressionReport defaults and summary."""

    def test_defaults(self) -> None:
        """An unmeasured report recommends reject."""
        r = RegressionReport()
        assert r.recommendation == "reject"
        assert not r.regression_acceptable
        assert r.baseline_accuracy == _BASELINE_ACCURACY

    def test_summary_format(self) -> None:
        """The summary shows accuracies and the recommendation."""
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
        assert "selection set" not in s

    def test_summary_marks_a_domain_set_that_chose_the_checkpoint(self) -> None:
        """A domain figure from the checkpoint-selection set is labelled."""
        r = RegressionReport(
            domain_accuracy=0.9,
            details={"domain_set_selected_checkpoint": True},
        )
        assert "Domain bal_acc: 90.0% (selection set)" in r.summary()


class TestLoadBenchmarkJsonl:
    """Loading of benchmark JSONL rows."""

    def test_loads_standard_fields(self, tmp_path: Path) -> None:
        """Standard premise, hypothesis and label rows load."""
        f = _make_benchmark_file(tmp_path, "bench.jsonl", 50)
        rows = _load_benchmark_jsonl(f)
        assert len(rows) == 50
        assert all(k in rows[0] for k in ("premise", "hypothesis", "label"))

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        """Blank lines are skipped."""
        f = tmp_path / "padded.jsonl"
        f.write_text(
            json.dumps({"premise": "S.", "hypothesis": "C.", "label": 1})
            + "\n\n   \n"
            + json.dumps({"premise": "S2.", "hypothesis": "C2.", "label": 0})
            + "\n",
            encoding="utf-8",
        )
        rows = _load_benchmark_jsonl(f)
        assert len(rows) == 2

    def test_loads_alternative_fields(self, tmp_path: Path) -> None:
        """The doc and claim aliases load as premise and hypothesis."""
        f = tmp_path / "alt.jsonl"
        f.write_text(
            json.dumps({"doc": "Source.", "claim": "Derived.", "label": 1}) + "\n",
            encoding="utf-8",
        )
        rows = _load_benchmark_jsonl(f)
        assert len(rows) == 1
        assert rows[0]["premise"] == "Source."

    def test_rejects_directory_paths(self, tmp_path: Path) -> None:
        """A directory is not accepted as benchmark data."""
        with pytest.raises(FileNotFoundError, match="not a file"):
            _load_benchmark_jsonl(tmp_path)

    def test_skips_invalid_json_and_accepts_response_field(
        self, tmp_path: Path
    ) -> None:
        """Invalid JSON is skipped and the response alias is accepted."""
        f = tmp_path / "mixed.jsonl"
        f.write_text(
            "{not-json}\n"
            + json.dumps({"context": "Grounding.", "response": "Claim.", "label": "1"})
            + "\n",
            encoding="utf-8",
        )

        rows = _load_benchmark_jsonl(f)

        assert rows == [{"premise": "Grounding.", "hypothesis": "Claim.", "label": 1}]

    def test_skips_incomplete(self, tmp_path: Path) -> None:
        """Rows missing a required field are skipped."""
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
    def __init__(self, value: list[str]) -> None:
        self.value = value

    def to(self, device: str) -> Self:
        self.device = device
        return self


class _FakePredictions:
    def __init__(self, values: list[int]) -> None:
        self._values = values

    def cpu(self) -> Self:
        return self

    def numpy(self) -> Self:
        return self

    def flatten(self) -> list[int]:
        return self._values


class _FakeNoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        return False


class _FakeTorch(types.SimpleNamespace):
    def __init__(self, predictions: list[list[int]]) -> None:
        super().__init__()
        self.predictions = list(predictions)
        self.devices: list[str] = []

    def device(self, name: str) -> str:
        self.devices.append(name)
        return name

    def no_grad(self) -> _FakeNoGrad:
        return _FakeNoGrad()

    def argmax(self, logits: object, dim: int = -1) -> _FakePredictions:
        return _FakePredictions(self.predictions.pop(0))


class _FakeTokenizer:
    sep_token = "[SEP]"
    calls: list[list[str]] = []
    revisions: list[str | None] = []

    @classmethod
    def from_pretrained(
        cls,
        model_source: str,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> Self:
        cls.revisions.append(revision)
        return cls()

    def __call__(
        self, batch_texts: list[str], **kwargs: object
    ) -> dict[str, _FakeTensor]:
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
    def from_pretrained(
        cls,
        model_source: str,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> Self:
        cls.revisions.append(revision)
        return cls()

    def eval(self) -> None:
        self.evaluated = True

    def to(self, device: str) -> Self:
        self.moved_to.append(str(device))
        return self

    def __call__(self, **encodings: _FakeTensor) -> types.SimpleNamespace:
        assert "input_ids" in encodings
        return types.SimpleNamespace(logits=object())


def _install_fake_inference_modules(
    monkeypatch: pytest.MonkeyPatch, *, predictions: list[list[int]]
) -> list[bool]:
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
        "director_ai.core.training.finetune_benchmark.resolve_model_revision",
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
    """Model evaluation with a faked inference stack."""

    def test_evaluate_model_batches_plain_nli_inputs_and_releases_device(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Plain NLI inputs are batched and the device is released."""
        released = _install_fake_inference_modules(
            monkeypatch,
            predictions=[[0, 1], [1]],
        )
        samples: list[TrainingRow] = [
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

    def test_evaluate_model_uses_factcg_template(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FactCG models receive the instruction template."""
        released = _install_fake_inference_modules(monkeypatch, predictions=[[1]])

        metrics = _evaluate_model(
            "opaque-job-artifact",
            [{"premise": "Source.", "hypothesis": "Claim.", "label": 1}],
            batch_size=4,
            template="factcg",
        )

        assert metrics["balanced_accuracy"] == 1.0
        assert "Source." in _FakeTokenizer.calls[0][0]
        assert '"Claim."' in _FakeTokenizer.calls[0][0]
        assert "OPTIONS:" in _FakeTokenizer.calls[0][0]
        assert released == [True]

    def test_evaluate_model_reports_missing_finetune_extras(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing inference stack names the extra to install."""
        monkeypatch.setitem(sys.modules, "torch", None)

        with pytest.raises(ImportError, match=r"director-ai\[finetune\]"):
            _evaluate_model(
                "plain-model",
                [{"premise": "A", "hypothesis": "A", "label": 0}],
            )


class TestBenchmarkDecisionLogic:
    """Test the regression decision logic by mocking _evaluate_model."""

    def _mock_eval(
        self, bal_acc: float, f1: float = 0.8
    ) -> Callable[..., dict[str, float]]:
        return lambda *a, **kw: {"balanced_accuracy": bal_acc, "f1": f1}

    @patch("director_ai.core.training.finetune_benchmark._evaluate_model")
    def test_deploy_no_regression(self, mock_eval: MagicMock, tmp_path: Path) -> None:
        """No regression allows deployment."""
        mock_eval.side_effect = self._mock_eval(0.76)
        general = _make_benchmark_file(tmp_path, "general.jsonl")
        report = benchmark_finetuned_model(
            _trained_model(tmp_path),
            general_path=general,
            baseline_accuracy=0.758,
        )
        assert report.recommendation == "deploy"
        assert report.regression_acceptable

    @patch("director_ai.core.training.finetune_benchmark._evaluate_model")
    def test_deploy_domain_only_moderate_regression(
        self, mock_eval: MagicMock, tmp_path: Path
    ) -> None:
        """A moderate regression allows domain-only deployment."""
        mock_eval.side_effect = self._mock_eval(0.71)
        general = _make_benchmark_file(tmp_path, "general.jsonl")
        report = benchmark_finetuned_model(
            _trained_model(tmp_path),
            general_path=general,
            baseline_accuracy=0.758,
        )
        assert report.recommendation == "deploy_domain_only"
        assert not report.regression_acceptable

    @patch("director_ai.core.training.finetune_benchmark._evaluate_model")
    def test_reject_catastrophic_regression(
        self, mock_eval: MagicMock, tmp_path: Path
    ) -> None:
        """A catastrophic regression is rejected."""
        mock_eval.side_effect = self._mock_eval(0.60)
        general = _make_benchmark_file(tmp_path, "general.jsonl")
        report = benchmark_finetuned_model(
            _trained_model(tmp_path),
            general_path=general,
            baseline_accuracy=0.758,
        )
        assert report.recommendation == "reject"
        assert not report.regression_acceptable

    @patch("director_ai.core.training.finetune_benchmark._evaluate_model")
    def test_domain_eval_metrics(self, mock_eval: MagicMock, tmp_path: Path) -> None:
        """Domain metrics come from the domain evaluation set."""
        mock_eval.side_effect = self._mock_eval(0.88, f1=0.85)
        domain = _make_benchmark_file(tmp_path, "domain.jsonl")
        general = _make_benchmark_file(tmp_path, "general.jsonl")
        report = benchmark_finetuned_model(
            _trained_model(tmp_path),
            general_path=general,
            eval_path=domain,
            baseline_accuracy=0.758,
        )
        assert report.domain_accuracy == 0.88
        assert report.domain_f1 == 0.85

    def test_no_general_data_rejects(self, tmp_path: Path) -> None:
        """Without general data the gate rejects."""
        report = benchmark_finetuned_model(
            _trained_model(tmp_path),
            general_path=None,
            baseline_accuracy=0.758,
        )
        assert report.recommendation == "reject"
        assert report.regression_acceptable is False
        assert report.details.get("reason") == "no general benchmark supplied"

    @patch("director_ai.core.training.finetune_benchmark._evaluate_model")
    def test_regression_pp_calculation(
        self, mock_eval: MagicMock, tmp_path: Path
    ) -> None:
        """Regression is reported in percentage points."""
        mock_eval.side_effect = self._mock_eval(0.72)
        general = _make_benchmark_file(tmp_path, "general.jsonl")
        report = benchmark_finetuned_model(
            _trained_model(tmp_path),
            general_path=general,
            baseline_accuracy=0.758,
        )
        expected_pp = (0.72 - 0.758) * 100  # -3.8pp
        assert abs(report.regression_pp - expected_pp) < 0.1

    @patch("director_ai.core.training.finetune_benchmark._evaluate_model")
    def test_improvement_is_deploy(self, mock_eval: MagicMock, tmp_path: Path) -> None:
        """An improvement over the baseline deploys."""
        mock_eval.side_effect = self._mock_eval(0.80)
        general = _make_benchmark_file(tmp_path, "general.jsonl")
        report = benchmark_finetuned_model(
            _trained_model(tmp_path),
            general_path=general,
            baseline_accuracy=0.758,
        )
        assert report.recommendation == "deploy"
        assert report.regression_pp > 0

    @patch("director_ai.core.training.finetune_benchmark._evaluate_model")
    def test_both_domain_and_general(
        self, mock_eval: MagicMock, tmp_path: Path
    ) -> None:
        """Domain and general metrics are both reported."""
        call_count = [0]

        def side_effect(*a: object, **kw: object) -> dict[str, float]:
            call_count[0] += 1
            if call_count[0] == 1:
                return {"balanced_accuracy": 0.92, "f1": 0.90}
            return {"balanced_accuracy": 0.74, "f1": 0.72}

        mock_eval.side_effect = side_effect
        domain = _make_benchmark_file(tmp_path, "domain.jsonl")
        general = _make_benchmark_file(tmp_path, "general.jsonl")
        report = benchmark_finetuned_model(
            _trained_model(tmp_path),
            general_path=general,
            eval_path=domain,
            baseline_accuracy=0.758,
        )
        assert report.domain_accuracy == 0.92
        assert report.general_accuracy == 0.74
        assert report.recommendation == "deploy"  # -1.8pp < 3pp threshold


class TestBenchmarkFailsClosed:
    """Rejections before measurement and on a zero general accuracy."""

    @patch("director_ai.core.training.finetune_benchmark._evaluate_model")
    def test_missing_provenance_rejects_before_evaluation(
        self, mock_eval: MagicMock, tmp_path: Path
    ) -> None:
        """A model directory without provenance is rejected unevaluated."""
        general = _make_benchmark_file(tmp_path, "general.jsonl")
        bare = tmp_path / "bare-model"
        bare.mkdir()
        report = benchmark_finetuned_model(bare, general_path=general)
        assert report.recommendation == "reject"
        assert report.details["reason"] == "training provenance unavailable"
        mock_eval.assert_not_called()

    @patch("director_ai.core.training.finetune_benchmark._evaluate_model")
    def test_hub_reference_needs_explicit_provenance(
        self, mock_eval: MagicMock, tmp_path: Path
    ) -> None:
        """A model-hub reference uses only an explicitly supplied provenance."""
        mock_eval.return_value = {"balanced_accuracy": 0.76, "f1": 0.7}
        general = _make_benchmark_file(tmp_path, "general.jsonl")
        without = benchmark_finetuned_model("org/hub-model", general_path=general)
        provenance = _trained_model(tmp_path) / PROVENANCE_FILENAME
        explicit = benchmark_finetuned_model(
            "org/hub-model", general_path=general, provenance_path=provenance
        )
        assert without.details["reason"] == "training provenance unavailable"
        assert explicit.recommendation == "deploy"
        assert mock_eval.call_count == 1

    def test_unreadable_provenance_rejects(self, tmp_path: Path) -> None:
        """A malformed provenance file counts as unavailable."""
        general = _make_benchmark_file(tmp_path, "general.jsonl")
        model = tmp_path / "model"
        model.mkdir()
        (model / PROVENANCE_FILENAME).write_text("{}", encoding="utf-8")
        report = benchmark_finetuned_model(model, general_path=general)
        assert report.details["reason"] == "training provenance unavailable"

    def test_general_benchmark_without_usable_rows_rejects(
        self, tmp_path: Path
    ) -> None:
        """A general benchmark with no usable rows is rejected."""
        general = tmp_path / "general.jsonl"
        general.write_text('{"premise": "only premise"}\n', encoding="utf-8")
        report = benchmark_finetuned_model(
            _trained_model(tmp_path), general_path=general
        )
        assert report.recommendation == "reject"
        assert report.details["reason"] == "general benchmark has no usable rows"

    @patch("director_ai.core.training.finetune_benchmark._evaluate_model")
    def test_rows_shared_with_training_reject_before_evaluation(
        self, mock_eval: MagicMock, tmp_path: Path
    ) -> None:
        """General content or domain clusters shared with training reject."""
        general = _make_benchmark_file(tmp_path, "general.jsonl")
        with general.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "premise": "TRAINING passage 3.",
                        "hypothesis": "training  claim 3.",
                        "label": 1,
                    }
                )
                + "\n"
            )
        domain = tmp_path / "domain.jsonl"
        domain.write_text(
            json.dumps(
                {
                    "premise": "Different text.",
                    "hypothesis": "Different claim.",
                    "label": 0,
                    "source_cluster": "train-doc-5",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        report = benchmark_finetuned_model(
            _trained_model(tmp_path), general_path=general, eval_path=domain
        )
        assert report.recommendation == "reject"
        assert report.details["reason"] == "evaluation rows overlap the training rows"
        assert report.details["shared_with_training"] == {"general": 1, "domain": 1}
        mock_eval.assert_not_called()

    @pytest.mark.parametrize("accuracy", [0.0, float("nan")])
    @patch("director_ai.core.training.finetune_benchmark._evaluate_model")
    def test_zero_or_undefined_general_accuracy_rejects(
        self, mock_eval: MagicMock, accuracy: float, tmp_path: Path
    ) -> None:
        """A zero or undefined general balanced accuracy is rejected."""
        mock_eval.return_value = {"balanced_accuracy": accuracy, "f1": 0.0}
        general = _make_benchmark_file(tmp_path, "general.jsonl")
        report = benchmark_finetuned_model(
            _trained_model(tmp_path), general_path=general, baseline_accuracy=0.0
        )
        assert report.recommendation == "reject"
        assert report.regression_acceptable is False
        assert (
            report.details["reason"] == "general balanced accuracy is zero or undefined"
        )

    @patch("director_ai.core.training.finetune_benchmark._evaluate_model")
    def test_sweep_rejects_candidates_without_a_general_benchmark(
        self, mock_eval: MagicMock, tmp_path: Path
    ) -> None:
        """A sweep without general data deploys no candidate."""
        report = benchmark_model_candidates(
            {"factcg-deberta-v3-large": _trained_model(tmp_path)},
        )
        assert report.best_model_alias == ""
        assert report.results[0].recommendation == "reject"
        assert report.results[0].details["reason"] == "no general benchmark supplied"
        mock_eval.assert_not_called()


class TestThresholdConstants:
    """Regression threshold constants."""

    def test_deploy_threshold(self) -> None:
        """The deploy threshold is three percentage points."""
        assert _DEPLOY_THRESHOLD_PP == 3.0

    def test_reject_threshold(self) -> None:
        """The reject threshold is eight percentage points."""
        assert _REJECT_THRESHOLD_PP == 8.0

    def test_baseline(self) -> None:
        """The baseline accuracy is the FactCG reference value."""
        assert pytest.approx(0.758, abs=0.001) == _BASELINE_ACCURACY


class TestModelBenchmarkSweep:
    """Same-input sweeps across model candidates."""

    @patch("director_ai.core.training.finetune_benchmark._evaluate_model")
    def test_sweep_selects_best_non_rejected_model(
        self, mock_eval: MagicMock, tmp_path: Path
    ) -> None:
        """The best non-rejected candidate wins the sweep."""
        general = _make_benchmark_file(tmp_path, "general.jsonl")
        mock_eval.side_effect = [
            {"balanced_accuracy": 0.74, "f1": 0.72},
            {"balanced_accuracy": 0.80, "f1": 0.78},
        ]
        report = benchmark_model_candidates(
            {
                "factcg-deberta-v3-large": _trained_model(tmp_path, "factcg-model"),
                "roberta-large-mnli": _trained_model(tmp_path, "roberta-model"),
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

    @patch("director_ai.core.training.finetune_benchmark._evaluate_model")
    def test_sweep_records_rejected_candidates(
        self, mock_eval: MagicMock, tmp_path: Path
    ) -> None:
        """Rejected candidates are recorded without a winner."""
        general = _make_benchmark_file(tmp_path, "general.jsonl")
        mock_eval.return_value = {"balanced_accuracy": 0.60, "f1": 0.50}
        report = benchmark_model_candidates(
            {"factcg-deberta-v3-large": _trained_model(tmp_path, "factcg-model")},
            general_path=general,
        )
        assert report.best_model_alias == ""
        assert report.results[0].recommendation == "reject"

    def test_sweep_rejects_unknown_without_experimental_flag(
        self, tmp_path: Path
    ) -> None:
        """Unknown models need the experimental opt-in."""
        report = benchmark_model_candidates(
            {"org/custom-model": tmp_path / "custom-model"},
        )
        assert report.results[0].recommendation == "reject"
        assert "stable fine-tune registry" in report.results[0].error

    def test_sweep_requires_models(self) -> None:
        """A sweep needs at least one model."""
        with pytest.raises(ValueError, match="at least one model"):
            benchmark_model_candidates({})

    def test_report_without_deployable_candidates_has_no_winner(self) -> None:
        """A sweep without deployable candidates names no winner."""
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

    def test_report_summary_includes_result_rows_and_errors(self) -> None:
        """The sweep summary lists results and errors."""
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

    def test_model_result_uses_requested_alias_for_custom_profiles(self) -> None:
        """Custom profiles keep the requested alias."""
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
    """Public exports of the benchmark API."""

    def test_importable_from_core(self) -> None:
        """The benchmark API imports from the core package."""
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

    @patch("director_ai.core.training.finetune_benchmark._evaluate_model")
    def test_deploy_threshold_boundary(
        self, mock_eval: MagicMock, tmp_path: Path
    ) -> None:
        """The deploy threshold separates deploy from domain-only."""
        general = _make_benchmark_file(tmp_path, "general-boundary.jsonl")

        mock_eval.return_value = {"balanced_accuracy": 0.729, "f1": 0.75}
        model = _trained_model(tmp_path)
        deploy = benchmark_finetuned_model(
            model,
            general_path=general,
            baseline_accuracy=0.758,
        )

        mock_eval.return_value = {"balanced_accuracy": 0.727, "f1": 0.75}
        domain_only = benchmark_finetuned_model(
            model,
            general_path=general,
            baseline_accuracy=0.758,
        )

        assert deploy.recommendation == "deploy"
        assert domain_only.recommendation == "deploy_domain_only"

    @patch("director_ai.core.training.finetune_benchmark._evaluate_model")
    def test_reject_threshold_boundary(
        self, mock_eval: MagicMock, tmp_path: Path
    ) -> None:
        """The reject threshold separates domain-only from reject."""
        general = _make_benchmark_file(tmp_path, "general-reject-boundary.jsonl")

        model = _trained_model(tmp_path)
        mock_eval.return_value = {"balanced_accuracy": 0.758 - 0.08, "f1": 0.75}
        domain_only = benchmark_finetuned_model(
            model,
            general_path=general,
            baseline_accuracy=0.758,
        )

        mock_eval.return_value = {"balanced_accuracy": 0.758 - 0.0801, "f1": 0.75}
        reject = benchmark_finetuned_model(
            model,
            general_path=general,
            baseline_accuracy=0.758,
        )

        assert domain_only.recommendation == "deploy_domain_only"
        assert reject.recommendation == "reject"
        assert reject.regression_acceptable is False

    def test_candidate_sweep_records_model_errors_without_aborting(
        self, tmp_path: Path
    ) -> None:
        """A failing candidate is recorded as rejected without stopping the sweep."""
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
