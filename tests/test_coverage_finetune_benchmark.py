# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Fine-tune Benchmark Coverage
"""Multi-angle coverage for fine-tune benchmark pipeline.

Covers: threshold regression testing, model comparison, metric computation,
dataset evaluation, pipeline integration, and performance documentation.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import director_ai.core.training.finetune_benchmark as bench_mod
from director_ai.core.training.finetune_benchmark import (
    ModelBenchmarkReport,
    ModelBenchmarkResult,
    RegressionReport,
    _evaluate_model,
    _load_benchmark_jsonl,
    benchmark_finetuned_model,
    benchmark_model_candidates,
)

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _bench_rows(n: int = 20) -> list[dict]:
    return [
        {"premise": f"P{i}", "hypothesis": f"H{i}", "label": i % 2} for i in range(n)
    ]


# ---------------------------------------------------------------------------
# _load_benchmark_jsonl — uncovered branches
# ---------------------------------------------------------------------------


def test_load_skips_blank_lines(tmp_path):
    f = tmp_path / "blank.jsonl"
    f.write_text(
        json.dumps({"premise": "p", "hypothesis": "h", "label": 0})
        + "\n\n   \n"
        + json.dumps({"premise": "p2", "hypothesis": "h2", "label": 1})
        + "\n",
        encoding="utf-8",
    )
    rows = _load_benchmark_jsonl(f)
    assert len(rows) == 2


def test_load_skips_invalid_json_lines(tmp_path):
    f = tmp_path / "bad.jsonl"
    f.write_text(
        "not json\n"
        + json.dumps({"premise": "p", "hypothesis": "h", "label": 1})
        + "\n",
        encoding="utf-8",
    )
    rows = _load_benchmark_jsonl(f)
    assert len(rows) == 1


def test_load_alternative_field_names(tmp_path):
    f = tmp_path / "alt.jsonl"
    f.write_text(
        json.dumps({"doc": "document", "claim": "claim", "label": 0})
        + "\n"
        + json.dumps({"context": "ctx", "response": "resp", "label": 1})
        + "\n",
        encoding="utf-8",
    )
    rows = _load_benchmark_jsonl(f)
    assert len(rows) == 2
    assert rows[0]["premise"] == "document"
    assert rows[1]["premise"] == "ctx"
    assert rows[1]["hypothesis"] == "resp"


def test_load_skips_row_missing_premise(tmp_path):
    f = tmp_path / "missing.jsonl"
    f.write_text(
        json.dumps({"hypothesis": "h", "label": 0})
        + "\n"
        + json.dumps({"premise": "p", "hypothesis": "h", "label": 1})
        + "\n",
        encoding="utf-8",
    )
    rows = _load_benchmark_jsonl(f)
    assert len(rows) == 1


def test_load_skips_row_missing_label(tmp_path):
    f = tmp_path / "nolabel.jsonl"
    f.write_text(
        json.dumps({"premise": "p", "hypothesis": "h"}) + "\n",
        encoding="utf-8",
    )
    rows = _load_benchmark_jsonl(f)
    assert len(rows) == 0


def test_load_rejects_directory_paths(tmp_path):
    with pytest.raises(FileNotFoundError, match="benchmark data path is not a file"):
        _load_benchmark_jsonl(tmp_path)


# ---------------------------------------------------------------------------
# ModelBenchmarkResult / ModelBenchmarkReport public payloads
# ---------------------------------------------------------------------------


def test_model_benchmark_result_to_dict_preserves_operational_fields():
    result = ModelBenchmarkResult(
        requested_model="factcg-deberta-v3-large",
        alias="factcg-deberta-v3-large",
        model_id="manueldeprada/FactCG-DeBERTa-v3-Large",
        model_path="/models/factcg",
        status="stable",
        template="factcg",
        label_count=2,
        baseline_accuracy=0.758,
        recommended_batch_size=24,
        domain_accuracy=0.91,
        domain_f1=0.88,
        general_accuracy=0.79,
        general_f1=0.77,
        regression_pp=3.2,
        recommendation="deploy",
        elapsed_seconds=12.5,
        error="",
        details={"general_samples": 1000, "domain_samples": 120},
    )

    payload = result.to_dict()

    assert payload == {
        "requested_model": "factcg-deberta-v3-large",
        "alias": "factcg-deberta-v3-large",
        "model_id": "manueldeprada/FactCG-DeBERTa-v3-Large",
        "model_path": "/models/factcg",
        "status": "stable",
        "template": "factcg",
        "label_count": 2,
        "baseline_accuracy": 0.758,
        "recommended_batch_size": 24,
        "domain_accuracy": 0.91,
        "domain_f1": 0.88,
        "general_accuracy": 0.79,
        "general_f1": 0.77,
        "regression_pp": 3.2,
        "recommendation": "deploy",
        "elapsed_seconds": 12.5,
        "error": "",
        "details": {"general_samples": 1000, "domain_samples": 120},
    }


def test_model_benchmark_report_keeps_manual_winner_and_summarizes_errors():
    rejected = ModelBenchmarkResult(
        requested_model="unstable-model",
        alias="unstable-model",
        model_id="unstable-model",
        model_path="/models/unstable",
        status="error",
        template="unknown",
        label_count=0,
        baseline_accuracy=0.0,
        recommended_batch_size=0,
        recommendation="reject",
        error="missing tokenizer",
    )
    deployable = ModelBenchmarkResult(
        requested_model="factcg-deberta-v3-large",
        alias="factcg-deberta-v3-large",
        model_id="manueldeprada/FactCG-DeBERTa-v3-Large",
        model_path="/models/factcg",
        status="stable",
        template="factcg",
        label_count=2,
        baseline_accuracy=0.758,
        recommended_batch_size=24,
        general_accuracy=0.82,
        domain_accuracy=0.90,
        recommendation="deploy",
    )
    report = ModelBenchmarkReport(
        results=[rejected, deployable],
        general_path="/data/general.jsonl",
        eval_path="/data/domain.jsonl",
        generated_at=123.0,
        seed=7,
        best_model_alias="manually-approved",
        best_model_id="manual/model",
        selection_policy="operator_override",
    )

    payload = report.to_dict()
    summary = report.summary()

    assert payload["best_model_alias"] == "manually-approved"
    assert payload["best_model_id"] == "manual/model"
    assert payload["selection_policy"] == "operator_override"
    assert payload["results"][0]["error"] == "missing tokenizer"
    assert "General data: /data/general.jsonl" in summary
    assert "Domain data: /data/domain.jsonl" in summary
    assert "Best model: manually-approved" in summary
    assert (
        "- unstable-model: general=0.0%, domain=0.0%, rec=reject error=missing tokenizer"
        in summary
    )
    assert (
        "- factcg-deberta-v3-large: general=82.0%, domain=90.0%, rec=deploy" in summary
    )


# ---------------------------------------------------------------------------
# _evaluate_model — mocked heavy deps
# ---------------------------------------------------------------------------


def _make_mock_model_and_tokenizer(n_samples: int, label: int = 0):

    tokenizer = MagicMock()
    tokenizer.sep_token = "[SEP]"
    tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
    encoded = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
    encoded["input_ids"].to = MagicMock(return_value=encoded["input_ids"])
    encoded["attention_mask"].to = MagicMock(return_value=encoded["attention_mask"])
    tokenizer.side_effect = lambda *a, **kw: {
        k: _make_tensor(n_samples) for k in ("input_ids", "attention_mask")
    }

    model = MagicMock()
    model.eval = MagicMock(return_value=None)
    model.to = MagicMock(return_value=model)

    import torch

    logits = torch.zeros(n_samples, 2)
    logits[:, label] = 10.0
    output = MagicMock()
    output.logits = logits
    model.return_value = output

    return tokenizer, model


def _make_tensor(n):
    import torch

    t = torch.zeros(n, 10, dtype=torch.long)
    t.to = MagicMock(return_value=t)
    return t


def _make_transformers_mocks(n: int, all_label_0: bool = True):
    import torch

    tokenizer = MagicMock()
    tokenizer.sep_token = "[SEP]"

    logits = torch.zeros(n, 2)
    logits[:, 0 if all_label_0 else 1] = 10.0
    output = MagicMock()
    output.logits = logits

    def tokenizer_call(*args, **kwargs):
        result = MagicMock()
        t = torch.zeros(n, 5, dtype=torch.long)
        t.to = MagicMock(return_value=t)
        result.items.return_value = [("input_ids", t)]
        return result

    tokenizer.side_effect = tokenizer_call

    model = MagicMock()
    model.eval.return_value = None
    model.to.return_value = model
    model.return_value = output

    return tokenizer, model


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_evaluate_model_non_factcg():
    import sys

    n = 10
    samples = [
        {"premise": f"p{i}", "hypothesis": f"h{i}", "label": 0} for i in range(n)
    ]
    tokenizer, model = _make_transformers_mocks(n)

    mock_transformers = MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
    mock_transformers.AutoModelForSequenceClassification.from_pretrained.return_value = model

    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        with patch("torch.cuda.is_available", return_value=False):
            result = _evaluate_model("/some/model", samples, batch_size=48)

    mock_transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
        "/some/model",
        revision=None,
    )
    mock_transformers.AutoModelForSequenceClassification.from_pretrained.assert_called_once_with(
        "/some/model",
        revision=None,
    )
    assert "balanced_accuracy" in result
    assert "f1" in result


def test_evaluate_model_rejects_unpinned_remote() -> None:
    with pytest.raises(ValueError, match="requires an explicit immutable revision"):
        _evaluate_model("unverified-org/unverified-model", [], batch_size=48)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_evaluate_model_factcg_path():
    import sys

    n = 4
    samples = [
        {"premise": f"p{i}", "hypothesis": f"h{i}", "label": 0} for i in range(n)
    ]
    tokenizer, model = _make_transformers_mocks(n)

    mock_transformers = MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
    mock_transformers.AutoModelForSequenceClassification.from_pretrained.return_value = model

    mock_finetune = MagicMock()
    mock_finetune._FACTCG_TEMPLATE = "{premise} {hypothesis}"

    with patch.dict(
        sys.modules,
        {
            "transformers": mock_transformers,
            "director_ai.core.training.finetune": mock_finetune,
        },
    ):
        with patch("torch.cuda.is_available", return_value=False):
            result = _evaluate_model("/path/to/factcg-model", samples, batch_size=48)

    assert "balanced_accuracy" in result


def test_evaluate_model_raises_on_missing_transformers():
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("no transformers")
        return real_import(name, *args, **kwargs)

    samples = [{"premise": "p", "hypothesis": "h", "label": 0}]
    with patch("builtins.__import__", side_effect=mock_import):
        with pytest.raises(ImportError, match="pip install director-ai"):
            _evaluate_model("/model", samples)


class _FakeTensor:
    def __init__(self, data):
        self._data = data

    def to(self, device):
        del device
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self

    def flatten(self):
        if self._data and isinstance(self._data[0], list):
            return [item for row in self._data for item in row]
        return list(self._data)


class _FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False


class _FakeTorch(types.ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
        )

    def device(self, name):
        return name

    def no_grad(self):
        return _FakeNoGrad()

    def argmax(self, logits, dim=-1):
        del dim
        winners = [
            max(range(len(row)), key=lambda index: row[index]) for row in logits._data
        ]
        return _FakeTensor(winners)


class _FakeTokenizer:
    sep_token = "[SEP]"

    def __init__(self):
        self.batch_texts: list[list[str]] = []

    def __call__(self, batch_texts, **kwargs):
        assert kwargs["truncation"] is True
        assert kwargs["padding"] is True
        assert kwargs["max_length"] == 512
        assert kwargs["return_tensors"] == "pt"
        self.batch_texts.append(list(batch_texts))
        return {"input_ids": _FakeTensor([[1, 2, 3] for _ in batch_texts])}


class _FakeSequenceClassifier:
    def __init__(self, predictions):
        self._predictions = list(predictions)
        self.to_calls: list[str] = []
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def to(self, device):
        self.to_calls.append(str(device))
        return self

    def __call__(self, **encodings):
        batch_size = len(encodings["input_ids"]._data)
        batch_preds = [self._predictions.pop(0) for _ in range(batch_size)]
        logits = [[0.1, 0.9] if pred else [0.9, 0.1] for pred in batch_preds]
        return types.SimpleNamespace(logits=_FakeTensor(logits))


def test_evaluate_model_inference_path_without_optional_ml_stack(monkeypatch):
    tokenizer = _FakeTokenizer()
    model = _FakeSequenceClassifier(predictions=[0, 1, 0, 1])
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=MagicMock(return_value=tokenizer),
    )
    fake_transformers.AutoModelForSequenceClassification = types.SimpleNamespace(
        from_pretrained=MagicMock(return_value=model),
    )
    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        "director_ai.core.training.finetune_benchmark.select_torch_device",
        lambda: "cpu",
        raising=False,
    )
    monkeypatch.setattr(
        "director_ai.core.training.finetune_benchmark.release_torch_cuda",
        lambda: None,
        raising=False,
    )
    samples = [
        {"premise": "p0", "hypothesis": "h0", "label": 0},
        {"premise": "p1", "hypothesis": "h1", "label": 1},
        {"premise": "p2", "hypothesis": "h2", "label": 0},
        {"premise": "p3", "hypothesis": "h3", "label": 1},
    ]

    result = _evaluate_model("/local/model", samples, batch_size=2)

    assert result == {"balanced_accuracy": 1.0, "f1": 1.0}
    assert model.eval_called is True
    assert model.to_calls == ["cpu", "cpu"]
    assert tokenizer.batch_texts == [
        ["p0 [SEP] h0", "p1 [SEP] h1"],
        ["p2 [SEP] h2", "p3 [SEP] h3"],
    ]


# ---------------------------------------------------------------------------
# benchmark_finetuned_model — uncovered branches
# ---------------------------------------------------------------------------


@patch("director_ai.core.training.finetune_benchmark._evaluate_model")
def test_benchmark_uses_pkg_data_when_general_path_none(mock_eval, tmp_path):
    mock_eval.return_value = {"balanced_accuracy": 0.76, "f1": 0.75}
    candidate = tmp_path / "aggrefact_benchmark_1k.jsonl"
    _write_jsonl(candidate, _bench_rows())

    with patch(
        "director_ai.core.training.finetune_benchmark.Path.__file__",
        new_callable=lambda: property(lambda self: None),
        create=True,
    ):
        pass

    pkg_dir = Path(__file__).parent.parent / "src" / "director_ai" / "core"
    data_file = pkg_dir / "data" / "aggrefact_benchmark_1k.jsonl"

    if data_file.exists():
        report = benchmark_finetuned_model("/fake/model", general_path=None)
        assert report.general_accuracy > 0 or report.details.get("general_skipped")
    else:
        report = benchmark_finetuned_model("/fake/model", general_path=None)
        assert report.recommendation == "deploy_domain_only"
        assert report.details.get("reason") == "no general benchmark available"


@patch("director_ai.core.training.finetune_benchmark._evaluate_model")
def test_benchmark_general_path_none_pkg_file_exists(mock_eval, tmp_path):
    mock_eval.return_value = {"balanced_accuracy": 0.76, "f1": 0.75}

    bench_data = _bench_rows(50)
    package_root = tmp_path / "director_ai" / "core"
    training_dir = package_root / "training"
    data_dir = package_root / "data"
    training_dir.mkdir(parents=True)
    data_dir.mkdir()
    monkeypatch_target = training_dir / "finetune_benchmark.py"
    monkeypatch_target.write_text("# synthetic module location\n", encoding="utf-8")
    bench_file = data_dir / "aggrefact_benchmark_1k.jsonl"
    _write_jsonl(bench_file, bench_data)

    with patch.object(
        bench_mod,
        "__file__",
        str(monkeypatch_target),
    ):
        report = benchmark_finetuned_model(
            "/fake/model",
            general_path=None,
            baseline_accuracy=0.758,
        )

    assert report.general_accuracy == pytest.approx(0.76)
    assert report.details["general_samples"] == 50
    assert report.recommendation == "deploy"


@patch("director_ai.core.training.finetune_benchmark._evaluate_model")
def test_benchmark_empty_general_file_triggers_skip(mock_eval, tmp_path):
    f = tmp_path / "empty.jsonl"
    f.write_text("", encoding="utf-8")

    report = benchmark_finetuned_model("/fake/model", general_path=f)
    assert report.general_accuracy == 0.0
    assert report.recommendation == "deploy_domain_only"
    assert report.details.get("reason") == "no general benchmark available"
    mock_eval.assert_not_called()


@patch("director_ai.core.training.finetune_benchmark._evaluate_model")
def test_benchmark_empty_domain_file_skips_domain(mock_eval, tmp_path):
    domain = tmp_path / "domain.jsonl"
    domain.write_text("", encoding="utf-8")
    general = tmp_path / "general.jsonl"
    _write_jsonl(general, _bench_rows())
    mock_eval.return_value = {"balanced_accuracy": 0.76, "f1": 0.75}

    report = benchmark_finetuned_model(
        "/fake/model",
        general_path=general,
        eval_path=domain,
        baseline_accuracy=0.758,
    )
    assert report.domain_accuracy == 0.0
    assert report.general_accuracy == pytest.approx(0.76)


@patch("director_ai.core.training.finetune_benchmark._evaluate_model")
def test_benchmark_general_samples_recorded_in_details(mock_eval, tmp_path):
    mock_eval.return_value = {"balanced_accuracy": 0.77, "f1": 0.75}
    general = tmp_path / "general.jsonl"
    rows = _bench_rows(30)
    _write_jsonl(general, rows)

    report = benchmark_finetuned_model("/fake/model", general_path=general)
    assert report.details.get("general_samples") == 30


@patch("director_ai.core.training.finetune_benchmark._evaluate_model")
def test_benchmark_domain_samples_recorded_in_details(mock_eval, tmp_path):
    mock_eval.return_value = {"balanced_accuracy": 0.80, "f1": 0.78}
    domain = tmp_path / "domain.jsonl"
    rows = _bench_rows(15)
    _write_jsonl(domain, rows)

    report = benchmark_finetuned_model("/fake/model", eval_path=domain)
    assert report.details.get("domain_samples") == 15


@patch("director_ai.core.training.finetune_benchmark._evaluate_model")
def test_benchmark_deploy_threshold_boundary(mock_eval, tmp_path):
    general = tmp_path / "g.jsonl"
    _write_jsonl(general, _bench_rows())

    # 2.9pp regression → deploy (< 3pp)
    mock_eval.return_value = {"balanced_accuracy": 0.729, "f1": 0.75}
    report = benchmark_finetuned_model(
        "/fake/model", general_path=general, baseline_accuracy=0.758
    )
    assert report.recommendation == "deploy"

    # 3.1pp regression → deploy_domain_only (> 3pp)
    mock_eval.return_value = {"balanced_accuracy": 0.727, "f1": 0.75}
    report = benchmark_finetuned_model(
        "/fake/model", general_path=general, baseline_accuracy=0.758
    )
    assert report.recommendation == "deploy_domain_only"


@patch("director_ai.core.training.finetune_benchmark._evaluate_model")
def test_benchmark_reject_threshold_boundary(mock_eval, tmp_path):
    general = tmp_path / "g.jsonl"
    _write_jsonl(general, _bench_rows())

    mock_eval.return_value = {"balanced_accuracy": 0.758 - 0.08, "f1": 0.75}
    report = benchmark_finetuned_model(
        "/fake/model", general_path=general, baseline_accuracy=0.758
    )
    assert report.recommendation == "deploy_domain_only"

    mock_eval.return_value = {"balanced_accuracy": 0.758 - 0.0801, "f1": 0.75}
    report = benchmark_finetuned_model(
        "/fake/model", general_path=general, baseline_accuracy=0.758
    )
    assert report.recommendation == "reject"
    assert not report.regression_acceptable


@patch("director_ai.core.training.finetune_benchmark._evaluate_model")
def test_benchmark_general_skipped_flag_set(mock_eval, tmp_path):
    report = benchmark_finetuned_model(
        "/fake/model",
        general_path=None,
        baseline_accuracy=0.758,
    )
    assert report.details.get("general_skipped") is True
    mock_eval.assert_not_called()


@patch("director_ai.core.training.finetune_benchmark._evaluate_model")
def test_candidate_sweep_selects_best_deployable_result(mock_eval, tmp_path):
    general = tmp_path / "general.jsonl"
    domain = tmp_path / "domain.jsonl"
    _write_jsonl(general, _bench_rows())
    _write_jsonl(domain, _bench_rows(10))
    mock_eval.side_effect = [
        {"balanced_accuracy": 0.78, "f1": 0.76},
        {"balanced_accuracy": 0.82, "f1": 0.80},
        {"balanced_accuracy": 0.79, "f1": 0.78},
        {"balanced_accuracy": 0.87, "f1": 0.85},
    ]

    report = benchmark_model_candidates(
        {
            "factcg-deberta-v3-large": tmp_path / "factcg-model",
            "roberta-large-mnli": tmp_path / "roberta-model",
        },
        general_path=general,
        eval_path=domain,
        batch_size=4,
        allow_experimental=True,
        seed=99,
    )

    assert report.best_model_alias == "roberta-large-mnli"
    assert report.best_model_id
    assert report.general_path == str(general)
    assert report.eval_path == str(domain)
    assert report.seed == 99
    assert [result.requested_model for result in report.results] == [
        "factcg-deberta-v3-large",
        "roberta-large-mnli",
    ]
    assert report.results[0].details["general_samples"] == 20
    assert report.results[1].recommended_batch_size > 0


@patch("director_ai.core.training.finetune_benchmark._evaluate_model")
def test_candidate_sweep_records_model_errors_without_aborting(mock_eval, tmp_path):
    del mock_eval

    report = benchmark_model_candidates(
        {"org/custom-model": tmp_path / "custom-model"},
        allow_experimental=False,
        batch_size=8,
    )

    assert report.best_model_alias == ""
    assert len(report.results) == 1
    result = report.results[0]
    assert result.requested_model == "org/custom-model"
    assert result.alias == "org/custom-model"
    assert result.model_path == str(tmp_path / "custom-model")
    assert result.status == "error"
    assert result.recommendation == "reject"
    assert result.recommended_batch_size == 8
    assert "stable fine-tune registry" in result.error


def test_candidate_sweep_requires_at_least_one_model() -> None:
    with pytest.raises(ValueError, match="model_artifacts"):
        benchmark_model_candidates({})


# ---------------------------------------------------------------------------
# RegressionReport.summary
# ---------------------------------------------------------------------------


def test_summary_contains_all_fields():
    r = RegressionReport(
        domain_accuracy=0.90,
        general_accuracy=0.72,
        baseline_accuracy=0.758,
        regression_pp=-3.8,
        recommendation="deploy_domain_only",
    )
    s = r.summary()
    assert "90.0%" in s
    assert "72.0%" in s
    assert "75.8%" in s
    assert "-3.8pp" in s
    assert "deploy_domain_only" in s


def test_summary_positive_regression():
    r = RegressionReport(
        general_accuracy=0.80,
        baseline_accuracy=0.758,
        regression_pp=4.2,
        recommendation="deploy",
    )
    s = r.summary()
    assert "+4.2pp" in s
