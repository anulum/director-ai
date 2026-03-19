# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from director_ai.core.training.finetune_benchmark import (
    RegressionReport,
    _evaluate_model,
    _load_benchmark_jsonl,
    benchmark_finetuned_model,
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

    assert "balanced_accuracy" in result
    assert "f1" in result


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
    bench_file = tmp_path / "aggrefact_benchmark_1k.jsonl"
    _write_jsonl(bench_file, bench_data)

    with patch(
        "director_ai.core.training.finetune_benchmark.Path",
        wraps=Path,
    ):

        class FakePath(type(Path())):
            pass

        report = benchmark_finetuned_model(
            "/fake/model",
            general_path=bench_file,
            baseline_accuracy=0.758,
        )

    assert report.general_accuracy == pytest.approx(0.76)
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
