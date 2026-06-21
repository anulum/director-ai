# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Focused CLI benchmark/training path tests.

These tests exercise command handlers with deterministic fakes so the CLI
contracts are covered without invoking external model, benchmark, or export
toolchains.
"""

from __future__ import annotations

import json
import random
import sys
from types import ModuleType, SimpleNamespace

import pytest

from director_ai import _cli_bench


def test_eval_quantize_rejects_unknown_mode(capsys):
    with pytest.raises(SystemExit) as exc_info:
        _cli_bench._cmd_eval(["--quantize", "bf16"])

    assert exc_info.value.code == 1
    assert "int8" in capsys.readouterr().out


def test_eval_quantize_exports_with_selected_mode(monkeypatch, capsys):
    calls: list[str | None] = []
    fake_nli = ModuleType("director_ai.core.scoring.nli")
    fake_nli.export_onnx = lambda quantize=None, **_kwargs: calls.append(quantize)
    monkeypatch.setitem(sys.modules, "director_ai.core.scoring.nli", fake_nli)

    _cli_bench._cmd_eval(["--quantize", "fp16"])

    assert calls == ["fp16"]
    assert "Export complete" in capsys.readouterr().out


def test_eval_ignores_unknown_tokens_before_quantize(monkeypatch, capsys):
    calls: list[str | None] = []
    fake_nli = ModuleType("director_ai.core.scoring.nli")
    fake_nli.export_onnx = lambda quantize=None, **_kwargs: calls.append(quantize)
    monkeypatch.setitem(sys.modules, "director_ai.core.scoring.nli", fake_nli)

    _cli_bench._cmd_eval(["--ignored", "--quantize", "int8"])

    assert calls == ["int8"]
    assert "Export complete" in capsys.readouterr().out


def test_eval_rejects_invalid_max_samples(capsys):
    with pytest.raises(SystemExit) as exc_info:
        _cli_bench._cmd_eval(["--max-samples", "many"])

    assert exc_info.value.code == 1
    assert "invalid --max-samples value: many" in capsys.readouterr().out


def test_eval_reports_missing_benchmarks_package(monkeypatch, capsys):
    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "benchmarks.run_all":
            raise ImportError("benchmarks unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(SystemExit) as exc_info:
        _cli_bench._cmd_eval([])

    assert exc_info.value.code == 1
    assert "benchmarks package not found" in capsys.readouterr().out


def test_bench_warn_only_failure_is_reported_but_not_fatal(monkeypatch, capsys):
    def ok() -> None:
        return None

    def latency_failure() -> None:
        raise AssertionError("p95 exceeded")

    latency_failure.__name__ = "test_latency_ceiling"
    fake_suite = ModuleType("benchmarks.regression_suite")
    fake_suite.test_heuristic_accuracy = ok
    fake_suite.test_streaming_stability = ok
    fake_suite.test_latency_ceiling = latency_failure
    fake_suite.test_metrics_integrity = ok
    fake_suite.test_evidence_schema = ok
    fake_suite.test_e2e_heuristic_delta = ok
    fake_suite.test_false_halt_rate = ok
    fake_benchmarks = ModuleType("benchmarks")
    fake_benchmarks.regression_suite = fake_suite
    monkeypatch.setitem(sys.modules, "benchmarks", fake_benchmarks)
    monkeypatch.setitem(sys.modules, "benchmarks.regression_suite", fake_suite)

    _cli_bench._cmd_bench(["--dataset", "regression"])

    out = capsys.readouterr().out
    assert "WARN: test_latency_ceiling" in out
    assert "1 warned, 0 failed" in out


def test_bench_seed_and_max_samples_truncate_regression_suite(
    monkeypatch,
    capsys,
):
    calls: list[str] = []
    seeds: list[int] = []

    def first() -> None:
        calls.append("first")

    def second() -> None:
        calls.append("second")

    first.__name__ = "test_heuristic_accuracy"
    second.__name__ = "test_streaming_stability"
    fake_suite = ModuleType("benchmarks.regression_suite")
    fake_suite.test_heuristic_accuracy = first
    fake_suite.test_streaming_stability = second
    fake_suite.test_latency_ceiling = second
    fake_suite.test_metrics_integrity = second
    fake_suite.test_evidence_schema = second
    fake_suite.test_e2e_heuristic_delta = second
    fake_suite.test_false_halt_rate = second
    fake_benchmarks = ModuleType("benchmarks")
    fake_benchmarks.regression_suite = fake_suite
    monkeypatch.setitem(sys.modules, "benchmarks", fake_benchmarks)
    monkeypatch.setitem(sys.modules, "benchmarks.regression_suite", fake_suite)
    monkeypatch.setattr(random, "seed", lambda seed: seeds.append(seed))

    _cli_bench._cmd_bench(["--ignored", "--seed", "17", "--max-samples", "1"])

    assert seeds == [17]
    assert calls == ["first"]
    assert "1 passed, 0 warned, 0 failed" in capsys.readouterr().out


def test_bench_rejects_invalid_seed_and_max_samples(capsys):
    with pytest.raises(SystemExit) as seed_exc:
        _cli_bench._cmd_bench(["--seed", "not-an-int"])

    assert seed_exc.value.code == 1
    assert "invalid --seed value: not-an-int" in capsys.readouterr().out

    with pytest.raises(SystemExit) as max_exc:
        _cli_bench._cmd_bench(["--max-samples", "not-an-int"])

    assert max_exc.value.code == 1
    assert "invalid --max-samples value: not-an-int" in capsys.readouterr().out


def test_bench_rejects_unknown_dataset(capsys):
    with pytest.raises(SystemExit) as exc_info:
        _cli_bench._cmd_bench(["--dataset", "external"])

    assert exc_info.value.code == 1
    assert "Unknown dataset 'external'" in capsys.readouterr().out


def test_bench_reports_missing_benchmarks_package(monkeypatch, capsys):
    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "benchmarks" and fromlist == ("regression_suite",):
            raise ImportError("benchmarks unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(SystemExit) as exc_info:
        _cli_bench._cmd_bench(["--dataset", "regression"])

    assert exc_info.value.code == 1
    assert "benchmarks package not found" in capsys.readouterr().out


def test_bench_failed_required_test_exits_and_writes_report(
    monkeypatch,
    tmp_path,
    capsys,
):
    def ok() -> None:
        return None

    def e2e_failure() -> None:
        raise AssertionError("delta regressed")

    e2e_failure.__name__ = "test_e2e_heuristic_delta"
    fake_suite = ModuleType("benchmarks.regression_suite")
    fake_suite.test_heuristic_accuracy = ok
    fake_suite.test_latency_ceiling = ok
    fake_suite.test_metrics_integrity = ok
    fake_suite.test_evidence_schema = ok
    fake_suite.test_e2e_heuristic_delta = e2e_failure
    fake_suite.test_false_halt_rate = ok
    fake_suite.test_streaming_stability = ok
    fake_benchmarks = ModuleType("benchmarks")
    fake_benchmarks.regression_suite = fake_suite
    monkeypatch.setitem(sys.modules, "benchmarks", fake_benchmarks)
    monkeypatch.setitem(sys.modules, "benchmarks.regression_suite", fake_suite)
    output = tmp_path / "bench.json"

    with pytest.raises(SystemExit) as exc_info:
        _cli_bench._cmd_bench(["--dataset", "e2e", "--output", str(output)])

    assert exc_info.value.code == 1
    saved = json.loads(output.read_text())
    assert saved["dataset"] == "e2e"
    assert saved["failed"] == 1
    assert saved["results"] == [
        {
            "test": "test_e2e_heuristic_delta",
            "status": "failed",
            "error": "delta regressed",
        },
    ]
    out = capsys.readouterr().out
    assert "FAIL: test_e2e_heuristic_delta: delta regressed" in out
    assert "Results written to" in out


def test_tune_prints_usage_without_args(capsys):
    with pytest.raises(SystemExit) as exc_info:
        _cli_bench._cmd_tune([])

    assert exc_info.value.code == 1
    assert "Usage: director-ai tune" in capsys.readouterr().out


def test_tune_unknown_option_without_dataset_reports_missing_input(capsys):
    with pytest.raises(SystemExit) as exc_info:
        _cli_bench._cmd_tune(["--unknown-option"])

    assert exc_info.value.code == 1
    assert "missing dataset file" in capsys.readouterr().out


def test_tune_rejects_missing_file(tmp_path, capsys):
    missing = tmp_path / "missing.jsonl"

    with pytest.raises(SystemExit) as exc_info:
        _cli_bench._cmd_tune(["--dataset", str(missing)])

    assert exc_info.value.code == 1
    assert f"file not found: {missing}" in capsys.readouterr().out


def test_tune_skips_invalid_rows_and_writes_profile_overlay(
    monkeypatch,
    tmp_path,
    capsys,
):
    dataset = tmp_path / "labels.jsonl"
    dataset.write_text(
        "\n".join(
            [
                "",
                "{not json",
                '{"prompt":"p"}',
                '{"prompt":"p","response":"r","label":true}',
            ],
        ),
        encoding="utf-8",
    )
    output = tmp_path / "overlay.yaml"
    observed: dict[str, object] = {}
    fake_tuner = ModuleType("director_ai.core.training.tuner")

    result = SimpleNamespace(
        threshold=0.7,
        w_logic=0.4,
        w_fact=0.6,
        balanced_accuracy=0.8,
        precision=0.75,
        recall=0.5,
        f1=0.6,
        samples=1,
    )

    def fake_tune(samples):
        observed["samples"] = samples
        return result

    def fake_overlay(tune_result, *, profile, base_profile):
        observed["overlay"] = (tune_result, profile, base_profile)
        return "overlay: yes\n"

    fake_tuner.tune = fake_tune
    fake_tuner.format_confidence_report = lambda tune_result: "confidence: ok"
    fake_tuner.format_profile_overlay = fake_overlay
    monkeypatch.setitem(sys.modules, "director_ai.core.training.tuner", fake_tuner)

    _cli_bench._cmd_tune(
        [
            "--dataset",
            str(dataset),
            "--profile",
            "strict",
            "--output",
            str(output),
        ],
    )

    assert observed["samples"] == [{"prompt": "p", "response": "r", "label": True}]
    assert observed["overlay"] == (result, "strict_tuned", "strict")
    assert output.read_text(encoding="utf-8") == "overlay: yes\n"
    out = capsys.readouterr().out
    assert "skipping line 2" in out
    assert "missing required fields" in out
    assert "Best threshold: 0.7" in out


def test_tune_exits_when_no_valid_samples(tmp_path, capsys):
    dataset = tmp_path / "empty.jsonl"
    dataset.write_text('{"prompt":"p"}\n', encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        _cli_bench._cmd_tune([str(dataset)])

    assert exc_info.value.code == 1
    assert "no valid samples found" in capsys.readouterr().out


def test_finetune_prints_usage_without_args(capsys):
    with pytest.raises(SystemExit) as exc_info:
        _cli_bench._cmd_finetune([])

    assert exc_info.value.code == 1
    assert "Usage: director-ai finetune" in capsys.readouterr().out


def test_finetune_rejects_missing_train_file(tmp_path, capsys):
    missing = tmp_path / "train.jsonl"

    with pytest.raises(SystemExit) as exc_info:
        _cli_bench._cmd_finetune([str(missing)])

    assert exc_info.value.code == 1
    assert f"file not found: {missing}" in capsys.readouterr().out


def test_finetune_success_prints_all_optional_result_fields(
    monkeypatch,
    tmp_path,
    capsys,
):
    train_file = tmp_path / "train.jsonl"
    eval_file = tmp_path / "eval.jsonl"
    train_file.write_text('{"premise":"p","hypothesis":"h","label":1}\n')
    eval_file.write_text('{"premise":"p","hypothesis":"h","label":1}\n')
    observed: dict[str, object] = {}

    class FakeConfig:
        output_dir = "./director-finetuned"
        epochs = 3
        learning_rate = 2e-5
        batch_size = 16
        base_model = "base"
        mix_general_data = False
        general_data_path = None
        early_stopping_patience = None
        class_weighted_loss = False
        auto_benchmark = False
        auto_onnx_export = False

    def fake_finetune(path, *, eval_path, config):
        observed["path"] = path
        observed["eval_path"] = eval_path
        observed["config"] = config
        return SimpleNamespace(
            output_dir="/models/domain",
            train_samples=42,
            mixed_general_samples=8,
            epochs_completed=4,
            final_loss=0.125,
            eval_samples=12,
            best_balanced_accuracy=0.875,
            regression_report={"regression_pp": 1.5, "recommendation": "ship"},
            onnx_path="/models/domain/model.onnx",
        )

    fake_module = ModuleType("director_ai.core.training.finetune")
    fake_module.FinetuneConfig = FakeConfig
    fake_module.finetune_nli = fake_finetune
    monkeypatch.setitem(sys.modules, "director_ai.core.training.finetune", fake_module)

    _cli_bench._cmd_finetune(
        [
            str(train_file),
            "--eval",
            str(eval_file),
            "--output",
            "/models/domain",
            "--epochs",
            "4",
            "--lr",
            "0.001",
            "--batch-size",
            "2",
            "--base-model",
            "domain-base",
            "--mix-general",
            "--general-data",
            "general.jsonl",
            "--early-stopping",
            "3",
            "--class-weights",
            "--auto-benchmark",
            "--auto-onnx",
        ],
    )

    config = observed["config"]
    assert observed["path"] == str(train_file)
    assert observed["eval_path"] == str(eval_file)
    assert config.output_dir == "/models/domain"
    assert config.epochs == 4
    assert config.learning_rate == 0.001
    assert config.batch_size == 2
    assert config.base_model == "domain-base"
    assert config.mix_general_data is True
    assert config.general_data_path == "general.jsonl"
    assert config.early_stopping_patience == 3
    assert config.class_weighted_loss is True
    assert config.auto_benchmark is True
    assert config.auto_onnx_export is True
    out = capsys.readouterr().out
    assert "Mixed general:   8" in out
    assert "Best bal. acc:   87.5%" in out
    assert "Regression:      +1.5pp" in out
    assert "ONNX export:     /models/domain/model.onnx" in out


def test_finetune_rejects_unknown_option(tmp_path, monkeypatch, capsys):
    train_file = tmp_path / "train.jsonl"
    train_file.write_text('{"premise":"p","hypothesis":"h","label":1}\n')

    class FakeConfig:
        pass

    fake_module = ModuleType("director_ai.core.training.finetune")
    fake_module.FinetuneConfig = FakeConfig
    fake_module.finetune_nli = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "director_ai.core.training.finetune", fake_module)

    with pytest.raises(SystemExit) as exc_info:
        _cli_bench._cmd_finetune([str(train_file), "--unsupported"])

    assert exc_info.value.code == 1
    assert "Unknown option: --unsupported" in capsys.readouterr().out


def test_validate_data_prints_warnings_and_exits_on_errors(
    monkeypatch,
    tmp_path,
    capsys,
):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text('{"premise":"p","hypothesis":"h","label":1}\n')

    report = SimpleNamespace(
        warnings=["minor class imbalance"],
        errors=["invalid label"],
        summary=lambda: "1 valid, 1 invalid",
    )
    fake_module = ModuleType("director_ai.core.training.finetune_validator")
    fake_module.validate_finetune_data = lambda path: report
    monkeypatch.setitem(
        sys.modules,
        "director_ai.core.training.finetune_validator",
        fake_module,
    )

    with pytest.raises(SystemExit) as exc_info:
        _cli_bench._cmd_validate_data([str(data_file)])

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "1 valid, 1 invalid" in out
    assert "minor class imbalance" in out
    assert "invalid label" in out


def test_validate_data_prints_usage_without_args(capsys):
    with pytest.raises(SystemExit) as exc_info:
        _cli_bench._cmd_validate_data([])

    assert exc_info.value.code == 1
    assert "Usage: director-ai validate-data" in capsys.readouterr().out


def test_validate_data_rejects_missing_file(tmp_path, capsys):
    missing = tmp_path / "train.jsonl"

    with pytest.raises(SystemExit) as exc_info:
        _cli_bench._cmd_validate_data([str(missing)])

    assert exc_info.value.code == 1
    assert f"file not found: {missing}" in capsys.readouterr().out


def test_export_forwards_model_name_and_ignores_unrecognised_tokens(
    monkeypatch,
    capsys,
):
    calls: list[dict[str, object]] = []
    fake_nli = ModuleType("director_ai.core.scoring.nli")
    fake_nli.export_onnx = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "director_ai.core.scoring.nli", fake_nli)

    _cli_bench._cmd_export(
        [
            "--format",
            "onnx",
            "--model",
            "custom-nli",
            "--output",
            "out-dir",
            "--quantize",
            "int8",
            "--ignored",
        ],
    )

    assert calls == [
        {"model_name": "custom-nli", "output_dir": "out-dir", "quantize": "int8"},
    ]
    assert "out-dir" in capsys.readouterr().out


def test_export_tensorrt_uses_no_fp16(monkeypatch, capsys):
    calls: list[dict[str, object]] = []
    fake_nli = ModuleType("director_ai.core.scoring.nli")
    fake_nli.export_tensorrt = lambda **kwargs: calls.append(kwargs) or "cache-dir"
    monkeypatch.setitem(sys.modules, "director_ai.core.scoring.nli", fake_nli)

    _cli_bench._cmd_export(
        ["--format", "tensorrt", "--onnx-dir", "onnx", "--output", "trt", "--no-fp16"],
    )

    assert calls == [{"onnx_dir": "onnx", "output_dir": "trt", "fp16": False}]
    assert "cache-dir" in capsys.readouterr().out


def test_export_rejects_unknown_format(capsys):
    with pytest.raises(SystemExit) as exc_info:
        _cli_bench._cmd_export(["--format", "gguf"])

    assert exc_info.value.code == 1
    assert "Unknown format 'gguf'" in capsys.readouterr().out


def test_eval_runs_suite_and_writes_results(monkeypatch, tmp_path, capsys):
    fake_run_all = ModuleType("benchmarks.run_all")
    fake_run_all._run_suite = lambda model, max_samples: {
        "model": model,
        "max_samples": max_samples,
        "accuracy": 0.91,
    }
    fake_run_all._print_comparison_table = lambda table: print(
        json.dumps(table, sort_keys=True),
    )
    fake_benchmarks = ModuleType("benchmarks")
    monkeypatch.setitem(sys.modules, "benchmarks", fake_benchmarks)
    monkeypatch.setitem(sys.modules, "benchmarks.run_all", fake_run_all)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    output = tmp_path / "results.json"

    _cli_bench._cmd_eval(
        [
            "--dataset",
            "aggrefact",
            "--max-samples",
            "5",
            "--model",
            "judge",
            "--output",
            str(output),
        ],
    )

    saved = json.loads(output.read_text())
    assert saved == {"model": "judge", "max_samples": 5, "accuracy": 0.91}
    out = capsys.readouterr().out
    assert "HF_TOKEN not set" in out
    assert '"judge"' in out
