# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Strict unit guards for benchmark-family optional CLI branches."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from director_ai import _cli_bench


class _RunAllModule(ModuleType):
    _run_suite: Callable[[str | None, int | None], dict[str, float]]
    _print_comparison_table: Callable[[dict[str, object]], None]


class _TunerModule(ModuleType):
    tune: Callable[[list[dict[str, object]]], SimpleNamespace]
    format_confidence_report: Callable[[SimpleNamespace], str]
    format_profile_overlay: Callable[..., str]


class _FinetuneModule(ModuleType):
    FinetuneConfig: type[object]
    finetune_nli: Callable[..., SimpleNamespace]


class _ValidatorModule(ModuleType):
    validate_finetune_data: Callable[[str], SimpleNamespace]


def test_eval_runs_suite_without_warning_or_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Eval should skip AggreFact token warning and output writing when absent."""
    fake_run_all = _RunAllModule("benchmarks.run_all")

    def run_suite(model: str | None, max_samples: int | None) -> dict[str, float]:
        return {"accuracy": 0.5}

    def print_comparison_table(table: dict[str, object]) -> None:
        print("comparison")

    fake_run_all._run_suite = run_suite
    fake_run_all._print_comparison_table = print_comparison_table
    monkeypatch.setitem(sys.modules, "benchmarks", ModuleType("benchmarks"))
    monkeypatch.setitem(sys.modules, "benchmarks.run_all", fake_run_all)
    monkeypatch.setenv("HF_TOKEN", "token")

    _cli_bench._cmd_eval(["--dataset", "local"])

    out = capsys.readouterr().out
    assert "HF_TOKEN not set" not in out
    assert "comparison" in out
    assert "Results written" not in out


def test_tune_without_output_prints_metrics_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Tune should not write a profile overlay when no output path is supplied."""
    dataset = tmp_path / "labels.jsonl"
    dataset.write_text(
        '{"prompt":"p","response":"r","label":false}\n',
        encoding="utf-8",
    )
    fake_tuner = _TunerModule("director_ai.core.calibration.tuner")
    result = SimpleNamespace(
        threshold=0.4,
        w_logic=0.7,
        w_fact=0.3,
        balanced_accuracy=0.9,
        precision=0.8,
        recall=0.75,
        f1=0.77,
        samples=1,
    )

    def tune(samples: list[dict[str, object]]) -> SimpleNamespace:
        return result

    def format_confidence_report(tune_result: SimpleNamespace) -> str:
        return "confidence: ok"

    def format_profile_overlay(
        tune_result: SimpleNamespace,
        *,
        profile: str,
        base_profile: str,
    ) -> str:
        return "unused\n"

    fake_tuner.tune = tune
    fake_tuner.format_confidence_report = format_confidence_report
    fake_tuner.format_profile_overlay = format_profile_overlay
    monkeypatch.setitem(sys.modules, "director_ai.core.calibration.tuner", fake_tuner)

    _cli_bench._cmd_tune([str(dataset)])

    out = capsys.readouterr().out
    assert "Best threshold: 0.4" in out
    assert "Profile overlay written" not in out


def test_finetune_success_omits_absent_optional_result_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fine-tune output should omit optional result sections when absent."""
    train_file = tmp_path / "train.jsonl"
    train_file.write_text('{"premise":"p","hypothesis":"h","label":1}\n')

    class FakeConfig:
        output_dir = "./director-finetuned"

    def finetune_nli(
        path: str,
        *,
        eval_path: str | None,
        config: FakeConfig,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            output_dir="/models/domain",
            train_samples=7,
            mixed_general_samples=0,
            epochs_completed=1,
            final_loss=0.5,
            eval_samples=0,
            best_balanced_accuracy=0.0,
            regression_report=None,
            onnx_path="",
        )

    fake_module = _FinetuneModule("director_ai.core.training.finetune")
    fake_module.FinetuneConfig = FakeConfig
    fake_module.finetune_nli = finetune_nli
    monkeypatch.setitem(sys.modules, "director_ai.core.training.finetune", fake_module)

    _cli_bench._cmd_finetune([str(train_file)])

    out = capsys.readouterr().out
    assert "Model saved to:  /models/domain" in out
    assert "Mixed general" not in out
    assert "Eval samples" not in out
    assert "Regression" not in out
    assert "ONNX export" not in out


def test_validate_data_clean_report_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Validation should print only the summary for warning-free valid data."""
    data_file = tmp_path / "train.jsonl"
    data_file.write_text('{"premise":"p","hypothesis":"h","label":1}\n')

    report = SimpleNamespace(
        warnings=[],
        errors=[],
        summary=lambda: "1 valid, 0 invalid",
    )
    fake_module = _ValidatorModule("director_ai.core.training.finetune_validator")
    fake_module.validate_finetune_data = lambda path: report
    monkeypatch.setitem(
        sys.modules,
        "director_ai.core.training.finetune_validator",
        fake_module,
    )

    _cli_bench._cmd_validate_data([str(data_file)])

    out = capsys.readouterr().out
    assert "1 valid, 0 invalid" in out
    assert "Warnings:" not in out
    assert "Errors:" not in out


def test_tuner_functions_from_rejects_incomplete_modules() -> None:
    incomplete = ModuleType("director_ai.core.calibration.tuner")
    incomplete.tune = lambda records: SimpleNamespace()  # type: ignore[attr-defined]

    assert _cli_bench._tuner_functions_from(incomplete) is None
    assert _cli_bench._tuner_functions_from(object()) is None


def test_load_tuner_functions_raises_when_canonical_lacks_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bare = ModuleType("director_ai.core.calibration.tuner")
    monkeypatch.setitem(sys.modules, "director_ai.core.calibration.tuner", bare)
    monkeypatch.delitem(sys.modules, "director_ai.core.training.tuner", raising=False)

    with pytest.raises(RuntimeError, match="lacks required functions"):
        _cli_bench._load_tuner_functions()
