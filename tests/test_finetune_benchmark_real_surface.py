# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - fine-tune benchmark real-surface tests
"""Real-surface tests for fine-tune benchmark report and data-path edges."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path

import pytest

import director_ai.core.training.finetune_benchmark as benchmark_mod
from director_ai.core.training.finetune import TrainingRow
from director_ai.core.training.finetune_benchmark import (
    ModelBenchmarkReport,
    ModelBenchmarkResult,
    benchmark_finetuned_model,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write benchmark rows to a JSONL file."""
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _deployable_result(alias: str = "candidate") -> ModelBenchmarkResult:
    """Return one deployable benchmark result."""
    return ModelBenchmarkResult(
        requested_model=alias,
        alias=alias,
        model_id=f"{alias}-model",
        model_path=f"/models/{alias}",
        status="stable",
        template="nli_pair",
        label_count=2,
        baseline_accuracy=0.75,
        recommended_batch_size=16,
        general_accuracy=0.81,
        domain_accuracy=0.79,
        recommendation="deploy",
    )


def test_report_to_dict_preserves_operator_selected_winner() -> None:
    """Explicit winner metadata should not be replaced during serialization."""
    report = ModelBenchmarkReport(
        results=[_deployable_result("candidate")],
        general_path="general.jsonl",
        eval_path="domain.jsonl",
        best_model_alias="manual-choice",
        best_model_id="manual-model",
    )

    payload = report.to_dict()

    assert payload["best_model_alias"] == "manual-choice"
    assert payload["best_model_id"] == "manual-model"
    assert payload["results"][0]["requested_model"] == "candidate"


def test_empty_domain_and_general_data_do_not_invoke_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty benchmark files should produce a domain-only recommendation."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n{not-json}\n", encoding="utf-8")

    def fail_evaluate_model(
        model_path: str | Path,
        samples: list[TrainingRow],
        batch_size: int = 48,
    ) -> dict[str, float]:
        """Fail if empty benchmark data reaches model inference."""
        raise AssertionError(
            f"unexpected model evaluation for {model_path} "
            f"with {len(samples)} samples and batch_size={batch_size}",
        )

    monkeypatch.setattr(benchmark_mod, "_evaluate_model", fail_evaluate_model)

    report = benchmark_finetuned_model(
        "/models/candidate",
        general_path=empty,
        eval_path=empty,
    )

    assert report.recommendation == "deploy_domain_only"
    assert report.details == {"reason": "no general benchmark available"}


def test_default_package_benchmark_data_is_loaded_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default package benchmark lookup should evaluate bundled data."""
    fake_module = tmp_path / "core" / "training" / "finetune_benchmark.py"
    bundled_data = tmp_path / "core" / "data" / "aggrefact_benchmark_1k.jsonl"
    fake_module.parent.mkdir(parents=True)
    bundled_data.parent.mkdir(parents=True)
    fake_module.write_text("# placeholder\n", encoding="utf-8")
    _write_jsonl(
        bundled_data,
        [
            {
                "premise": "Verified evidence.",
                "hypothesis": "Supported claim.",
                "label": 1,
            }
        ],
    )
    original_path = Path
    module_file = str(benchmark_mod.__file__)
    calls: list[tuple[str, int, int]] = []

    def path_factory(value: str | PathLike[str]) -> Path:
        """Route the module file lookup to the temporary package layout."""
        if str(value) == module_file:
            return fake_module
        return original_path(value)

    def evaluate_model(
        model_path: str | Path,
        samples: list[TrainingRow],
        batch_size: int = 48,
    ) -> dict[str, float]:
        """Record the real benchmark samples passed to model evaluation."""
        calls.append((str(model_path), len(samples), batch_size))
        return {"balanced_accuracy": 0.80, "f1": 0.77}

    monkeypatch.setattr(benchmark_mod, "Path", path_factory)
    monkeypatch.setattr(benchmark_mod, "_evaluate_model", evaluate_model)

    report = benchmark_finetuned_model("/models/candidate", batch_size=7)

    assert calls == [("/models/candidate", 1, 7)]
    assert report.recommendation == "deploy"
    assert report.general_accuracy == 0.80
    assert report.general_f1 == 0.77
    assert report.details["general_samples"] == 1
