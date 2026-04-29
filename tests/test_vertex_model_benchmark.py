# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vertex model benchmark entrypoint tests

"""Tests for the Vertex model-choice benchmark wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from director_ai.core.training.finetune_benchmark import (
    ModelBenchmarkResult,
    RegressionReport,
)
from director_ai.core.training.model_registry import TrainingModelProfile

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "training" / "vertex_model_benchmark.py"
)
_SPEC = importlib.util.spec_from_file_location("vertex_model_benchmark", _MODULE_PATH)
assert _SPEC is not None
vertex_model_benchmark = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(vertex_model_benchmark)

_parse_model_specs = vertex_model_benchmark._parse_model_specs
_should_skip_artifact = vertex_model_benchmark._should_skip_artifact


def test_parse_semicolon_model_specs() -> None:
    specs = _parse_model_specs(
        "factcg=gs://bucket/a;deberta-v3-large-nli=gs://bucket/b"
    )

    assert specs == {
        "factcg": "gs://bucket/a",
        "deberta-v3-large-nli": "gs://bucket/b",
    }


def test_parse_json_model_specs() -> None:
    specs = _parse_model_specs('{"roberta-large-mnli": "gs://bucket/model"}')

    assert specs == {"roberta-large-mnli": "gs://bucket/model"}


def test_parse_model_specs_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="required"):
        _parse_model_specs("")


def test_checkpoint_artifacts_are_skipped() -> None:
    assert _should_skip_artifact("checkpoint-29420/model.safetensors")
    assert _should_skip_artifact("checkpoint-29420/trainer_state.json")


def test_deployable_model_artifacts_are_kept() -> None:
    assert not _should_skip_artifact("model.safetensors")
    assert not _should_skip_artifact("tokenizer.json")
    assert not _should_skip_artifact("config.json")


def test_custom_model_result_uses_requested_alias() -> None:
    profile = TrainingModelProfile(
        alias="custom-experimental",
        model_id="customer/model-a",
        status="experimental",
        template="sequence-pair",
        label_count=2,
        baseline_accuracy=0.0,
        default_max_length=512,
        recommended_batch_size=8,
        recommended_learning_rate=1e-5,
        hardware_profile="benchmark-required",
    )

    result = ModelBenchmarkResult.from_report(
        requested_model="customer-model-a",
        profile=profile,
        model_path="/tmp/customer-model-a",
        report=RegressionReport(general_accuracy=0.7),
        elapsed_seconds=1.0,
    )

    assert result.alias == "customer-model-a"
    assert result.model_id == "customer/model-a"
