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
