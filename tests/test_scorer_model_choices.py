# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Scorer model choice tests

import pytest

from director_ai.core.config import DirectorConfig
from director_ai.core.scoring import nli
from director_ai.core.scoring.model_choices import (
    DEFAULT_SCORER_MODEL_ALIAS,
    FINAL_VERTEX_REPORT_URI,
    list_scorer_model_choices,
    resolve_scorer_model_choice,
    scorer_model_choices_to_dict,
)


def test_scorer_choices_hide_domain_only_by_default():
    choices = list_scorer_model_choices()
    aliases = {choice.alias for choice in choices}

    assert DEFAULT_SCORER_MODEL_ALIAS in aliases
    assert "deberta-small" in aliases
    assert "distilroberta-fast" not in aliases


def test_scorer_choices_can_include_domain_only():
    choices = scorer_model_choices_to_dict(include_domain_only=True)
    aliases = {choice["alias"] for choice in choices}

    assert "distilroberta-fast" in aliases
    assert "roberta-mnli-legacy" in aliases


def test_domain_only_choice_requires_opt_in():
    with pytest.raises(ValueError, match="domain-only"):
        resolve_scorer_model_choice("distilroberta-fast")


def test_resolve_blank_and_model_id_paths():
    default = resolve_scorer_model_choice(" ")
    by_model = resolve_scorer_model_choice("microsoft/deberta-v3-small")

    assert default.alias == DEFAULT_SCORER_MODEL_ALIAS
    assert by_model.alias == "deberta-small"
    assert by_model.requires_managed_artifact is True
    assert by_model.runtime_model.startswith("gs://")
    assert by_model.to_dict()["benchmark_report_uri"] == FINAL_VERTEX_REPORT_URI


def test_resolve_domain_only_with_opt_in_and_custom_validation():
    domain_only = resolve_scorer_model_choice(
        "distilroberta-fast",
        allow_domain_only=True,
    )
    custom = resolve_scorer_model_choice(
        "customer/custom-model:v1",
        allow_custom=True,
    )

    assert domain_only.is_domain_only is True
    assert custom.status == "custom"
    assert custom.requires_managed_artifact is False
    assert custom.runtime_model == "customer/custom-model:v1"

    with pytest.raises(ValueError, match="invalid custom scorer_model"):
        resolve_scorer_model_choice("../bad", allow_custom=True)


def test_config_scorer_alias_wires_managed_artifact():
    cfg = DirectorConfig(scorer_model="deberta-small")

    assert cfg.nli_model.startswith("gs://")
    assert cfg.nli_model_artifact_uri == cfg.nli_model
    assert cfg.nli_model_revision == ""
    assert cfg.nli_max_length == 512


def test_config_domain_only_alias_requires_opt_in():
    with pytest.raises(ValueError, match="domain-only"):
        DirectorConfig(scorer_model="distilroberta-fast")

    cfg = DirectorConfig(
        scorer_model="distilroberta-fast",
        allow_domain_only_scorer_model=True,
    )
    assert cfg.nli_model.startswith("gs://")


def test_config_custom_model_requires_opt_in():
    with pytest.raises(ValueError, match="unknown scorer_model"):
        DirectorConfig(scorer_model="custom/model")

    cfg = DirectorConfig(
        scorer_model="custom/model",
        allow_custom_scorer_model=True,
    )
    assert cfg.nli_model == "custom/model"
    assert cfg.nli_model_artifact_uri == ""


def test_gcs_artifact_cache_helpers():
    bucket, prefix = nli._split_gs_uri("gs://bucket/path/to/model")

    assert bucket == "bucket"
    assert prefix == "path/to/model"
    assert nli._safe_cache_name("gs://bucket/path/to/model").startswith("bucket-path")
    assert nli._should_skip_artifact("checkpoint-1/model.safetensors")
    assert nli._should_skip_artifact("trainer_state.json")
    assert not nli._should_skip_artifact("config.json")
