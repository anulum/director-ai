# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Fine-tune Model Registry Tests
"""Tests for user-selectable fine-tune model profiles."""

from __future__ import annotations

import pytest

from director_ai.core.training.model_registry import (
    DEFAULT_FINE_TUNE_MODEL_ALIAS,
    finetune_model_registry_to_dict,
    list_finetune_model_profiles,
    resolve_finetune_model,
)


class TestTrainingModelRegistry:
    def test_default_model_is_stable(self):
        profile = resolve_finetune_model(DEFAULT_FINE_TUNE_MODEL_ALIAS)
        assert profile.status == "stable"
        assert profile.model_id == "yaxili96/FactCG-DeBERTa-v3-Large"

    def test_stable_list_excludes_experimental(self):
        profiles = list_finetune_model_profiles()
        assert profiles
        assert all(profile.status == "stable" for profile in profiles)

    def test_include_experimental_adds_candidates(self):
        stable_count = len(list_finetune_model_profiles())
        profiles = list_finetune_model_profiles(include_experimental=True)
        all_count = len(profiles)
        assert all_count > stable_count
        aliases = {profile.alias for profile in profiles}
        assert {"deberta-v3-small", "distilroberta-base"} <= aliases

    def test_experimental_requires_explicit_flag(self):
        with pytest.raises(ValueError, match="experimental"):
            resolve_finetune_model("roberta-large-mnli")

        profile = resolve_finetune_model(
            "roberta-large-mnli",
            allow_experimental=True,
        )
        assert profile.status == "experimental"

    def test_empty_model_name_is_rejected(self):
        with pytest.raises(ValueError, match="base_model is required"):
            resolve_finetune_model("")

    def test_unknown_model_requires_explicit_flag(self):
        with pytest.raises(ValueError, match="stable fine-tune registry"):
            resolve_finetune_model("org/custom-model")

        profile = resolve_finetune_model(
            "org/custom-model",
            allow_experimental=True,
        )
        assert profile.alias == "custom-experimental"
        assert profile.model_id == "org/custom-model"

    def test_unknown_model_rejects_bad_id(self):
        with pytest.raises(ValueError, match="model path characters"):
            resolve_finetune_model("bad model id", allow_experimental=True)

    def test_registry_serialises_for_api(self):
        payload = finetune_model_registry_to_dict(include_experimental=True)
        assert payload[0]["alias"] == DEFAULT_FINE_TUNE_MODEL_ALIAS
        assert "recommended_batch_size" in payload[0]
