# SPDX-License-Identifier: Apache-2.0
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
    """Check profile selection and serialization without loading model weights."""

    def test_default_model_is_stable(self) -> None:
        """Resolve the maintained default alias to the expected stable profile."""
        profile = resolve_finetune_model(DEFAULT_FINE_TUNE_MODEL_ALIAS)
        assert profile.status == "stable"
        assert profile.model_id == "yaxili96/FactCG-DeBERTa-v3-Large"

    def test_stable_list_excludes_experimental(self) -> None:
        """Return only stable profiles when experimental selection is disabled."""
        profiles = list_finetune_model_profiles()
        assert profiles
        assert all(profile.status == "stable" for profile in profiles)

    def test_include_experimental_adds_candidates(self) -> None:
        """Include the declared experimental candidates when explicitly requested."""
        stable_count = len(list_finetune_model_profiles())
        profiles = list_finetune_model_profiles(include_experimental=True)
        all_count = len(profiles)
        assert all_count > stable_count
        aliases = {profile.alias for profile in profiles}
        assert {"deberta-v3-small", "distilroberta-base"} <= aliases

    def test_experimental_requires_explicit_flag(self) -> None:
        """Refuse experimental profiles until the caller enables that selection."""
        with pytest.raises(ValueError, match="experimental"):
            resolve_finetune_model("roberta-large-mnli")

        profile = resolve_finetune_model(
            "roberta-large-mnli",
            allow_experimental=True,
        )
        assert profile.status == "experimental"

    def test_empty_model_name_is_rejected(self) -> None:
        """Reject an empty model identifier before resolving a profile."""
        with pytest.raises(ValueError, match="base_model is required"):
            resolve_finetune_model("")

    def test_unknown_model_requires_explicit_flag(self) -> None:
        """Require explicit experimental selection for a custom model identifier."""
        with pytest.raises(ValueError, match="stable fine-tune registry"):
            resolve_finetune_model("org/custom-model")

        profile = resolve_finetune_model(
            "org/custom-model",
            allow_experimental=True,
        )
        assert profile.alias == "custom-experimental"
        assert profile.model_id == "org/custom-model"

    def test_unknown_model_rejects_bad_id(self) -> None:
        """Reject invalid custom model-path characters despite experimental opt-in."""
        with pytest.raises(ValueError, match="model path characters"):
            resolve_finetune_model("bad model id", allow_experimental=True)

    def test_registry_serialises_for_api(self) -> None:
        """Serialize the registry with its declared alias and training metadata."""
        payload = finetune_model_registry_to_dict(include_experimental=True)
        assert payload[0]["alias"] == DEFAULT_FINE_TUNE_MODEL_ALIAS
        assert "recommended_batch_size" in payload[0]
