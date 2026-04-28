# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Fine-tune Model Registry

"""Controlled base-model registry for managed fine-tuning."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

DEFAULT_FINE_TUNE_MODEL_ALIAS = "factcg-deberta-v3-large"
_CUSTOM_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{1,200}$")


@dataclass(frozen=True)
class TrainingModelProfile:
    """Product-safe description of a model that can be fine-tuned."""

    alias: str
    model_id: str
    status: str
    template: str
    label_count: int
    baseline_accuracy: float
    default_max_length: int
    recommended_batch_size: int
    recommended_learning_rate: float
    hardware_profile: str
    notes: str = ""

    @property
    def is_experimental(self) -> bool:
        return self.status != "stable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "model_id": self.model_id,
            "status": self.status,
            "template": self.template,
            "label_count": self.label_count,
            "baseline_accuracy": self.baseline_accuracy,
            "default_max_length": self.default_max_length,
            "recommended_batch_size": self.recommended_batch_size,
            "recommended_learning_rate": self.recommended_learning_rate,
            "hardware_profile": self.hardware_profile,
            "notes": self.notes,
        }


_MODEL_PROFILES: tuple[TrainingModelProfile, ...] = (
    TrainingModelProfile(
        alias=DEFAULT_FINE_TUNE_MODEL_ALIAS,
        model_id="yaxili96/FactCG-DeBERTa-v3-Large",
        status="stable",
        template="factcg",
        label_count=2,
        baseline_accuracy=0.758,
        default_max_length=512,
        recommended_batch_size=16,
        recommended_learning_rate=2e-5,
        hardware_profile="single-l4-or-better",
        notes="Current production baseline for factual consistency fine-tuning.",
    ),
    TrainingModelProfile(
        alias="deberta-v3-large-nli",
        model_id="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        status="experimental",
        template="sequence-pair",
        label_count=3,
        baseline_accuracy=0.0,
        default_max_length=512,
        recommended_batch_size=8,
        recommended_learning_rate=1e-5,
        hardware_profile="single-l4-or-a100",
        notes="Candidate general NLI baseline; requires same-data benchmark before activation.",
    ),
    TrainingModelProfile(
        alias="roberta-large-mnli",
        model_id="roberta-large-mnli",
        status="experimental",
        template="sequence-pair",
        label_count=3,
        baseline_accuracy=0.0,
        default_max_length=512,
        recommended_batch_size=8,
        recommended_learning_rate=1e-5,
        hardware_profile="single-l4-or-a100",
        notes="Legacy MNLI candidate for comparison sweeps; not production default.",
    ),
)

_BY_ALIAS = {profile.alias: profile for profile in _MODEL_PROFILES}
_BY_MODEL_ID = {profile.model_id.lower(): profile for profile in _MODEL_PROFILES}


def list_finetune_model_profiles(
    *,
    include_experimental: bool = False,
) -> list[TrainingModelProfile]:
    """Return allowed model profiles for product model selection."""

    return [
        profile
        for profile in _MODEL_PROFILES
        if include_experimental or not profile.is_experimental
    ]


def resolve_finetune_model(
    name_or_id: str,
    *,
    allow_experimental: bool = False,
) -> TrainingModelProfile:
    """Resolve a registry alias or model id to a training profile.

    Unknown model ids are allowed only when the caller explicitly opts into the
    experimental path. They still carry ``status=experimental`` so downstream
    benchmark gates can refuse default activation until measured.
    """

    if not name_or_id:
        raise ValueError("base_model is required")

    key = name_or_id.strip()
    profile = _BY_ALIAS.get(key) or _BY_MODEL_ID.get(key.lower())
    if profile is not None:
        if profile.is_experimental and not allow_experimental:
            raise ValueError(
                f"model {key!r} is experimental; pass allow_experimental_model",
            )
        return profile

    if not allow_experimental:
        raise ValueError(
            f"model {key!r} is not in the stable fine-tune registry",
        )
    if not _CUSTOM_MODEL_RE.fullmatch(key):
        raise ValueError(
            "experimental model ids may contain only model path characters"
        )

    return TrainingModelProfile(
        alias="custom-experimental",
        model_id=key,
        status="experimental",
        template="sequence-pair",
        label_count=2,
        baseline_accuracy=0.0,
        default_max_length=512,
        recommended_batch_size=8,
        recommended_learning_rate=1e-5,
        hardware_profile="benchmark-required",
        notes="Caller-supplied model id; must pass same-data benchmark before activation.",
    )


def finetune_model_registry_to_dict(
    *,
    include_experimental: bool = False,
) -> list[dict[str, Any]]:
    """Return JSON-safe registry entries."""

    return [
        profile.to_dict()
        for profile in list_finetune_model_profiles(
            include_experimental=include_experimental,
        )
    ]
