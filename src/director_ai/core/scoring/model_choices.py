# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Runtime scorer model choices

"""Benchmarked runtime scorer model choices.

This registry is intentionally separate from managed-training base models:
these entries are models a deployment can use for live scoring after the
Vertex benchmark gate has measured their accuracy and regression profile.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

DEFAULT_SCORER_MODEL_ALIAS = "balanced-default"
FINAL_VERTEX_REPORT_URI = (
    "gs://gotm-director-ai-training/managed-training/benchmarks/"
    "20260429T1409-e9859f8-full-e1-20260429-alias/"
    "model_benchmark_report.json"
)

_CUSTOM_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{1,300}$")


@dataclass(frozen=True)
class ScorerModelChoice:
    """One benchmarked runtime scorer option."""

    alias: str
    display_name: str
    model_id: str
    status: str
    recommendation: str
    template: str
    parameter_count_b: float
    max_length: int
    general_balanced_accuracy: float
    general_f1: float
    regression_pp: float
    artifact_uri: str = ""
    revision: str = ""
    notes: str = ""

    @property
    def is_default(self) -> bool:
        return self.alias == DEFAULT_SCORER_MODEL_ALIAS

    @property
    def is_domain_only(self) -> bool:
        return self.status == "domain_only"

    @property
    def requires_managed_artifact(self) -> bool:
        return self.artifact_uri.startswith("gs://")

    @property
    def runtime_model(self) -> str:
        return self.artifact_uri or self.model_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "display_name": self.display_name,
            "model_id": self.model_id,
            "runtime_model": self.runtime_model,
            "artifact_uri": self.artifact_uri,
            "revision": self.revision,
            "status": self.status,
            "recommendation": self.recommendation,
            "template": self.template,
            "parameter_count_b": self.parameter_count_b,
            "max_length": self.max_length,
            "general_balanced_accuracy": self.general_balanced_accuracy,
            "general_f1": self.general_f1,
            "regression_pp": self.regression_pp,
            "is_default": self.is_default,
            "is_domain_only": self.is_domain_only,
            "requires_managed_artifact": self.requires_managed_artifact,
            "benchmark_report_uri": FINAL_VERTEX_REPORT_URI,
            "notes": self.notes,
        }


_CHOICES: tuple[ScorerModelChoice, ...] = (
    ScorerModelChoice(
        alias=DEFAULT_SCORER_MODEL_ALIAS,
        display_name="Balanced default",
        model_id="yaxili96/FactCG-DeBERTa-v3-Large",
        artifact_uri=(
            "gs://gotm-director-ai-training/managed-training/sweeps/"
            "managed-full-e1-20260428/"
            "naturalfull-factcg-deberta-v3-large-e1-b1"
        ),
        status="stable",
        recommendation="deploy",
        template="factcg",
        parameter_count_b=0.4,
        max_length=512,
        general_balanced_accuracy=0.752,
        general_f1=0.7915966386554621,
        regression_pp=-0.6,
        notes="Default selected by the final Vertex anti-regression gate.",
    ),
    ScorerModelChoice(
        alias="deberta-small",
        display_name="DeBERTa small",
        model_id="microsoft/deberta-v3-small",
        artifact_uri=(
            "gs://gotm-director-ai-training/managed-training/sweeps/"
            "managed-full-e1-20260428/naturalfull-deberta-v3-small-e1-b1"
        ),
        status="stable",
        recommendation="deploy",
        template="sequence-pair",
        parameter_count_b=0.14,
        max_length=512,
        general_balanced_accuracy=0.747,
        general_f1=0.784313725490196,
        regression_pp=-1.1,
        notes="Lower-cost option with near-default general accuracy.",
    ),
    ScorerModelChoice(
        alias="deberta-large-nli",
        display_name="DeBERTa large NLI",
        model_id="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        artifact_uri=(
            "gs://gotm-director-ai-training/managed-training/sweeps/"
            "managed-full-e1-20260428/"
            "naturalfull-deberta-v3-large-nli-e1-b1"
        ),
        status="stable",
        recommendation="deploy",
        template="sequence-pair",
        parameter_count_b=0.4,
        max_length=512,
        general_balanced_accuracy=0.740,
        general_f1=0.7822445561139029,
        regression_pp=-1.8,
        notes="Alternative large NLI baseline with deployable regression.",
    ),
    ScorerModelChoice(
        alias="distilroberta-fast",
        display_name="DistilRoBERTa fast",
        model_id="distilroberta-base",
        artifact_uri=(
            "gs://gotm-director-ai-training/managed-training/sweeps/"
            "managed-full-e1-20260428/naturalfull-distilroberta-base-e1-b1"
        ),
        status="domain_only",
        recommendation="deploy_domain_only",
        template="sequence-pair",
        parameter_count_b=0.08,
        max_length=512,
        general_balanced_accuracy=0.719,
        general_f1=0.7604433077578857,
        regression_pp=-3.9,
        notes="Fast option; requires explicit domain-only opt-in.",
    ),
    ScorerModelChoice(
        alias="roberta-mnli-legacy",
        display_name="RoBERTa MNLI legacy",
        model_id="roberta-large-mnli",
        artifact_uri=(
            "gs://gotm-director-ai-training/managed-training/sweeps/"
            "managed-full-e1-20260428/naturalfull-roberta-large-mnli-e1-b1"
        ),
        status="domain_only",
        recommendation="deploy_domain_only",
        template="sequence-pair",
        parameter_count_b=0.36,
        max_length=512,
        general_balanced_accuracy=0.706,
        general_f1=0.7529411764705883,
        regression_pp=-5.2,
        notes="Legacy compatibility option; requires explicit domain-only opt-in.",
    ),
)

_CHOICES_BY_ALIAS = {choice.alias: choice for choice in _CHOICES}
_CHOICES_BY_MODEL_ID = {choice.model_id: choice for choice in _CHOICES}


def list_scorer_model_choices(
    *,
    include_domain_only: bool = False,
) -> tuple[ScorerModelChoice, ...]:
    """Return benchmarked scorer choices visible to normal users."""
    if include_domain_only:
        return _CHOICES
    return tuple(choice for choice in _CHOICES if not choice.is_domain_only)


def scorer_model_choices_to_dict(
    *,
    include_domain_only: bool = False,
) -> list[dict[str, Any]]:
    return [
        choice.to_dict()
        for choice in list_scorer_model_choices(
            include_domain_only=include_domain_only,
        )
    ]


def resolve_scorer_model_choice(
    name_or_id: str,
    *,
    allow_domain_only: bool = False,
    allow_custom: bool = False,
) -> ScorerModelChoice:
    """Resolve an alias, known model ID, or explicitly allowed custom model."""
    name = name_or_id.strip()
    if not name:
        return _CHOICES_BY_ALIAS[DEFAULT_SCORER_MODEL_ALIAS]

    choice = _CHOICES_BY_ALIAS.get(name) or _CHOICES_BY_MODEL_ID.get(name)
    if choice is not None:
        if choice.is_domain_only and not allow_domain_only:
            raise ValueError(
                f"scorer_model {name!r} is domain-only; set "
                "allow_domain_only_scorer_model=True to use it",
            )
        return choice

    if not allow_custom:
        raise ValueError(
            f"unknown scorer_model {name!r}; choose a benchmarked alias or "
            "set allow_custom_scorer_model=True",
        )
    if not _CUSTOM_MODEL_RE.match(name):
        raise ValueError(f"invalid custom scorer_model {name!r}")
    return ScorerModelChoice(
        alias=name,
        display_name=name,
        model_id=name,
        status="custom",
        recommendation="benchmark_required",
        template="sequence-pair",
        parameter_count_b=0.0,
        max_length=512,
        general_balanced_accuracy=0.0,
        general_f1=0.0,
        regression_pp=0.0,
        notes="Custom model: run the Vertex benchmark gate before production use.",
    )
