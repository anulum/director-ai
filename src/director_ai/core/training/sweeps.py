# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Managed Training Sweeps

"""Scenario planning for managed fine-tuning sweeps."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from .jobs import TrainingHardware, TrainingJobSpec
from .model_registry import resolve_finetune_model

_SLUG_RE = re.compile(r"[^a-z0-9-]+")


@dataclass(frozen=True)
class TrainingDatasetSplit:
    """Named train/eval dataset pair used by a sweep."""

    name: str
    train_uri: str
    eval_uri: str | None = None

    def validate(self) -> None:
        if not self.name:
            raise ValueError("dataset split name is required")
        if not self.train_uri:
            raise ValueError("dataset split train_uri is required")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "train_uri": self.train_uri,
            "eval_uri": self.eval_uri,
        }


@dataclass(frozen=True)
class TrainingScenario:
    """One concrete managed training scenario."""

    scenario_id: str
    dataset: TrainingDatasetSplit
    base_model: str
    resolved_model_id: str
    model_alias: str
    epochs: int
    batch_size: int
    learning_rate: float
    output_uri: str
    display_name: str
    labels: dict[str, str] = field(default_factory=dict)

    def to_spec(
        self,
        *,
        project: str,
        region: str,
        container_image_uri: str,
        hardware: TrainingHardware,
        timeout_minutes: int,
        caller: str,
        allow_experimental_model: bool,
        service_account: str | None = None,
        network: str | None = None,
    ) -> TrainingJobSpec:
        """Convert this scenario into a managed fine-tune job spec."""

        return TrainingJobSpec(
            display_name=self.display_name,
            caller=caller,
            dataset_uri=self.dataset.train_uri,
            output_uri=self.output_uri,
            eval_uri=self.dataset.eval_uri,
            project=project,
            region=region,
            base_model=self.base_model,
            allow_experimental_model=allow_experimental_model,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            timeout_minutes=timeout_minutes,
            container_image_uri=container_image_uri,
            service_account=service_account,
            network=network,
            hardware=hardware,
            labels=dict(self.labels),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "dataset": self.dataset.to_dict(),
            "base_model": self.base_model,
            "resolved_model_id": self.resolved_model_id,
            "model_alias": self.model_alias,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "output_uri": self.output_uri,
            "display_name": self.display_name,
            "labels": dict(self.labels),
        }


@dataclass(frozen=True)
class TrainingSweepPlan:
    """A reproducible managed fine-tuning scenario matrix."""

    sweep_id: str
    scenarios: list[TrainingScenario]
    generated_at: float = field(default_factory=time.time)
    selection_policy: str = (
        "same-eval-set comparison; promote only after anti-regression benchmark"
    )

    def to_specs(
        self,
        *,
        project: str,
        region: str,
        container_image_uri: str,
        hardware: TrainingHardware,
        timeout_minutes: int,
        caller: str = "internal",
        allow_experimental_model: bool = False,
        service_account: str | None = None,
        network: str | None = None,
    ) -> list[TrainingJobSpec]:
        """Convert every scenario in the plan into a managed job spec."""

        return [
            scenario.to_spec(
                project=project,
                region=region,
                container_image_uri=container_image_uri,
                hardware=hardware,
                timeout_minutes=timeout_minutes,
                caller=caller,
                allow_experimental_model=allow_experimental_model,
                service_account=service_account,
                network=network,
            )
            for scenario in self.scenarios
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sweep_id": self.sweep_id,
            "generated_at": self.generated_at,
            "selection_policy": self.selection_policy,
            "scenario_count": len(self.scenarios),
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }


def build_training_sweep_plan(
    *,
    sweep_id: str,
    datasets: list[TrainingDatasetSplit],
    base_models: list[str],
    epochs: list[int],
    output_prefix: str,
    batch_sizes: list[int] | None = None,
    learning_rate: float | None = None,
    allow_experimental_model: bool = False,
    display_prefix: str = "director-ai-managed-sweep",
) -> TrainingSweepPlan:
    """Build a managed fine-tuning matrix without submitting jobs."""

    if not sweep_id:
        raise ValueError("sweep_id is required")
    if not datasets:
        raise ValueError("at least one dataset split is required")
    if not base_models:
        raise ValueError("at least one base model is required")
    if not epochs:
        raise ValueError("at least one epoch value is required")
    if not output_prefix:
        raise ValueError("output_prefix is required")

    for dataset in datasets:
        dataset.validate()
    for value in epochs:
        if value < 1:
            raise ValueError("epochs values must be >= 1")
    if batch_sizes is not None:
        for value in batch_sizes:
            if value < 1:
                raise ValueError("batch_sizes values must be >= 1")

    scenarios: list[TrainingScenario] = []
    for dataset in datasets:
        for base_model in base_models:
            profile = resolve_finetune_model(
                base_model,
                allow_experimental=allow_experimental_model,
            )
            model_batch_sizes = batch_sizes or [profile.recommended_batch_size]
            lr = (
                learning_rate
                if learning_rate is not None
                else (profile.recommended_learning_rate)
            )
            for n_epochs in epochs:
                for batch_size in model_batch_sizes:
                    scenario_id = _scenario_id(
                        dataset_name=dataset.name,
                        model_alias=profile.alias,
                        epochs=n_epochs,
                        batch_size=batch_size,
                    )
                    scenarios.append(
                        TrainingScenario(
                            scenario_id=scenario_id,
                            dataset=dataset,
                            base_model=base_model,
                            resolved_model_id=profile.model_id,
                            model_alias=profile.alias,
                            epochs=n_epochs,
                            batch_size=batch_size,
                            learning_rate=lr,
                            output_uri=f"{output_prefix.rstrip('/')}/{scenario_id}",
                            display_name=f"{display_prefix}-{scenario_id}",
                            labels={
                                "sweep": sweep_id,
                                "scenario": scenario_id,
                                "train_set": dataset.name,
                                "model": profile.alias,
                                "epochs": str(n_epochs),
                                "batch_size": str(batch_size),
                            },
                        )
                    )

    return TrainingSweepPlan(sweep_id=sweep_id, scenarios=scenarios)


def _scenario_id(
    *,
    dataset_name: str,
    model_alias: str,
    epochs: int,
    batch_size: int,
) -> str:
    parts = [
        _slug(dataset_name),
        _slug(model_alias),
        f"e{epochs}",
        f"b{batch_size}",
    ]
    return "-".join(part for part in parts if part)


def _slug(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.lower().replace("_", "-")).strip("-")
    return slug[:48] or "unnamed"
