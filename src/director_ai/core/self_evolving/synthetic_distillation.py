# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Synthetic Distillation Provenance

"""Provenance-preserving synthetic examples for reviewed distillation data."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any
from urllib.parse import urlparse

from director_ai.core.training.jobs import TrainingJobSpec

from .feedback import FeedbackEvent, FeedbackLabel

__all__ = [
    "SyntheticDistillationBuilder",
    "SyntheticDistillationManifest",
    "SyntheticDistillationPlan",
    "SyntheticExample",
]


@dataclass(frozen=True)
class SyntheticExample:
    """Synthetic training example linked to reviewed source events."""

    prompt: str
    response: str
    label: FeedbackLabel
    source_event_ids: Sequence[str]
    reviewer_id: str
    generator_id: str
    seed: int

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt is required")
        if not self.reviewer_id.strip():
            raise ValueError("reviewer_id is required")
        if not self.generator_id.strip():
            raise ValueError("generator_id is required")
        refs = tuple(str(ref) for ref in self.source_event_ids if str(ref).strip())
        if not refs:
            raise ValueError("source_event_ids must not be empty")
        object.__setattr__(self, "source_event_ids", refs)

    @property
    def dedupe_key(self) -> str:
        """Normalised key used to prevent duplicate synthetic rows."""
        return " ".join(self.prompt.casefold().split())

    def to_dict(self, *, include_generated_text: bool = False) -> dict[str, Any]:
        """Serialise tenant-safe audit metadata by default."""
        payload: dict[str, Any] = {
            "label": self.label,
            "source_event_ids": list(self.source_event_ids),
            "reviewer_id": self.reviewer_id,
            "generator_id": self.generator_id,
            "seed": self.seed,
            "synthetic": True,
            "benchmark_evidence": False,
            "dedupe_digest": sha256(self.dedupe_key.encode("utf-8")).hexdigest(),
        }
        if include_generated_text:
            payload["prompt"] = self.prompt
            payload["response"] = self.response
        return payload

    def to_training_row(self) -> dict[str, Any]:
        """Return the row shape consumed by training jobs."""
        return {
            "prompt": self.prompt,
            "response": self.response,
            "label": self.label,
            "source_event_ids": list(self.source_event_ids),
            "reviewer_id": self.reviewer_id,
            "synthetic": True,
            "benchmark_evidence": False,
        }


@dataclass(frozen=True)
class SyntheticDistillationManifest:
    """Tenant-safe manifest for a mixed real/synthetic distillation set."""

    manifest_id: str
    synthetic_event_count: int
    real_event_count: int
    label_counts: Mapping[str, int]
    source_event_ids: Sequence[str]
    generator_ids: Sequence[str]
    benchmark_evidence: bool = False

    def __post_init__(self) -> None:
        if not self.manifest_id.strip():
            raise ValueError("manifest_id is required")
        if self.synthetic_event_count <= 0:
            raise ValueError("synthetic_event_count must be positive")
        if self.real_event_count < 0:
            raise ValueError("real_event_count must be non-negative")
        if self.benchmark_evidence:
            raise ValueError("synthetic manifests cannot be benchmark evidence")
        object.__setattr__(self, "label_counts", dict(self.label_counts))
        object.__setattr__(
            self, "source_event_ids", tuple(map(str, self.source_event_ids))
        )
        object.__setattr__(self, "generator_ids", tuple(map(str, self.generator_ids)))

    @classmethod
    def from_examples(
        cls,
        *,
        examples: Sequence[SyntheticExample],
        real_event_count: int,
        manifest_id: str,
    ) -> SyntheticDistillationManifest:
        """Build a manifest after duplicate and provenance checks."""
        if not examples:
            raise ValueError("examples must not be empty")
        seen: set[str] = set()
        labels: Counter[str] = Counter()
        source_ids: set[str] = set()
        generator_ids: set[str] = set()
        for example in examples:
            if example.dedupe_key in seen:
                raise ValueError(f"duplicate synthetic example {example.dedupe_key!r}")
            seen.add(example.dedupe_key)
            labels[example.label] += 1
            source_ids.update(example.source_event_ids)
            generator_ids.add(example.generator_id)
        return cls(
            manifest_id=manifest_id,
            synthetic_event_count=len(examples),
            real_event_count=real_event_count,
            label_counts=dict(sorted(labels.items())),
            source_event_ids=tuple(sorted(source_ids)),
            generator_ids=tuple(sorted(generator_ids)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise manifest metadata without generated prompt text."""
        return {
            "manifest_id": self.manifest_id,
            "synthetic_event_count": self.synthetic_event_count,
            "real_event_count": self.real_event_count,
            "label_counts": dict(self.label_counts),
            "source_event_ids": list(self.source_event_ids),
            "generator_ids": list(self.generator_ids),
            "benchmark_evidence": self.benchmark_evidence,
        }


@dataclass(frozen=True)
class SyntheticDistillationPlan:
    """Reviewed synthetic rows, tenant-safe manifest, and managed job request."""

    examples: Sequence[SyntheticExample]
    manifest: SyntheticDistillationManifest
    training_job: TrainingJobSpec

    def __post_init__(self) -> None:
        examples = tuple(self.examples)
        if not examples:
            raise ValueError("examples must not be empty")
        if len(examples) != self.manifest.synthetic_event_count:
            raise ValueError("example count must match manifest")
        object.__setattr__(self, "examples", examples)

    def training_rows(self) -> tuple[dict[str, Any], ...]:
        """Return synthetic rows for the controlled dataset writer."""
        return tuple(example.to_training_row() for example in self.examples)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the plan without generated prompt or response text."""
        return {
            "manifest": self.manifest.to_dict(),
            "examples": [example.to_dict() for example in self.examples],
            "training_job": {
                "submitted": False,
                "spec": self.training_job.to_redacted_dict(),
                "dataset_hash": self.training_job.dataset_hash,
                "config_hash": self.training_job.config_hash,
            },
        }


class SyntheticDistillationBuilder:
    """Deterministically derive reviewed synthetic examples from feedback."""

    def __init__(self, *, generator_id: str) -> None:
        if not generator_id.strip():
            raise ValueError("generator_id is required")
        self._generator_id = generator_id

    def generate(
        self,
        events: Iterable[FeedbackEvent],
        *,
        reviewer_id: str,
        seed: int,
        max_examples: int,
    ) -> tuple[SyntheticExample, ...]:
        """Generate deterministic synthetic examples with source provenance."""
        if not reviewer_id.strip():
            raise ValueError("reviewer_id is required")
        if max_examples <= 0:
            raise ValueError("max_examples must be positive")
        reviewed = tuple(
            _reviewed_event(event, index) for index, event in enumerate(events)
        )
        rng = random.Random(seed)
        examples: list[SyntheticExample] = []
        seen: set[str] = set()
        for event in reviewed:
            if len(examples) >= max_examples:
                break
            event_id = event.metadata["event_id"]
            prompt = _synthetic_prompt(event.prompt, rng)
            candidate = SyntheticExample(
                prompt=prompt,
                response=event.response,
                label=event.label,
                source_event_ids=(event_id,),
                reviewer_id=reviewer_id,
                generator_id=self._generator_id,
                seed=seed,
            )
            if candidate.dedupe_key in seen:
                continue
            seen.add(candidate.dedupe_key)
            examples.append(candidate)
        return tuple(examples)

    def build_training_plan(
        self,
        events: Iterable[FeedbackEvent],
        *,
        reviewer_id: str,
        seed: int,
        max_examples: int,
        real_event_count: int,
        manifest_id: str,
        dataset_uri: str,
        output_uri: str,
        base_model_ref: str,
        schedule_id: str,
    ) -> SyntheticDistillationPlan:
        """Build a synthetic distillation plan without submitting a job."""
        _reject_embedded_credentials("dataset_uri", dataset_uri)
        _reject_embedded_credentials("output_uri", output_uri)
        if not dataset_uri.strip():
            raise ValueError("dataset_uri is required")
        if not output_uri.strip():
            raise ValueError("output_uri is required")
        if not base_model_ref.strip():
            raise ValueError("base_model_ref is required")
        if not schedule_id.strip():
            raise ValueError("schedule_id is required")
        examples = self.generate(
            events,
            reviewer_id=reviewer_id,
            seed=seed,
            max_examples=max_examples,
        )
        manifest = SyntheticDistillationManifest.from_examples(
            examples=examples,
            real_event_count=real_event_count,
            manifest_id=manifest_id,
        )
        training_job = TrainingJobSpec(
            display_name=f"director-ai-{manifest.manifest_id}",
            dataset_uri=dataset_uri,
            output_uri=output_uri,
            base_model=base_model_ref,
            labels={
                "synthetic": "true",
                "benchmark_evidence": "false",
                "manifest_id": manifest.manifest_id,
                "schedule_id": schedule_id,
            },
            env={
                "DIRECTOR_DISTILLATION_MANIFEST_ID": manifest.manifest_id,
                "DIRECTOR_DISTILLATION_SYNTHETIC_ROWS": str(
                    manifest.synthetic_event_count
                ),
                "DIRECTOR_DISTILLATION_REAL_ROWS": str(manifest.real_event_count),
            },
            args=[
                "--synthetic-distillation-manifest",
                manifest.manifest_id,
                "--schedule-id",
                schedule_id,
            ],
        )
        training_job.validate("local")
        return SyntheticDistillationPlan(
            examples=examples,
            manifest=manifest,
            training_job=training_job,
        )


def _reviewed_event(event: FeedbackEvent, index: int) -> FeedbackEvent:
    if not event.metadata.get("event_id", "").strip():
        raise ValueError(f"feedback event {index} is missing event_id")
    if not event.metadata.get("reviewer_id", "").strip():
        raise ValueError(f"feedback event {index} is missing reviewer_id")
    return event


def _synthetic_prompt(prompt: str, rng: random.Random) -> str:
    tokens = prompt.split()
    if len(tokens) >= 2:
        left = tokens[:]
        rng.shuffle(left)
        return "synthetic reviewed variant: " + " ".join(left)
    return f"synthetic reviewed variant: {prompt}"


def _reject_embedded_credentials(name: str, uri: str) -> None:
    parsed = urlparse(uri)
    if parsed.netloc and "@" in parsed.netloc:
        raise ValueError(f"{name} must not contain embedded credentials")
