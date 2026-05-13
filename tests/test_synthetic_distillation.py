# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Synthetic distillation provenance tests."""

from __future__ import annotations

import pytest

from director_ai.core.self_evolving import (
    FeedbackEvent,
    SyntheticDistillationBuilder,
    SyntheticDistillationManifest,
    SyntheticExample,
)


def _reviewed_events() -> list[FeedbackEvent]:
    return [
        FeedbackEvent(
            prompt=f"unsafe reviewed prompt {index}",
            response="",
            label="unsafe",
            metadata={
                "event_id": f"sevt-{index}",
                "reviewer_id": "reviewer-passport-1",
            },
            timestamp=float(index),
        )
        for index in range(4)
    ]


def test_synthetic_example_requires_source_events_and_reviewer():
    with pytest.raises(ValueError, match="source_event_ids"):
        SyntheticExample(
            prompt="synthetic prompt",
            response="synthetic response",
            label="unsafe",
            source_event_ids=(),
            reviewer_id="reviewer-passport-1",
            generator_id="deterministic-v1",
            seed=17,
        )

    with pytest.raises(ValueError, match="reviewer_id"):
        SyntheticExample(
            prompt="synthetic prompt",
            response="synthetic response",
            label="unsafe",
            source_event_ids=("sevt-1",),
            reviewer_id="",
            generator_id="deterministic-v1",
            seed=17,
        )


def test_synthetic_example_audit_payload_excludes_generated_text_by_default():
    example = SyntheticExample(
        prompt="synthetic prompt with private tenant detail",
        response="synthetic response with private tenant detail",
        label="unsafe",
        source_event_ids=("sevt-1",),
        reviewer_id="reviewer-passport-1",
        generator_id="deterministic-v1",
        seed=17,
    )

    payload = example.to_dict()

    assert "private tenant detail" not in str(payload)
    assert payload["synthetic"] is True
    assert payload["benchmark_evidence"] is False
    assert example.to_training_row()["prompt"].startswith("synthetic prompt")


def test_distillation_builder_is_seed_deterministic_and_provenance_preserving():
    builder = SyntheticDistillationBuilder(generator_id="deterministic-v1")

    first = builder.generate(
        _reviewed_events(),
        reviewer_id="reviewer-passport-1",
        seed=123,
        max_examples=3,
    )
    second = builder.generate(
        _reviewed_events(),
        reviewer_id="reviewer-passport-1",
        seed=123,
        max_examples=3,
    )

    assert first == second
    assert len(first) == 3
    assert first[0].source_event_ids == ("sevt-0",)
    assert {example.generator_id for example in first} == {"deterministic-v1"}


def test_distillation_manifest_rejects_duplicates_and_separates_dataset_counts():
    example = SyntheticExample(
        prompt="duplicate synthetic prompt",
        response="",
        label="unsafe",
        source_event_ids=("sevt-1",),
        reviewer_id="reviewer-passport-1",
        generator_id="deterministic-v1",
        seed=17,
    )

    with pytest.raises(ValueError, match="duplicate"):
        SyntheticDistillationManifest.from_examples(
            examples=[example, example],
            real_event_count=10,
            manifest_id="distill-1",
        )

    manifest = SyntheticDistillationManifest.from_examples(
        examples=[example],
        real_event_count=10,
        manifest_id="distill-1",
    )

    payload = manifest.to_dict()

    assert payload["synthetic_event_count"] == 1
    assert payload["real_event_count"] == 10
    assert payload["benchmark_evidence"] is False
    assert "duplicate synthetic prompt" not in str(payload)


def test_builder_rejects_unreviewed_source_events():
    builder = SyntheticDistillationBuilder(generator_id="deterministic-v1")
    events = [
        FeedbackEvent(
            prompt="unsafe reviewed prompt",
            response="",
            label="unsafe",
            metadata={"event_id": "sevt-1"},
        )
    ]

    with pytest.raises(ValueError, match="reviewer_id"):
        builder.generate(
            events,
            reviewer_id="reviewer-passport-1",
            seed=0,
            max_examples=1,
        )
