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
    SyntheticDistillationPlan,
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


def test_builder_creates_training_plan_without_submitting_job():
    builder = SyntheticDistillationBuilder(generator_id="deterministic-v1")

    plan = builder.build_training_plan(
        _reviewed_events(),
        reviewer_id="reviewer-passport-1",
        seed=123,
        max_examples=3,
        real_event_count=4,
        manifest_id="distill-20260513-a",
        dataset_uri="env://DIRECTOR_SYNTHETIC_DISTILLATION_DATASET",
        output_uri="env://DIRECTOR_SYNTHETIC_DISTILLATION_OUTPUT",
        base_model_ref="factcg-deberta-v3-large",
        schedule_id="nightly-reviewed-feedback",
    )

    assert isinstance(plan, SyntheticDistillationPlan)
    assert len(plan.examples) == 3
    assert len(plan.training_rows()) == 3
    assert plan.manifest.synthetic_event_count == 3
    assert plan.training_job.display_name == "director-ai-distill-20260513-a"
    assert plan.training_job.dataset_uri == (
        "env://DIRECTOR_SYNTHETIC_DISTILLATION_DATASET"
    )
    assert plan.training_job.output_uri == (
        "env://DIRECTOR_SYNTHETIC_DISTILLATION_OUTPUT"
    )
    assert plan.training_job.base_model == "factcg-deberta-v3-large"
    assert plan.training_job.labels["synthetic"] == "true"
    assert plan.training_job.labels["benchmark_evidence"] == "false"
    assert plan.training_job.labels["manifest_id"] == "distill-20260513-a"
    assert plan.training_job.labels["schedule_id"] == "nightly-reviewed-feedback"
    assert plan.training_job.env == {
        "DIRECTOR_DISTILLATION_MANIFEST_ID": "distill-20260513-a",
        "DIRECTOR_DISTILLATION_SYNTHETIC_ROWS": "3",
        "DIRECTOR_DISTILLATION_REAL_ROWS": "4",
    }
    audit_payload = plan.to_dict()
    assert "unsafe reviewed prompt" not in str(audit_payload)
    assert "synthetic reviewed variant" not in str(audit_payload)
    assert audit_payload["training_job"]["submitted"] is False


def test_training_plan_rejects_dataset_uri_with_embedded_credentials():
    builder = SyntheticDistillationBuilder(generator_id="deterministic-v1")

    with pytest.raises(ValueError, match="embedded credentials"):
        builder.build_training_plan(
            _reviewed_events(),
            reviewer_id="reviewer-passport-1",
            seed=123,
            max_examples=3,
            real_event_count=4,
            manifest_id="distill-20260513-a",
            dataset_uri="https://user@example.test/distill.jsonl",
            output_uri="env://DIRECTOR_SYNTHETIC_DISTILLATION_OUTPUT",
            base_model_ref="factcg-deberta-v3-large",
            schedule_id="nightly-reviewed-feedback",
        )


def test_synthetic_example_validates_prompt_generator_and_includes_text_on_request():
    with pytest.raises(ValueError, match="prompt"):
        SyntheticExample(
            prompt="",
            response="synthetic response",
            label="unsafe",
            source_event_ids=("sevt-1",),
            reviewer_id="reviewer-passport-1",
            generator_id="deterministic-v1",
            seed=17,
        )
    with pytest.raises(ValueError, match="generator_id"):
        SyntheticExample(
            prompt="synthetic prompt",
            response="synthetic response",
            label="unsafe",
            source_event_ids=("sevt-1",),
            reviewer_id="reviewer-passport-1",
            generator_id="",
            seed=17,
        )

    example = SyntheticExample(
        prompt="One   Mixed CASE prompt",
        response="synthetic response",
        label="unsafe",
        source_event_ids=("sevt-1",),
        reviewer_id="reviewer-passport-1",
        generator_id="deterministic-v1",
        seed=17,
    )

    assert example.dedupe_key == "one mixed case prompt"
    assert example.to_dict(include_generated_text=True)["prompt"].startswith("One")


def test_distillation_manifest_and_plan_validate_counts_and_flags():
    example = SyntheticExample(
        prompt="unique synthetic prompt",
        response="",
        label="unsafe",
        source_event_ids=("sevt-1",),
        reviewer_id="reviewer-passport-1",
        generator_id="deterministic-v1",
        seed=17,
    )
    manifest = SyntheticDistillationManifest.from_examples(
        examples=[example],
        real_event_count=0,
        manifest_id="distill-1",
    )

    with pytest.raises(ValueError, match="examples"):
        SyntheticDistillationManifest.from_examples(
            examples=[],
            real_event_count=0,
            manifest_id="distill-empty",
        )
    with pytest.raises(ValueError, match="manifest_id"):
        SyntheticDistillationManifest("", 1, 0, {}, ("sevt-1",), ("gen",))
    with pytest.raises(ValueError, match="synthetic_event_count"):
        SyntheticDistillationManifest("distill-1", 0, 0, {}, ("sevt-1",), ("gen",))
    with pytest.raises(ValueError, match="real_event_count"):
        SyntheticDistillationManifest("distill-1", 1, -1, {}, ("sevt-1",), ("gen",))
    with pytest.raises(ValueError, match="benchmark evidence"):
        SyntheticDistillationManifest(
            "distill-1",
            1,
            0,
            {},
            ("sevt-1",),
            ("gen",),
            benchmark_evidence=True,
        )
    with pytest.raises(ValueError, match="examples"):
        SyntheticDistillationPlan((), manifest, object())  # type: ignore[arg-type]


def test_builder_validates_generation_and_training_plan_arguments():
    with pytest.raises(ValueError, match="generator_id"):
        SyntheticDistillationBuilder(generator_id="")

    builder = SyntheticDistillationBuilder(generator_id="deterministic-v1")
    with pytest.raises(ValueError, match="reviewer_id"):
        builder.generate(
            _reviewed_events(),
            reviewer_id="",
            seed=1,
            max_examples=1,
        )
    with pytest.raises(ValueError, match="max_examples"):
        builder.generate(
            _reviewed_events(),
            reviewer_id="reviewer-passport-1",
            seed=1,
            max_examples=0,
        )

    invalid_plan_args = [
        {"dataset_uri": ""},
        {"output_uri": ""},
        {"base_model_ref": ""},
        {"schedule_id": ""},
        {"output_uri": "https://user@example.test/out"},
    ]
    for override in invalid_plan_args:
        kwargs = {
            "reviewer_id": "reviewer-passport-1",
            "seed": 123,
            "max_examples": 3,
            "real_event_count": 4,
            "manifest_id": "distill-20260513-a",
            "dataset_uri": "env://DIRECTOR_SYNTHETIC_DISTILLATION_DATASET",
            "output_uri": "env://DIRECTOR_SYNTHETIC_DISTILLATION_OUTPUT",
            "base_model_ref": "factcg-deberta-v3-large",
            "schedule_id": "nightly-reviewed-feedback",
            **override,
        }
        with pytest.raises(ValueError):
            builder.build_training_plan(_reviewed_events(), **kwargs)


def test_builder_handles_single_token_prompts_and_deduplicates_generated_rows():
    builder = SyntheticDistillationBuilder(generator_id="deterministic-v1")
    events = [
        FeedbackEvent(
            prompt="single",
            response="",
            label="unsafe",
            metadata={"event_id": "sevt-1", "reviewer_id": "reviewer-passport-1"},
        ),
        FeedbackEvent(
            prompt="single",
            response="",
            label="unsafe",
            metadata={"event_id": "sevt-2", "reviewer_id": "reviewer-passport-1"},
        ),
    ]

    examples = builder.generate(
        events,
        reviewer_id="reviewer-passport-1",
        seed=1,
        max_examples=2,
    )

    assert len(examples) == 1
    assert examples[0].prompt == "synthetic reviewed variant: single"
