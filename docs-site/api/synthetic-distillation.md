# Synthetic Distillation

Synthetic distillation creates reviewed training examples from reviewed source
events. Synthetic rows are useful for hard-negative coverage, but they are not
real benchmark evidence and must not be reported as such.

## Provenance Contract

`SyntheticExample` requires:

- at least one reviewed source event ID
- reviewer identity
- generator ID
- deterministic seed
- an explicit `synthetic=True` training marker
- `benchmark_evidence=False`

The default audit payload excludes generated prompt and response text. Use
`to_training_row()` only inside the controlled training dataset builder.

```python
from director_ai.core.self_evolving import SyntheticDistillationBuilder

builder = SyntheticDistillationBuilder(generator_id="deterministic-v1")
examples = builder.generate(
    reviewed_events,
    reviewer_id="reviewer-passport-1",
    seed=123,
    max_examples=32,
)
```

`SyntheticDistillationManifest` deduplicates generated examples, separates real
and synthetic counts, and remains marked `benchmark_evidence=False`.

Use `build_training_plan()` to bind reviewed feedback, synthetic rows, manifest
metadata, and a managed training job spec without submitting compute. The plan
keeps the generated rows available for the controlled dataset writer, while
`to_dict()` remains tenant-safe and omits prompt and response text.

```python
plan = builder.build_training_plan(
    reviewed_events,
    reviewer_id="reviewer-passport-1",
    seed=123,
    max_examples=32,
    real_event_count=real_event_count,
    manifest_id="distill-20260513-a",
    dataset_uri="env://DIRECTOR_SYNTHETIC_DISTILLATION_DATASET",
    output_uri="env://DIRECTOR_SYNTHETIC_DISTILLATION_OUTPUT",
    base_model_ref="factcg-deberta-v3-large",
    schedule_id="nightly-reviewed-feedback",
)

rows = plan.training_rows()
training_job = plan.training_job
```

The training job is a request object only. It is not submitted by the builder,
and dataset/output URIs with embedded credentials are rejected.

## Full API

::: director_ai.core.self_evolving.synthetic_distillation.SyntheticExample

::: director_ai.core.self_evolving.synthetic_distillation.SyntheticDistillationBuilder

::: director_ai.core.self_evolving.synthetic_distillation.SyntheticDistillationManifest

::: director_ai.core.self_evolving.synthetic_distillation.SyntheticDistillationPlan
