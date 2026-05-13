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

## Full API

::: director_ai.core.self_evolving.synthetic_distillation.SyntheticExample

::: director_ai.core.self_evolving.synthetic_distillation.SyntheticDistillationBuilder

::: director_ai.core.self_evolving.synthetic_distillation.SyntheticDistillationManifest
