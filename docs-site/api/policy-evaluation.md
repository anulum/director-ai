# Policy Evaluation

`evaluate_policy_variants()` compares thresholds, profiles, or scorer policies
on the same labelled sample set. It is intended for controlled internal A/B
evaluation before changing production thresholds.

```python
from director_ai import (
    LabelledPolicySample,
    PolicyVariant,
    evaluate_policy_variants,
)

samples = [
    LabelledPolicySample(
        prompt="q1",
        response="supported answer",
        label=True,
        score=0.91,
        dataset="regression",
        benchmark_evidence=True,
    ),
    LabelledPolicySample(
        prompt="q2",
        response="unsupported answer",
        label=False,
        score=0.12,
        dataset="regression",
        benchmark_evidence=True,
    ),
]

report = evaluate_policy_variants(
    samples,
    variants=[
        PolicyVariant(name="balanced", threshold=0.5),
        PolicyVariant(name="strict", threshold=0.7, profile="medical"),
    ],
)
```

Each variant receives the same samples. The result contains balanced accuracy,
precision, recall, false-positive rate, false-negative rate, and the confusion
matrix.

## A/B Comparison

Use `compare_policy_variants()` for a two-arm baseline/candidate comparison.

```python
from director_ai import compare_policy_variants

comparison = compare_policy_variants(
    samples,
    baseline=PolicyVariant(name="current", threshold=0.7),
    candidate=PolicyVariant(name="candidate", threshold=0.5),
)

print(comparison.delta_balanced_accuracy)
print(comparison.winner)
```

`comparison.baseline` and `comparison.candidate` are `PolicyVariantResult`
records — the confusion-matrix metrics (`tp`/`fp`/`tn`/`fn`, balanced
accuracy, false-positive and false-negative rates) for one variant.

## Scoring Callback

If samples do not carry cached scores, pass a score function. The callback
receives the sample and variant so deployments can build profile-specific
scorers outside the harness.

```python
def score_fn(sample, variant):
    scorer = scorers[variant.name]
    approved, score = scorer.review(sample.prompt, sample.response)
    return score.score

report = evaluate_policy_variants(samples, variants=variants, score_fn=score_fn)
```

## Provenance Guardrails

The report separates real benchmark rows, internal rows, and synthetic rows.
`public_benchmark_eligible` is true only when every sample is real benchmark
evidence from exactly one named dataset.

Synthetic or mixed-provenance reports are valid for internal engineering
decisions, but they must not be copied into public benchmark claims.

## Full API

::: director_ai.core.evaluation.policy.LabelledPolicySample

::: director_ai.core.evaluation.policy.PolicyVariant

::: director_ai.core.evaluation.policy.PolicyEvaluationReport

::: director_ai.core.evaluation.policy.PolicyComparisonReport

::: director_ai.core.evaluation.policy.evaluate_policy_variants

::: director_ai.core.evaluation.policy.compare_policy_variants
