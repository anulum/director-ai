<!--
SPDX-License-Identifier: CC-BY-4.0
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-AI — Online Calibration
-->

# Online Calibration

Improve the guardrail from reviewed production feedback. Feedback can propose
better thresholds and training jobs, but runtime changes still require an
explicit deployment approval.

## Scores and abstention

`CoherenceScore.score` is the configured scorer's composite coherence signal.
Higher scores indicate stronger coherence on that route. Read `approved`
for the policy decision, since task-specific gates, strict mode and other
checks can also affect it. A score of 0.8 gives no probability that an
answer is true. The default `DirectorConfig.coherence_threshold` is 0.60
and serves as a starting point. Review the effective profile and scorer
for each deployment.

If grounding is expected and `abstained` is true, retrieval yielded no
usable evidence for the factual signal. The logical signal can still produce
a verdict. Route these cases to review or obtain better evidence before
accepting them as grounded. Report **selective coverage** as the fraction
of eligible requests receiving a decision without abstention. Report error
rates on that selected subset with their denominators and keep the
abstention rate visible. A stricter policy can lower coverage even when its
error rate on selected requests improves.

A conformal routing interval is derived from residuals against reviewed
binary labels. `coverage` is the requested target and `is_reliable` measures
whether the sample-count floor was met. An uncalibrated predictor returns
the full range `[0, 1]`, and the routing policy sends unreliable estimates
to human review. Standard split-conformal coverage is marginal over
exchangeable calibration and evaluation examples with a suitable
calibration quantile. It does not establish a case-specific probability,
coverage for every tenant or an error ceiling on automatically approved
answers. Evaluate deployment coverage and error separately by task and
recheck after changes to the model, retrieval, prompts or traffic mix.
[Split-conformal assumptions](https://arxiv.org/abs/2107.07511)

## Workflow

```
Deploy → Collect Feedback → Calibrate → Review Proposal → Deploy
    ↑                                                │
    └────────────────────────────────────────────────┘
```

## Collecting Feedback

```python
from director_ai import FeedbackStore

store = FeedbackStore("my_deployment.db")

# After a human reviews a guardrail decision:
store.report(
    prompt="What is our refund policy?",
    response="We offer 60-day refunds on all products.",
    guardrail_approved=True,  # guardrail said: approved
    human_approved=False,  # human says: wrong (it's 30-day)
    guardrail_score=0.62,
    domain="customer_support",
)

# Corrections accumulate over time
print(f"Total corrections: {store.count()}")
print(f"Disagreements: {len(store.get_disagreements())}")
```

### What Gets Stored

Each correction records:
- The prompt and response text
- The guardrail's score and verdict
- The human's verdict (approved or not)
- A domain tag (optional, for per-domain calibration)
- Timestamp

Storage is SQLite by default (single file, no server needed). Thread-safe with WAL mode for concurrent access.

## Calibrating

```python
from director_ai import OnlineCalibrator, FeedbackStore

store = FeedbackStore("my_deployment.db")
calibrator = OnlineCalibrator(store, min_corrections=20)

# After enough corrections accumulate:
report = calibrator.calibrate()

print(f"Corrections: {report.correction_count}")
print(f"Current accuracy: {report.current_accuracy:.1%}")
print(f"False approval rate: {report.fpr:.3f}")
print(f"False rejection rate: {report.fnr:.3f}")

if report.optimal_threshold is not None:
    print(f"Optimal threshold: {report.optimal_threshold}")
else:
    print("Insufficient data for threshold optimization")
```

### Per-Domain Calibration

```python
support_report = calibrator.calibrate(domain="customer_support")
print(f"Reviewed responses: {support_report.correction_count}")
print(f"Proposed threshold: {support_report.optimal_threshold}")
```

### Error rates and confidence intervals

In `CalibrationReport`, approval is the positive class. `fpr` counts false
approvals among responses humans rejected. `fnr` counts false rejections
among responses humans approved. Preserve those two denominators. A zero
rate with no examples of that class is an unavailable estimate. A threshold
proposal also needs enough positively scored examples and both label classes.
The proposal maximizes balanced accuracy over the selected feedback and
needs separate evaluation on held-out reviews.

For example, 3 false approvals among 50 human-rejected responses give an
observed rate of `3/50 = 0.0600` and a Wilson 95% interval of
`[0.0206, 0.1622]`. The report's `fpr_ci` and `fnr_ci` fields carry the
interval half-width. Wilson intervals are centred differently from the
observed proportion, so adding and subtracting those fields from the
observed rate does not recover the endpoints. [Wilson intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)

The interval expresses sampling uncertainty under the binomial model for a
specified review population. Targeted feedback, correlated examples and
traffic drift can limit how far that estimate generalizes. Publish the
review population, sample counts, model revision, KB snapshot and policy
together. Use a separate, representative evaluation before adopting a
candidate threshold and monitor error and coverage after deployment.

## Task-routed operating points

Keep the deployed NLI checker and premise composition when scoring the
calibration samples. Use JSONL records with `prompt`, `response` and a
boolean `label` that is true for a correct response.

```bash
DIRECTOR_USE_NLI=true DIRECTOR_SCORER_BACKEND=deberta \
  director-ai operating-points calibration.jsonl \
  --target-fpr dialogue=0.045 --target-fpr summarization=0.025 \
  --output operating-points.env
```

This example selects the model-backed DeBERTa path. If the deployment uses
ONNX, use its configured ONNX model for calibration. The automatic backend
can select the Rust heuristic, which cannot supply raw NLI support for
this command. Review the generated environment overlay before deployment.

The command uses a different convention from feedback reports. Here, FPR counts
**false rejections of correct responses**, and catch rate counts rejected
incorrect responses. The fitted FPR is measured on the input sample.
A route with no correct examples cannot be calibrated. A route with no
incorrect examples reports catch rate as `n/a`.

In the archived NLI-only HaluEval evaluation on an NVIDIA L4, the labelled
pair calibration and the end-to-end harness used different premise
composition paths. The reports show why a calibration target needs
independent validation on the deployment path.

| Route | Target false rejection | Pair calibration | End-to-end |
| --- | ---: | ---: | ---: |
| Dialogue | 4.5% | 8/191 (4.2%) | 17/200 (8.5%) |
| Summarization, weakest link | 2.5% | 4/188 (2.1%) | 10/200 (5.0%) |

[Pair calibration packet](https://github.com/anulum/director-ai/blob/9e066ce0019fc0c89685f4e3f8febbfe55b42848/benchmarks/results/wcs2a_operating_points_calibration.json) ·
[Dialogue evaluation](https://github.com/anulum/director-ai/blob/9e066ce0019fc0c89685f4e3f8febbfe55b42848/benchmarks/results/e2e_nli_only_200_wcs2a_dialogue_raw.json) ·
[Summarization evaluation](https://github.com/anulum/director-ai/blob/9e066ce0019fc0c89685f4e3f8febbfe55b42848/benchmarks/results/e2e_nli_only_200_wcs2a_weakest_link.json)

The FactCG LLM-AggreFact results measure per-dataset mean balanced accuracy
on a different evaluation. The leaderboard reports 75.6% at 0.50, the
committed score packet reports 75.8% at 0.46, and per-dataset threshold
replay reports 77.76%. Keep these operating points separate from deployment
error and coverage estimates. [Benchmark operating points](../benchmarks.md)

## Exporting Training Data

After sufficient corrections accumulate (500+), export as a fine-tuning dataset:

```python
training_data = store.export_training_data()
# [{"prompt": "...", "response": "...", "label": 0, "domain": "medical"}, ...]

import json

with open("finetune_data.jsonl", "w") as f:
    for item in training_data:
        f.write(json.dumps(item) + "\n")
```

This dataset can be used with `finetune_nli()` to train a domain-specific NLI model.

## Exporting Calibration Artefacts

For analytics, audit, and MLOps pipelines, export the canonical calibration
schema. The export preserves guardrail score, guardrail verdict, human verdict,
tenant/domain/review identifiers, disagreement status, timestamp, and a
versioned schema marker.

```python
rows = store.export_calibration_rows(domain="medical")

parquet_path = store.export_parquet(
    "exports/medical-feedback.parquet",
    domain="medical",
)
```

Parquet export imports `pyarrow` only at call time and writes via a
same-directory temporary file before replacing the final path. Use
`include_text=False` when the export destination should not receive prompt or
response text.

Optional MLOps artefact logging is also runtime-only:

```python
store.log_export_artifact(
    parquet_path,
    backend="mlflow",
    artifact_name="medical-calibration-feedback",
    metadata={"rows": len(rows), "domain": "medical"},
)
```

Supported backends are `mlflow` and `wandb`; both require an already active run.
Director-AI does not silently create remote runs during feedback export.

## REMANENTIA Recall-Correctness Feedback

`RemanentiaCorrectnessClient` posts derived recall-correctness labels to
`POST /recall/correctness` on a REMANENTIA service. Use it when a
Director-AI verdict should label the recall record that supplied the answer.

```python
from director_ai.core.calibration.recall_correctness import RecallOutcome
from director_ai.core.calibration.recall_correctness_client import (
    RemanentiaCorrectnessClient,
)

client = RemanentiaCorrectnessClient(
    "https://memory.example/api",
    token="remanentia-bearer-token",
)
event_id = client.record(
    RecallOutcome(
        query="What was retrieved?",
        was_correct=True,
        by="director-ai",
    )
)
```

The client returns `None` when REMANENTIA reports that no prior recall exists
for the query. Transport, authentication, and protocol failures raise
`RemanentiaCorrectnessError`; use `try_record()` on answering hot paths where
memory-feedback failures must not break the response. Keep credentials in the
`token` argument. URLs with embedded `user:pass@host` credentials are rejected.

## Self-Improving Guard Loop Gate

Use `SelfImprovingGuardLoop` when calibration or fine-tuning should enter a
production approval workflow. It builds a tenant-safe reviewed-feedback
manifest, rejects unreviewed rows, checks confidence interval width, requires a
rollback ID, and returns a proposal payload only after explicit approval.

```python
from director_ai.core.guard_control import RiskEnvelope
from director_ai.core.self_evolving import SelfImprovingGuardLoop

loop = SelfImprovingGuardLoop(
    store=reviewed_feedback_store,
    risk_envelope=RiskEnvelope(
        action_category="training",
        reversibility="costly",
        domain="regulated",
        calibrated_threshold=0.45,
        no_go_threshold=0.8,
    ),
    policy_id="policy.self_improving.regulated",
)

proposal = loop.propose_calibration_update(
    source_ref="feedback://recent-reviewed",
    current_threshold=0.55,
    candidate_threshold=0.58,
    confidence_low=0.51,
    confidence_high=0.61,
    rollback_id="threshold-profile-20260513-a",
)
```

The loop does not mutate runtime configuration, submit training jobs, or promote
models. It creates an auditable proposal for an external deployment controller.

## Synthetic Distillation

Synthetic examples can improve hard-negative coverage only when they preserve
source provenance. Use `SyntheticDistillationBuilder` to derive deterministic
examples from reviewed feedback events, then attach a
`SyntheticDistillationManifest` to the training proposal.

```python
from director_ai.core.self_evolving import (
    SyntheticDistillationBuilder,
    SyntheticDistillationManifest,
)

builder = SyntheticDistillationBuilder(generator_id="deterministic-v1")
examples = builder.generate(
    reviewed_events,
    reviewer_id="reviewer-passport-1",
    seed=123,
    max_examples=32,
)
manifest = SyntheticDistillationManifest.from_examples(
    examples=examples,
    real_event_count=real_event_count,
    manifest_id="distill-20260513-a",
)
```

For scheduled distillation, build a managed training plan instead of assembling
the job manually:

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
```

`plan.training_job` is a managed training job spec and remains unsubmitted until
an external controller approves and submits it. The plan audit payload excludes
generated prompt and response text.

Synthetic rows stay marked `synthetic=True` and
`benchmark_evidence=False`. Keep benchmark reports split into real, synthetic,
and mixed sets so generated examples are never presented as real measured
evidence.

## Calibration Report

```python
@dataclass
class CalibrationReport:
    correction_count: int
    optimal_threshold: float | None  # None if insufficient data
    current_accuracy: float
    tpr: float  # true positive rate
    tnr: float  # true negative rate
    fpr: float  # false positive rate
    fnr: float  # false negative rate
    fpr_ci: float  # 95% CI half-width
    fnr_ci: float  # 95% CI half-width
```
