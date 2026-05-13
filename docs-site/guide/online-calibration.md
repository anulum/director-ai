# Online Calibration

Improve the guardrail from reviewed production feedback. Feedback can propose
better thresholds and training jobs, but runtime changes still require an
explicit deployment approval.

## Why Online Calibration?

Every guardrail ships with a default threshold (0.30 for director-ai). But the optimal threshold depends on your deployment: your documents, your domain, your tolerance for false positives vs false negatives. Online calibration collects human corrections and proposes threshold updates that minimise errors on your actual data.

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
    guardrail_approved=True,   # guardrail said: approved
    human_approved=False,      # human says: wrong (it's 30-day)
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
print(f"FPR: {report.fpr:.3f} ± {report.fpr_ci:.3f}")
print(f"FNR: {report.fnr:.3f} ± {report.fnr_ci:.3f}")

if report.optimal_threshold is not None:
    print(f"Optimal threshold: {report.optimal_threshold}")
else:
    print("Insufficient data for threshold optimization")
```

### Per-Domain Calibration

```python
# Calibrate medical domain separately
med_report = calibrator.calibrate(domain="medical")
fin_report = calibrator.calibrate(domain="finance")

print(f"Medical FPR: {med_report.fpr:.3f}")
print(f"Finance FPR: {fin_report.fpr:.3f}")
```

### Confidence Intervals

Error rates include Wilson score 95% confidence intervals. With 50 corrections where 3 are false positives:

```
FPR: 0.060 ± 0.042  (95% CI: [0.018, 0.102])
```

This is what makes "we guarantee <X% hallucination rate" measurable per deployment — not a marketing claim, but a statistical fact with confidence bounds.

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

## Data Moat

The feedback loop creates a switching cost: the longer a customer uses director-ai, the more calibration data they accumulate. Switching to a competitor means losing that deployment-specific accuracy improvement.
