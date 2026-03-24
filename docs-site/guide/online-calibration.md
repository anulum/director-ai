# Online Calibration

Improve the guardrail from production feedback — the longer you use it, the better it gets for your data.

## Why Online Calibration?

Every guardrail ships with a default threshold (0.30 for director-ai). But the optimal threshold depends on your deployment: your documents, your domain, your tolerance for false positives vs false negatives. Online calibration collects human corrections and automatically adjusts the threshold to minimize errors on your actual data.

## Workflow

```
Deploy → Collect Feedback → Calibrate → Apply → Deploy (improved)
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
