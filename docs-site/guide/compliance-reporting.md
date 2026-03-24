# EU AI Act Compliance Reporting

Automated Article 15 documentation — accuracy metrics, drift detection, and audit trails from production data.

## Why Compliance Reporting?

The EU AI Act Article 15 requires high-risk AI systems to document accuracy metrics, maintain audit trails, and demonstrate continuous monitoring. Enforcement begins **August 2, 2026**. Fines reach up to **€35M or 7% of global turnover**.

Director-AI generates this documentation automatically from production scoring data. No manual effort. No consultants. Self-hosted, so your data never leaves your infrastructure.

## Quick Start

```python
from director_ai import AuditLog, AuditEntry, ComplianceReporter
import time

# 1. Log every scored LLM interaction
log = AuditLog("production_audit.db")

log.log(AuditEntry(
    prompt="What is our refund policy?",
    response="We offer a 30-day refund policy on all products.",
    model="gpt-4o",
    provider="openai",
    score=0.85,
    approved=True,
    verdict_confidence=0.92,
    task_type="qa",
    domain="customer_support",
    latency_ms=18.5,
    timestamp=time.time(),
))

# 2. Generate Article 15 report
reporter = ComplianceReporter(log)
report = reporter.generate_report()

# 3. Export as Markdown
print(report.to_markdown())
```

## What the Report Contains

### 1. Accuracy Metrics (Article 15(1))

| Metric | Description |
|--------|-------------|
| Overall hallucination rate | Fraction of responses rejected, with 95% Wilson CI |
| Average coherence score | Mean NLI-based coherence across all interactions |
| Average verdict confidence | Mean guardrail self-confidence |
| Average scoring latency | Time to score each response |

### 2. Human Oversight (Article 14)

| Metric | Description |
|--------|-------------|
| Human overrides recorded | How often humans disagreed with the guardrail |
| Human override rate | Override fraction — indicates calibration quality |

### 3. Per-Model Breakdown

Each LLM model used gets its own accuracy stats:
- Hallucination rate with confidence intervals
- Average score and confidence
- Latency comparison

### 4. Drift Detection (Article 15(3))

The reporter splits the time range into weekly windows and compares hallucination rates across periods. If the rate increases by more than the drift threshold (default 5pp), an alert fires.

```python
reporter = ComplianceReporter(
    log,
    drift_window_days=7,
    drift_threshold=0.05,  # 5pp increase triggers alert
)
report = reporter.generate_report(
    since=time.time() - 30 * 86400,  # last 30 days
)

if report.drift_detected:
    print(f"Drift severity: {report.drift_severity:.2%}")
    # Action: retrain, recalibrate, or switch models
```

### 5. Incident Summary

Total rejections (potential hallucinations blocked) during the reporting period.

## Integration with Gateway

When director-ai runs as a proxy/gateway, every LLM call gets automatically scored and logged. The compliance reporter reads from the same audit database.

```python
# In your gateway setup:
from director_ai import AuditLog, ComplianceReporter

log = AuditLog("/var/lib/director-ai/audit.db")
reporter = ComplianceReporter(log)

# Weekly cron job:
report = reporter.generate_report()
with open(f"/reports/article15_{date}.md", "w") as f:
    f.write(report.to_markdown())
```

## Filtering

Reports can be filtered by model, domain, tenant, and time range:

```python
# Medical domain only, last 7 days
report = reporter.generate_report(
    since=time.time() - 7 * 86400,
    domain="medical",
)

# Specific model comparison
gpt_report = reporter.generate_report(model="gpt-4o")
claude_report = reporter.generate_report(model="claude-4")
```

## Data Types

```python
@dataclass
class AuditEntry:
    prompt: str
    response: str
    model: str
    provider: str
    score: float
    approved: bool
    verdict_confidence: float
    task_type: str
    domain: str
    latency_ms: float
    timestamp: float
    tenant_id: str = ""
    human_override: bool | None = None

@dataclass
class Article15Report:
    total_interactions: int
    overall_hallucination_rate: float  # with CI
    model_metrics: list[ModelMetrics]
    drift_detected: bool
    drift_severity: float
    incident_count: int
    # ... full fields in API reference
```
