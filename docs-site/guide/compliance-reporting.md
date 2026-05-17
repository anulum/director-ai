# EU AI Act Compliance Reporting

Automated Article 15 documentation — accuracy metrics, drift detection, and audit trails from production data.

## Why Compliance Reporting?

The EU AI Act Article 15 requires high-risk AI systems to document accuracy metrics, maintain audit trails, and demonstrate continuous monitoring. Enforcement begins **August 2, 2026**. Fines reach up to **€35M or 7% of global turnover**.

Director-AI generates this documentation automatically from production scoring data. No manual effort. No consultants. Self-hosted, so your data never leaves your infrastructure.

## Quick Start

```python
from director_ai import (
    Article15TemplateContext,
    AuditLog,
    AuditEntry,
    ComplianceReporter,
)
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

# 4. Produce regulator-facing Article 15 technical documentation
context = Article15TemplateContext(
    system_name="Director-AI customer-support guard",
    intended_purpose="Score generated answers against approved support facts.",
    deployment_context="EU customer-support assistant gateway.",
    risk_management_summary="Low-score answers are blocked and routed to review.",
    data_governance_summary="Audit rows are tenant-scoped and PII redaction is enabled.",
    robustness_summary="NLI scoring, streaming halt, drift checks, and red-team tests run.",
    cybersecurity_summary="API-key tenant binding, rate limits, and signed KB entries are enabled.",
    human_oversight_summary="Reviewers can approve, reject, or request regeneration.",
    post_market_monitoring_summary="Operations reviews drift, incidents, and overrides weekly.",
    known_limitations=("Does not replace human approval for regulated advice.",),
    residual_risks=("Knowledge-base facts can become stale between reviews.",),
    evidence_refs=("docs/PRODUCTION_CHECKLIST.md#compliance", "SECURITY.md"),
)
print(report.to_article15_markdown(context))
```

## SOC 2 / ISO 27001 Readiness

`build_soc2_iso_readiness_report()` generates a tenant-safe readiness crosswalk
for customer security reviews. It maps Director-AI evidence references to SOC 2
Trust Services Criteria categories and ISO/IEC 27001:2022 Annex A-style control
references, then produces JSON, Markdown, and Trust Console control rows. This
is readiness evidence only; it is not a SOC 2 report, ISO/IEC 27001
certification, or auditor opinion.

```python
from director_ai.compliance import (
    ReadinessStatus,
    Soc2IsoControl,
    build_soc2_iso_readiness_report,
)

report = build_soc2_iso_readiness_report(
    controls=[
        Soc2IsoControl(
            control_id="SEC-01",
            title="Tenant authentication and access isolation",
            soc2_criteria=("security", "confidentiality"),
            iso27001_refs=("A.5.15", "A.8.3"),
            status=ReadinessStatus.PASS,
            evidence_refs=("tests/test_server_auth.py", "tests/test_enterprise.py"),
            owner="security",
            updated_at="2026-05-17",
        ),
    ],
)

payload = report.to_dict()
markdown = report.to_markdown()
trust_controls = report.to_trust_controls()

assert payload["privacy"] == {
    "payload_classification": "tenant_safe",
    "raw_security_evidence_included": False,
    "certification_claimed": False,
}
```

The default catalogue covers tenant isolation, PII redaction, monitoring,
incident review, vulnerability evidence, and change management. Controls can be
overridden per deployment so operators can add auditor-owned evidence references
without serialising raw evidence or customer payloads.

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

### 6. Article 15 Technical Documentation Template

`Article15TemplateContext` adds the operator-controlled evidence that cannot be
derived from metrics alone: intended purpose, deployment context, risk
management, data governance, robustness controls, cybersecurity controls, human
oversight, post-market monitoring, known limitations, residual risks, and
evidence references. `Article15Report.to_article15_template(context)` returns a
tenant-safe dictionary with `privacy.raw_interaction_text_included = false`.
`Article15Report.to_article15_markdown(context)` renders the same structure as a
reviewable technical-documentation draft.

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
