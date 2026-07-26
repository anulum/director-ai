# EU AI Act Compliance Reporting

Automated Article 15 documentation — accuracy metrics, drift detection, and audit trails from production data.

## Why Compliance Reporting?

The EU AI Act Article 15 requires high-risk AI systems to document accuracy metrics, maintain audit trails, and demonstrate continuous monitoring. [Regulation (EU) 2026/1744](https://eur-lex.europa.eu/eli/reg/2026/1744/oj), the Digital Omnibus on AI, was published in the Official Journal on **24 July 2026** and enters into force on **27 July 2026**. It sets application of high-risk obligations to **2 December 2027** for stand-alone Annex III systems and **2 August 2028** for AI embedded in regulated products; transparency obligations (Article 50) still apply from **2 August 2026**. Penalties for non-compliance with high-risk obligations reach up to **€15M or 3% of global turnover** (Article 99(4); prohibited practices carry up to €35M or 7%).

Director-AI generates this documentation automatically from production scoring data. Self-hosted, so your data never leaves your infrastructure.

## Quick Start

```python
from director_ai import (
    AnnexIVTechnicalDocumentationContext,
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
    annex_iv=AnnexIVTechnicalDocumentationContext(
        provider_name="Example Provider GmbH",
        system_version="2026.07",
        previous_version_relationship="Supersedes 2026.06; policy-only update.",
        external_dependencies="Director-AI and the approved model endpoint.",
        software_firmware_requirements="CPython 3.12; no firmware dependency.",
        distribution_forms="Container image and authenticated API.",
        intended_hardware="Operator-qualified x86-64 server.",
        user_interface="Authenticated review API and operator dashboard.",
        instructions_for_use="See the deployment runbook.",
        development_methods="Reviewed source changes and pinned dependencies.",
        design_specifications="Thresholded evidence-grounding guardrail.",
        architecture_and_resources="Gateway, scorer, audit store, review queue.",
        data_requirements="Versioned grounding corpus and eval partitions.",
        predetermined_changes="Threshold changes require release review.",
        validation_and_testing="Focused tests, preflight, release evidence.",
        monitoring_functioning_control="Metrics, drift, incidents, overrides.",
        performance_metric_rationale="Rates and Wilson intervals match the risk.",
        lifecycle_changes="Changes are recorded in the release ledger.",
        standards_and_specifications="Operator-maintained standards register.",
        eu_declaration_of_conformity_ref="Pending applicability determination.",
    ),
)
print(report.to_article15_markdown(context))
```

## Computed NIST AI RMF / ISO 42001 / EU AI Act Controls

`compute_governance_controls()` builds a **computed** control set — unlike
the static readiness catalogue below, every status is derived from
observable deployment state at call time: DirectorConfig knobs (guard
thresholds, PII redaction, tenant routing, vector backend), the attached
tamper-evident audit log (including a live `verify_chain()` pass over the
sealed hash chain), and the presence of documentation evidence artefacts
under an operator-supplied evidence root. Each `GovernanceControl` carries
crosswalk references to NIST AI RMF 1.0 (function/category level, e.g.
`GOVERN 1`, `MEASURE 2`), ISO/IEC 42001:2023 (clause or Annex A level,
e.g. `Clause 6.1`, `A.7`), and EU AI Act articles, and derives its
status from named `ControlSignal` observations — every signal records
what was actually seen, so a missing audit log degrades honestly instead
of aborting. The eight controls cover risk management (Article 9), data
governance (Article 10), technical documentation (Article 11),
record-keeping (Article 12), transparency (Article 13), accuracy
monitoring (Article 15), human oversight (Article 14), and post-market
monitoring (Article 72).

```python
from director_ai.compliance import (
    AuditLog,
    GovernanceControlsReport,
    compute_governance_controls,
)
from director_ai.core.config import DirectorConfig

report: GovernanceControlsReport = compute_governance_controls(
    config=DirectorConfig.from_env(),
    audit_log=AuditLog("director_audit.db"),
    evidence_root=".",
)
print(report.to_markdown())        # or report.to_dict() for JSON
```

Server: `GET /v1/compliance/governance-controls` (add `?fmt=md` for
Markdown) — this endpoint never 503s; an unconfigured audit log is
reported as a failing record-keeping signal, which is the finding the
operator needs to see. CLI: `director-ai compliance governance
[--db PATH] [--format md|json] [--config-env] [--evidence-root DIR]`.

The report is computed governance-readiness evidence only; it is not an
EU AI Act conformity assessment, a NIST AI RMF attestation, an ISO/IEC
42001 certification, an audit opinion, or legal advice. Note: chain
verification across process restarts requires a durable
`DIRECTOR_AUDIT_HMAC_SECRET`; without it the seal is per-process and the
record-keeping signal will honestly report the mismatch.

## One Command: the Evidence Kit

`director-ai compliance evidence-kit` assembles every compliance artefact
into a single reviewable directory — the computed governance controls, the
Article 15 report (full technical documentation when `--context` is
supplied), the SOC 2 / ISO 27001 readiness report, the HIPAA documentation
packet, and an `INDEX.md` recording what was produced, what was skipped,
and why:

```bash
director-ai compliance evidence-kit \
    --db audit/compliance.sqlite \
    --context article15.json \
    --config-env \
    --output compliance_evidence/
```

Degradation is honest: without an audit database the Article 15 section is
skipped with an explicit note and the record-keeping controls report
failing signals — the kit never fabricates evidence it does not have. In
code, the same assembly is `director_ai.cli_verify.evidence_kit.build_evidence_kit()`.

## SOC 2 / ISO 27001 / HIPAA Readiness

`build_soc2_iso_readiness_report()` generates a tenant-safe readiness crosswalk
for customer security reviews. It maps Director-AI evidence references to SOC 2
Trust Services Criteria categories and ISO/IEC 27001:2022 Annex A-style control
references, then produces JSON, Markdown, and Trust Console control rows. The
same control rows can carry HIPAA Security Rule references where product
evidence supports the mapping. This is readiness evidence only; it is not a SOC
2 report, ISO/IEC 27001 certification, HIPAA legal advice, OCR determination, or
auditor opinion.

```python
from director_ai.compliance import (
    HipaaDeploymentObligation,
    ReadinessStatus,
    Soc2IsoControl,
    build_hipaa_documentation_packet,
    build_soc2_iso_readiness_report,
)

report = build_soc2_iso_readiness_report(
    controls=[
        Soc2IsoControl(
            control_id="SEC-01",
            title="Tenant authentication and access isolation",
            soc2_criteria=("security", "confidentiality"),
            iso27001_refs=("A.5.15", "A.8.3"),
            hipaa_security_refs=("45 CFR 164.308(a)(4)", "45 CFR 164.312(a)(1)"),
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
incident review, vulnerability evidence, and change management. It also exposes
a SOC 2 Type I path: define the system boundary, attach dated evidence,
remediate or document warnings, and freeze the observation point only after an
independent auditor or authorised internal exception accepts the packet.
Controls can be overridden per deployment so operators can add auditor-owned
evidence references without serialising raw evidence or customer payloads.

`build_hipaa_documentation_packet()` adds the deployment-owned HIPAA
documentation layer around the readiness report. It is based on the Security
Rule structure described by HHS: administrative, physical, and technical
safeguards for electronic protected health information, with the operative rule
text in 45 CFR Part 164 Subpart C. The packet records references and required
operator actions; it never includes raw PHI, raw interaction text, or raw
security evidence.

```python
packet = build_hipaa_documentation_packet(
    generated_at="2026-06-18T08:00:00Z",
    obligations=[
        HipaaDeploymentObligation(
            obligation_id="HIPAA-AUD-01",
            title="Audit controls and activity review",
            hipaa_security_refs=("45 CFR 164.312(b)",),
            status=ReadinessStatus.PASS,
            evidence_refs=("tests/test_audit_chain.py",),
            operator_action="Enable audit review and retain reviewer sign-off.",
        ),
    ],
    phi_handling_summary=(
        "Default exports exclude raw PHI; deployment evidence stays in the "
        "operator-controlled evidence store."
    ),
)

hipaa_payload = packet.to_dict()
hipaa_markdown = packet.to_markdown()

assert hipaa_payload["privacy"] == {
    "payload_classification": "tenant_safe",
    "raw_phi_included": False,
    "raw_interaction_text_included": False,
    "raw_security_evidence_included": False,
    "hipaa_compliance_claimed": False,
}
```

Default HIPAA obligations cover risk analysis and risk management, business
associate agreement review, audit controls and activity review, access control,
incident response, and contingency planning. The packet intentionally marks most
deployment obligations as `warning` until the operator attaches environment
evidence such as identity-provider controls, ePHI data-flow inventory, backup
restore tests, incident contacts, and agreement records.

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

When `context.annex_iv` is supplied, the JSON adds
`annex_iv_technical_documentation` and the Markdown appends the numbered Annex IV
document. Article 11(1) requires high-risk-system technical documentation to
contain at least the Annex IV elements. Director-AI therefore refuses a partial
nested Annex IV context instead of presenting it as complete. Explicit
`not applicable — <reason>` entries are accepted because applicability is a
provider/legal determination, not a telemetry inference. The generated file is
an evidence template, not a conformity assessment or legal advice.

Export the full template from the CLI by supplying the operator context as a JSON
file:

```bash
director-ai compliance report --db audit.db --format json --context article15.json
director-ai compliance report --db audit.db --context article15.json  # markdown
```

`article15.json` carries the operator-authored narrative fields (`system_name`,
`intended_purpose`, `risk_management_summary`, `human_oversight_summary`, …)
and may contain the nested `annex_iv` object shown above.
Without `--context`, `--format json` still emits the compact metrics summary for
quick checks; with it, the command emits the complete Article 15 record.

### PDF export (`reports` extra)

For a regulator- or auditor-facing document, render the report straight to PDF.
This needs the `reports` extra (`pip install 'director-ai[reports]'`, which pulls
in WeasyPrint); the PDF inherits the same print-ready layout as the HTML report:

```bash
director-ai compliance report --db audit.db --format pdf --output article15.pdf
```

In code, `director_ai.compliance.report_templates` exposes `render_compliance_pdf`,
`render_cost_pdf`, and `render_swarm_pdf` (each returns PDF `bytes`), plus the
generic `html_to_pdf(html)` for any of the HTML renderers. Without the extra
installed, these raise `DependencyError` with the install hint.

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
