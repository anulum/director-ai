# EU AI Act Compliance with Director-AI

A technical guide for engineering teams preparing high-risk AI systems for EU AI Act enforcement (August 2, 2026).

## Scope

The EU AI Act (Regulation 2024/1689) classifies AI systems by risk tier. **High-risk** systems — those used in employment, creditworthiness, education, critical infrastructure, law enforcement, and healthcare — face mandatory requirements under Articles 9–15. General-purpose AI (GPAI) models have separate obligations under Articles 51–56.

This guide covers what Director-AI provides out of the box, what it helps with, and what it does not address. Director-AI is a hallucination guardrail, not a full compliance platform. It covers a specific and critical slice: **factual accuracy measurement, continuous monitoring, and audit documentation**.

## Which Articles Director-AI Addresses

### Article 9 — Risk Management System

High-risk AI systems must implement a risk management system that identifies, evaluates, and mitigates foreseeable risks.

**What Director-AI provides:**

- **Measurable risk metric.** Every LLM response gets a coherence score (0.0–1.0) combining NLI contradiction detection (H_logical) and RAG factual divergence (H_factual). This is a quantified risk signal, not a qualitative assessment.
- **Threshold-based mitigation.** Responses below the configured threshold are blocked automatically. The threshold is configurable per domain (`DirectorConfig.from_profile("medical")` uses 0.3 by default).
- **Streaming halt.** `StreamingKernel` stops token generation mid-stream when coherence drops, preventing hallucinated content from reaching users.

**What Director-AI does not provide:**

- Identification of all foreseeable risks (toxicity, bias, privacy). Director-AI only measures factual coherence.
- Risk management documentation templates. You still need to write your risk management plan; Director-AI supplies the accuracy evidence that goes into it.

### Article 10 — Data and Data Governance

Training data quality requirements apply to the AI system's training data, not to the guardrail's knowledge base. However:

**What Director-AI provides:**

- **Knowledge base validation.** `DatasetTypeClassifier` and `validate_data()` check ingested training/tuning data for quality issues (duplicates, class imbalance, missing fields).
- **Fine-tuning data governance.** `director-ai validate-data <file.jsonl>` reports data quality metrics before fine-tuning the NLI model on domain data.

### Article 11 — Technical Documentation

Systems must be accompanied by technical documentation including performance metrics and known limitations.

**What Director-AI provides:**

- **Automated performance reports.** `ComplianceReporter.generate_report()` produces structured reports with accuracy metrics, confidence intervals, per-model breakdowns, and drift analysis.
- **Per-claim evidence.** `VerifiedScorer` produces per-claim verdicts (supported/contradicted/fabricated/unverifiable) with matched source chunks and traceability scores.
- **Markdown and structured export.** Reports export to Markdown via `report.to_markdown()` for inclusion in technical documentation.

### Article 12 — Record-Keeping

High-risk systems must automatically log events during operation.

**What Director-AI provides:**

- **Audit log.** `AuditLog` records every scored interaction: prompt, response, model, provider, score, verdict, confidence, latency, timestamp, and domain. Stored in SQLite (local) or PostgreSQL (enterprise).
- **Immutable records.** Audit entries are append-only. No deletion API.
- **Query and export.** `AuditLog.query()` supports time-range, model, domain, and score-range filters.

```python
from director_ai import AuditLog, ComplianceReporter

log = AuditLog("production_audit.db")
reporter = ComplianceReporter(log)

# Generate report for the last 30 days
report = reporter.generate_report(days=30)
print(report.to_markdown())
```

### Article 13 — Transparency

Users must be informed about the AI system's capabilities and limitations.

**What Director-AI provides:**

- **Score transparency.** Every response includes a coherence score. Users (or the application layer) can see exactly how confident the system is.
- **Evidence on rejection.** When a response is blocked, the specific KB chunks that contradicted it are returned. No black-box verdicts.
- **Claim-level detail.** `VerifiedScorer` breaks responses into individual claims and labels each with a verdict and source match.

### Article 14 — Human Oversight

High-risk systems must allow human oversight and intervention.

**What Director-AI provides:**

- **Human-in-the-loop pattern.** `ReviewQueue` collects low-confidence responses for human review before they reach end users.
- **Override tracking.** The audit log records human overrides (approved despite guardrail rejection, or rejected despite guardrail approval).
- **Configurable thresholds.** Operators control the coherence threshold, trading off between false positives and missed hallucinations. This is a human decision, not an automated one.

### Article 15 — Accuracy, Robustness, Cybersecurity

The article that names accuracy explicitly. Systems must achieve and maintain declared accuracy levels.

**What Director-AI provides:**

- **Declared accuracy with confidence intervals.** `ComplianceReporter` computes hallucination rate with Wilson score 95% CI. You declare "hallucination rate < X%" and the system monitors it continuously.
- **Drift detection.** `DriftDetector` compares current-period metrics against historical baselines. If accuracy degrades beyond a configured threshold, alerts fire.
- **Feedback loop detection.** `FeedbackLoopDetector` catches self-reinforcing error patterns where the system's own outputs contaminate future scoring.
- **Regression benchmarks.** `director-ai bench` runs the regression suite against your KB to verify accuracy after updates.

```python
from director_ai.compliance import DriftDetector

detector = DriftDetector(
    baseline_hallucination_rate=0.05,
    alert_threshold=0.02,  # alert if rate increases by >2 percentage points
)
detector.check(current_period_metrics)
```

## What Director-AI Does NOT Cover

| EU AI Act Requirement | Status | What You Need |
|---|---|---|
| Bias and fairness (Article 10) | Not covered | Fairness testing tools (Aequitas, Fairlearn) |
| Toxicity and harmful content | Not covered | Content moderation (Llama Guard, NeMo Guardrails) |
| Prompt injection defence | Not covered | Input validation (Rebuff, LLM-Guard) |
| PII and data protection | Not covered | PII detection (Presidio) |
| Conformity assessment | Not covered | Notified body or self-assessment per Annex VI |
| CE marking | Not covered | Legal/regulatory process |
| Post-market surveillance plan | Partially — drift detection feeds into it | Broader monitoring framework |
| Incident reporting (Article 62) | Partially — audit log provides evidence | Reporting process and channels |

Director-AI solves one problem well: factual accuracy of LLM outputs. It generates the accuracy evidence that feeds into your broader compliance documentation.

## Implementation Checklist

For teams deploying Director-AI as part of EU AI Act compliance:

1. **Deploy `AuditLog`** in production — log every scored interaction to SQLite or PostgreSQL.
2. **Set domain-appropriate thresholds** — use `DirectorConfig.from_profile()` or tune with `director-ai tune <labeled.jsonl>`.
3. **Schedule compliance reports** — run `ComplianceReporter.generate_report(days=30)` monthly. Include in Article 11 technical documentation.
4. **Enable drift detection** — configure `DriftDetector` with your baseline accuracy. Wire alerts to your incident management system.
5. **Implement human review** — route low-confidence responses through `ReviewQueue` before they reach users.
6. **Run regression benchmarks** — after every KB update or model change, run `director-ai bench` to verify accuracy hasn't degraded.
7. **Document limitations** — state clearly that Director-AI covers factual coherence only. Combine with other tools for toxicity, bias, PII.

## Timeline

| Date | Milestone |
|------|-----------|
| August 1, 2024 | EU AI Act entered into force |
| February 2, 2025 | Prohibited AI practices take effect |
| August 2, 2025 | GPAI model obligations take effect |
| **August 2, 2026** | **High-risk AI system obligations take effect** |
| August 2, 2027 | Obligations for Annex I high-risk systems |

The high-risk deadline is 4 months away at time of writing. Systems deployed before August 2, 2026 must comply by the deadline. Systems already on the market have until August 2, 2027 if they are substantially modified.

## Further Reading

- [EU AI Act full text (EUR-Lex)](https://eur-lex.europa.eu/eli/reg/2024/1689/oj)
- [Compliance Reporting guide](compliance-reporting.md) — detailed API reference for `ComplianceReporter`
- [Architecture overview](architecture.md) — how the scoring pipeline works
- [Threshold Tuning](threshold-tuning.md) — finding the right coherence threshold for your domain

---

*Director-AI provides the accuracy measurement and audit evidence layer. It is one component of a complete EU AI Act compliance programme, not a substitute for legal review or conformity assessment.*
