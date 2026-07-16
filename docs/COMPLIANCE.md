# Compliance Evidence Guide

> **Module**: Director-AI | **Version**: 3.18.0 | **License**: Apache-2.0 core / BUSL-1.1 advanced
>
> © Concepts 1996–2026 Miroslav Šotek. All rights reserved.

How to generate operator-reviewable compliance evidence from a Director-AI
deployment: EU AI Act governance controls and Article 15 reporting, SOC 2 /
ISO/IEC 27001 readiness, the HIPAA documentation packet, and the one-command
evidence kit that bundles all of them. The canonical API-level guide is
[anulum.github.io/director-ai/guide/compliance-reporting](https://anulum.github.io/director-ai/guide/compliance-reporting/);
this file is the repo-level operator guide.

> **What this is — and is not.** Everything Director-AI produces here is
> *readiness evidence for operator review*: computed control signals, accuracy
> metrics from recorded traffic, and documentation packets. None of it is a
> conformity assessment, a certification, an audit opinion, or legal advice.
> Regulatory determinations (whether a deployment is in scope, which
> obligations apply, whether they are met) remain with the deployment
> operator and their counsel.

## 1. Regulatory timeline (EU AI Act, as of July 2026)

The EU AI Act (Regulation (EU) 2024/1689) entered into force on 1 August 2024
and applies in stages. The "Digital Omnibus on AI" amendment — provisionally
agreed by the Council and European Parliament in May 2026, endorsed by
Parliament on 16 June 2026 and approved by the Council on 29 June 2026 —
changed the high-risk application dates:

| Obligation set | Applies from |
|---|---|
| Prohibited practices, AI literacy (Art. 4, 5) | 2 February 2025 |
| GPAI model obligations, governance | 2 August 2025 |
| Transparency obligations (Art. 50) | 2 August 2026 (watermarking grace period to 2 December 2026 for systems on the market before 2 August 2026) |
| High-risk systems — stand-alone (Annex III) | **2 December 2027** (deferred from 2 August 2026) |
| High-risk systems — embedded in regulated products (Annex I) | **2 August 2028** |

Sources: [Council press release, 7 May 2026](https://www.consilium.europa.eu/en/press/press-releases/2026/05/07/artificial-intelligence-council-and-parliament-agree-to-simplify-and-streamline-rules/) ·
[Gibson Dunn omnibus summary](https://www.gibsondunn.com/eu-ai-act-omnibus-agreement-postponed-high-risk-deadlines-and-other-key-changes/).
Verify current dates against the Official Journal before relying on them —
the omnibus was awaiting formal publication when this guide was written.

**Where Director-AI fits.** Director-AI is a hallucination/grounding guardrail
that deployers and providers place around their own AI systems. It is not
itself a high-risk AI system under Annex III, but it *generates the technical
evidence* several AI Act obligations expect from high-risk deployments:
accuracy monitoring (Art. 15), tamper-evident logging (Art. 12), transparency
and interpretability records (Art. 13), human-oversight tracking (Art. 14),
and post-market monitoring signals (Art. 72).

## 2. One command: the evidence kit

```bash
director-ai compliance evidence-kit \
    --db audit/compliance.sqlite \
    --context article15_context.json \
    --config-env \
    --output compliance_evidence/
```

This writes a single reviewable bundle:

| File | Content |
|---|---|
| `INDEX.md` | Contents, skip notes, disclaimer, version, timestamp |
| `governance_controls.{md,json}` | Computed controls with the NIST AI RMF / ISO/IEC 42001 / EU AI Act crosswalk |
| `article15_report.{md,json}` | Accuracy metrics, drift, incidents from the audit trail; full Article 15 technical documentation when `--context` is supplied |
| `soc2_iso_readiness.{md,json}` | SOC 2 / ISO/IEC 27001 readiness controls |
| `hipaa_documentation.{md,json}` | HIPAA documentation packet with deployment obligations |

Degradation is honest: without an audit database the Article 15 section is
skipped with an explicit note and the record-keeping controls report failing
signals — the kit never fabricates evidence it does not have.

The `--context` file supplies the operator-authored narrative Article 15
requires (system name, intended purpose, risk-management / data-governance /
robustness / cybersecurity / human-oversight / post-market-monitoring
summaries) — these cannot be derived from telemetry. See
`Article15TemplateContext` in `director_ai.compliance.reporter` for the field
list; every field is required.

## 3. Computed governance controls (EU AI Act crosswalk)

`director-ai compliance governance` (or `GET /v1/compliance/governance-controls`)
computes eight controls from **observable deployment state** — configuration
knobs, the attached audit log (including a live hash-chain verification), and
evidence artefacts on disk. No control is a stored checkbox.

| Control | EU AI Act | Computed from |
|---|---|---|
| GOV-RISK-01 risk management | Art. 9(1), 9(2) | guard thresholds, adaptive governor, drift wiring |
| GOV-DATA-01 data governance | Art. 10(2), 10(5) | grounding store, PII redaction, tenant isolation |
| GOV-DOC-01 technical documentation | Art. 11(1) | capability inventory, public benchmark evidence, Article 15 wiring |
| GOV-LOG-01 record-keeping | Art. 12(1), 12(2) | live tamper-evident chain verification, sealed entry count |
| GOV-TRA-01 transparency | Art. 13(1), 13(3) | operator documentation, declared profile, interpretability metadata |
| GOV-ACC-01 accuracy monitoring | Art. 15(1), 15(4) | recorded-traffic accuracy reporting availability |
| GOV-OVR-01 human oversight | Art. 14(1), 14(4) | review band configuration, override tracking |
| GOV-PMM-01 post-market monitoring | Art. 72(1), 72(2) | drift analysis over recorded windows, metrics export |

Each control also carries NIST AI RMF 1.0 and ISO/IEC 42001:2023 references,
so the same report serves all three framework conversations.

## 4. The audit trail (Article 12 record-keeping)

The compliance audit log (`compliance_db_path`, SQLite) seals every scored
interaction into a tamper-evident chain: a SHA-256 content hash, a link to the
previous entry's hash, and an HMAC chain tag. `verify_chain()` re-derives the
chain on demand — the GOV-LOG-01 control runs it live on every governance
report. Set `DIRECTOR_AUDIT_HMAC_SECRET` in production so chain tags are
keyed; without it the log warns and seals content hashes only.

**PII never reaches the sealed trail** when redaction is enabled: with
`redact_pii: true` the pipeline redactor masks prompts and responses *before*
sealing, so the seal covers exactly what is stored. The `production` profile
enables the audit trail, the compliance database, and PII redaction together:

```bash
director-ai serve --profile production
```

## 5. Article 15 reporting (accuracy, drift, incidents)

`director-ai compliance report` computes, from the sealed trail: false
positive/negative rates with Wilson confidence intervals, hallucination rate
per model and per time period, drift detection (two-proportion z-test over
review windows), human override rate, and an incident log of rejections.
Formats: Markdown, JSON, HTML, PDF (`--format`), with `--since`/`--until`
window bounds. `director-ai compliance drift` and `status` give quick views;
the same data backs `GET /v1/compliance/report`, `/drift`, and `/dashboard`.

## 6. Other frameworks

- **SOC 2 / ISO/IEC 27001** — `build_soc2_iso_readiness_report()` emits the
  product-readiness control catalogue with evidence references (no raw
  customer data). Included in the evidence kit.
- **HIPAA** — `build_hipaa_documentation_packet()` emits deployment
  obligations (45 CFR 164 references) plus the bounded product/operator
  responsibility statement. Included in the evidence kit.
- **GDPR-supporting features** — on-host inference (prompts and responses
  never leave the deployment), PII redaction before durable storage, tenant
  isolation, and tenant-safe evidence exports. These support a deployment's
  GDPR posture; data-controller obligations remain with the operator.

## 7. Operational checklist

1. Run with `--profile production` (or set `compliance_db_path`,
   `audit_log_path`, `redact_pii` explicitly).
2. Set `DIRECTOR_AUDIT_HMAC_SECRET` to a deployment-held secret.
3. Author the Article 15 context file once per deployment; keep it under
   operator version control.
4. Schedule `director-ai compliance evidence-kit` (e.g. monthly) and archive
   the bundles; regulators and customers expect a review trail, not a
   one-off export.
5. Watch `director-ai compliance drift` — degradation over time is exactly
   what Article 15(4) and Article 72 monitoring are about.
