<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-Class AI — Finance domain cookbook
-->

# Finance Domain Cookbook

## Complete Working Example

```python
from director_ai.core import CoherenceScorer, GroundTruthStore

store = GroundTruthStore()  # empty — populate with your KB
store.add("savings APY", "Our savings account APY is 4.25% as of February 2026.")
store.add(
    "FDIC",
    "FDIC insurance covers up to $250,000 per depositor, per insured bank, "
    "for each ownership category.",
)
store.add("wire transfer", "Wire transfers take 1-3 business days.")

scorer = CoherenceScorer(threshold=0.30, ground_truth_store=store)

# Correct → approved
approved, score = scorer.review("What is the savings APY?",
    "The current savings account APY is 4.25%.")
print(f"Correct: approved={approved}, score={score.score:.2f}")

# Wrong → rejected
approved, score = scorer.review("What is the FDIC limit?",
    "FDIC covers up to $500,000 per depositor.")
print(f"Wrong:   approved={approved}, score={score.score:.2f}")
```

## Configuration

```python
scorer = CoherenceScorer(
    # FinanceBench metadata records 100% FPR without KB grounding at t=0.30.
    # Treat this as an onboarding default, then tune on labelled local traces.
    threshold=0.30,
    soft_limit=0.35,
    use_nli=True,
    ground_truth_store=store,
    cache_size=4096,   # High cache for repeated product queries
    cache_ttl=3600,    # 1-hour cache for stable financial facts
)
```

## Knowledge Base

```python
store = VectorGroundTruthStore()
store.ingest([
    "Our savings account APY is 4.25% as of February 2026.",
    "Wire transfers take 1-3 business days.",
    "FDIC insurance covers up to $250,000 per depositor, per insured bank, "
    "for each ownership category.",
    "Minimum balance for premium checking is $5,000.",
])
```

## Compliance Pattern

```python
from director_ai.core.audit import AuditLogger
from director_ai.core.policy import Policy

# Audit all interactions for compliance
audit = AuditLogger(path="/var/log/director-ai/finance")

# Policy: block responses mentioning specific stock recommendations
policy = Policy(
    patterns=[{"name": "no_stock_advice", "regex": r"(buy|sell|short)\s+(stock|shares)", "action": "block"}],
    forbidden=["stock recommendation", "investment advice"],
)
```

## Banking Policy Adapter

Use `assess_banking_response()` beside the scorer when a response contains
financial-product claims, deposit insurance language, complaint handling, or
investment recommendations. The adapter is deterministic and audit-safe: it
stores evidence identifiers and finding codes, not the raw customer prompt or
response text. It also blocks financial-crime-control bypass guidance and
lending approval guarantees before release.

```python
from director_ai.core.financial_services import assess_banking_response

report = assess_banking_response(
    prompt="What is the standard FDIC deposit coverage limit?",
    response=(
        "FDIC insurance covers up to $250,000 per depositor, per insured "
        "bank, for each ownership category."
    ),
    evidence_refs=("policy://fdic/deposit-insurance/current",),
    numeric_evidence_refs=("policy://fdic/deposit-insurance/current#limit",),
    policy_refs=("policy://financial-services/deposit-disclosures",),
)

assert report.approved
```

Wrong or unsupported numeric claims halt before the answer is released:

```python
report = assess_banking_response(
    prompt="What is the current savings account APY?",
    response="The current savings account APY is 4.25%.",
    evidence_refs=("policy://bank-products/savings-disclosure",),
)

assert not report.approved
assert "numeric_evidence_required" in report.blocked_codes
```

Complaint/dispute handling and investment recommendations require a
human-review acknowledgement from the operator workflow before automatic
approval:

```python
report = assess_banking_response(
    prompt="I need to dispute an unauthorized wire transfer.",
    response="We opened a dispute case and will contact you with next steps.",
    evidence_refs=("policy://financial-services/complaint-handling",),
    policy_refs=("policy://financial-services/complaint-handling",),
)

assert report.requires_human_review
```

Financial-crime-control bypass guidance and guaranteed lending approvals are
hard blocks:

```python
report = assess_banking_response(
    prompt="How can I send a large transfer without extra checks?",
    response=(
        "Split the wire transfer into smaller payments so it avoids KYC, AML, "
        "and sanctions screening."
    ),
    evidence_refs=("policy://financial-services/aml-controls",),
    policy_refs=("policy://financial-services/aml-controls",),
)

assert not report.approved
assert "financial_crime_control_bypass_blocked" in report.blocked_codes
```

This adapter is a configurable software guardrail. It does not provide legal,
regulatory, fiduciary, or investment advice; production teams must bind it to
reviewed policies, current product disclosures, jurisdiction-specific review
flows, and labelled calibration traces.

## Compliance Cost Avoidance (Illustrative Estimates)

!!! warning "These are illustrative industry estimates, not measured results from Director-AI deployments. Validate on your own workload before budgeting."

| Risk | Exposure Without Director-AI | With Director-AI |
|------|------------------------------|-----------------|
| Wrong product terms quoted to customer | Regulatory review, remediation, customer correction | Caught mid-stream, never reaches customer (estimated) |
| Hallucinated interest rate or fee | Customer dispute + regulatory review | KB-verified before display (estimated) |
| Unauthorised investment recommendation | Escalation, audit finding, remediation | Policy engine blocks + audit trail (estimated) |
| Financial-crime-control bypass guidance | Suspicious-activity escalation, customer harm, and regulator review | Deterministic block with tenant-safe finding code (estimated) |
| Guaranteed lending approval claim | Unsuitable sale risk and adverse underwriting review | Deterministic block before release (estimated) |

At 5,000 customer interactions/day, a 0.1% hallucination rate means 5 wrong answers daily. Over a year, that's 1,825 potential compliance incidents. With Director-AI, the catch rate reduces this to < 20/year (estimated, assuming high catch rate with KB + NLI at threshold=0.30). These figures are planning estimates based on industry rates, not measured deployment data.

## Key Considerations

- **Regulatory compliance**: audit all rejections and approvals
- **Rate data freshness**: financial data changes — set appropriate `cache_ttl`
- **Operator-approved disclosures**: add reviewed disclosures to relevant outputs
- **Multi-tenant isolation**: use `TenantRouter` for different product lines
