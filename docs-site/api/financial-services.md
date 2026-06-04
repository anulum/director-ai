<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-Class AI — Financial services policy API
-->

# Financial Services Policy API

`director_ai.core.financial_services` contains deterministic policy adapters for
regulated financial-services responses. The first public adapter,
`assess_banking_response()`, checks whether a banking answer has evidence
references for product and regulatory claims, numeric evidence for amounts and
rates, and human-review acknowledgement for complaints or investment
recommendations.

The adapter returns tenant-safe audit payloads. Finding details and serialized
reports include codes, policy references, and evidence identifiers; they do not
include raw customer prompts or generated responses.

```python
from director_ai.core.financial_services import assess_banking_response

report = assess_banking_response(
    prompt="What is the current savings account APY?",
    response="The current savings account APY is 4.25%.",
    evidence_refs=("policy://bank-products/savings-disclosure",),
)

assert not report.approved
assert report.blocked_codes == ("numeric_evidence_required",)
```

This surface complements, rather than replaces, `CoherenceScorer` and tuned
finance profiles. Use the scorer for semantic/factual review against a curated
KB, then use this adapter for deterministic sector controls before releasing a
customer-facing answer.

The same check is wired into `ProductionGuard.check()`:

```python
from director_ai.guard import ProductionGuard

guard = ProductionGuard.from_profile("finance")
result = guard.check(
    "What is the standard FDIC deposit coverage limit?",
    "FDIC insurance covers up to $500,000 per depositor.",
    sector_policy="banking",
    evidence_refs=("policy://fdic/deposit-insurance/current",),
    numeric_evidence_refs=("policy://fdic/deposit-insurance/current#limit",),
    policy_refs=("policy://financial-services/deposit-disclosures",),
)

assert not result.approved
assert result.sector_policy_report is not None
assert result.sector_policy_report.blocked_codes == (
    "deposit_insurance_limit_mismatch",
)
```

It is also exposed through `/v1/review` for API deployments:

```json
{
  "prompt": "What is the standard FDIC deposit coverage limit?",
  "response": "FDIC insurance covers up to $500,000 per depositor.",
  "sector_policy": "banking",
  "evidence_refs": ["policy://fdic/deposit-insurance/current"],
  "numeric_evidence_refs": ["policy://fdic/deposit-insurance/current#limit"],
  "policy_refs": ["policy://financial-services/deposit-disclosures"]
}
```

The response includes a `sector_policy` object with the same audit-safe report
shape as `BankingPolicyReport.to_dict()`. Reference arrays are capped at 64
items; each reference must be 1-512 characters, and `jurisdiction` plus
`product_line` must be non-empty when supplied.

## Full API

::: director_ai.core.financial_services.BankingPolicyFinding

::: director_ai.core.financial_services.BankingPolicyReport

::: director_ai.core.financial_services.assess_banking_response
