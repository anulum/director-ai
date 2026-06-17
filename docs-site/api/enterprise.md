# Enterprise API

Multi-tenant scoring isolation, declarative policy rules, and audit logging. These modules are lazy-loaded — importing `director_ai` does not pull them in until accessed.

```python
from director_ai.enterprise import TenantRouter, Policy, AuditLogger
```

## PII Redaction

`PIIRedactor` masks sensitive values before prompts, responses, or audit
payloads leave the tenant boundary. The default path is dependency-free regex
PII detection with the Rust scanner used automatically when the
`backfire-kernel` wheel is present. Presidio can still be added as an optional
detector for named-entity enrichment in regulated deployments; the committed
PII benchmark records Director and Presidio span-level precision/recall on the
same labelled corpus so operators can compare detector behavior before enabling
the optional stack.

```python
from director_ai.enterprise import PIIRedactor

redactor = PIIRedactor()
report = redactor.redact_with_report(
    "Email jane.doe@example.com, call +1 415 555 0101, card 4111-1111-1111-1111."
)

assert report.redacted_text == "Email [EMAIL], call [PHONE], card [CARD]."
assert report.category_counts == {"card": 1, "email": 1, "phone": 1}

audit_payload = report.to_dict()
assert audit_payload["privacy"] == {
    "payload_classification": "tenant_safe",
    "raw_payload_included": False,
}
```

The structured report exposes detector name, category, offsets, replacement
token, score, and category counts. It never serialises the raw matched value,
so the report can be stored in tenant-safe audit metadata. Stable replacement
tokens are `[EMAIL]`, `[PHONE]`, `[CARD]`, `[SSN]`, `[IBAN]`, `[IPV4]`,
`[PASSPORT]`, `[PERSON]`, and `[PHI]`; unknown detector categories fall back to
an uppercase bracketed token.

## Content Moderation Wrapper

`ContentModerator` is the commercial wrapper for moderation enforcement. It
combines PII redaction with toxicity detection and returns one decision:
`allow`, `redact`, `warn`, or `block`. The default deployment path redacts PII
with the dependency-free detector stack and blocks keyword toxicity; operators
can inject stronger model-backed detectors or switch toxicity handling to
warn-only mode.

```python
from director_ai.enterprise import ContentModerator, ModerationAction

moderator = ContentModerator(toxicity_action=ModerationAction.BLOCK)
result = moderator.moderate("Email jane@example.com, then go kill yourself.")

assert result.blocked is True
assert result.safe_text == "Email [EMAIL], then go kill yourself."

audit_payload = result.to_dict()
assert audit_payload["privacy"] == {
    "payload_classification": "tenant_safe",
    "raw_input_included": False,
}
```

The wrapper's metadata contains detector names, categories, offsets, actions,
scores, replacement tokens, and aggregate category counts. It does not serialise
raw matched values. Use the lower-level `RegexPIIDetector`,
`PresidioPIIDetector`, `KeywordToxicityDetector`, or `DetoxifyDetector` only
when building a custom moderation policy.

## Custom Rules DSL

`CustomRuleset` loads strict operator-owned JSON or YAML policy documents and
compiles them into the existing runtime `Policy` engine. Unknown fields,
duplicate rule identifiers, invalid regex patterns, invalid actions, invalid
thresholds, and non-positive numeric limits are rejected before a policy can be
registered.

```yaml
version: director.rules.v1
name: tenant-alpha.safety
rules:
  - id: no-passwords
    kind: forbidden
    value: share passwords
    name: No password disclosure
    action: block
    source: security-baseline.md#L10
  - id: ticket-redaction
    kind: pattern
    value: "TICKET-\\d{6}"
    name: Ticket reference
    action: redact
  - id: length-cap
    kind: max_length
    value: 800
  - id: citations
    kind: required_citations
    value: 2
```

```python
from director_ai.enterprise import CustomRuleset

ruleset = CustomRuleset.from_file("tenant-alpha.rules.yaml")
policy = ruleset.to_policy()

violations = policy.check("share passwords in TICKET-123456")
assert {violation.rule for violation in violations} == {
    "forbidden",
    "pattern:Ticket reference",
    "required_citations",
}
```

Supported rule kinds are `forbidden`, `pattern`, `max_length`, and
`required_citations`; supported actions are `block`, `warn`, and `redact`.
`CustomRuleset.to_policy_bundle(version=...)` returns the immutable compiler
bundle when operators need deterministic registry hot-swap behaviour.

## TenantRouter

Isolates scoring configuration per tenant. Each tenant gets its own `CoherenceScorer` instance with independent thresholds, knowledge bases, and caching.

```python
from director_ai.enterprise import TenantRouter

router = TenantRouter()
router.register("tenant_a", threshold=0.7, use_nli=True)
router.register("tenant_b", threshold=0.5, use_nli=False)

scorer = router.get_scorer("tenant_a")
approved, score = scorer.review(query, response)
```

## Policy

Declarative rule engine for content filtering. Runs before coherence scoring.

```python
from director_ai.enterprise import Policy

policy = Policy(rules=[
    {"pattern": r"(buy|sell|short)\s+(stock|shares)", "action": "reject"},
    {"pattern": r"\b(SSN|social security)\b", "action": "redact"},
])

result = policy.evaluate(response_text)
if result.rejected:
    print(f"Policy violation: {result.rule}")
```

## AuditLogger

SQLite-backed audit logging for compliance. Records every review decision with full context.

```python
from director_ai.enterprise import AuditLogger

logger = AuditLogger(log_dir="/var/log/director-ai/audit")
logger.log(query, response, score, approved=True)

# Query audit trail
entries = logger.query(tenant_id="tenant_a", since="2026-01-01")
```

## License Matrix

The enterprise module is part of the **Advanced & Labs** tier, licensed under
**BUSL-1.1** (source-available).

| Use Case | License Required |
|----------|-----------------|
| Evaluation / prototyping | BUSL-1.1 (free) |
| Non-production internal use | BUSL-1.1 (free) |
| Production deployment | Commercial license |
| Hosted / SaaS product | Commercial license |

The Apache-2.0 core, by contrast, is free in production. See
[Licensing](../licensing.md) for pricing and terms.

## Full API

::: director_ai.core.tenant.TenantRouter

::: director_ai.core.safety.audit.AuditLogger

::: director_ai.enterprise.redactor.PIIRedactor

::: director_ai.enterprise.redactor.PIIRedactionReport

::: director_ai.enterprise.moderation.ContentModerator

::: director_ai.enterprise.moderation.ContentModerationResult

::: director_ai.enterprise.rules_dsl.CustomRuleset

::: director_ai.enterprise.rules_dsl.CustomRule
