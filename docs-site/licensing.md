# Licensing & Pricing

<div class="hero-subtitle" style="font-size: 1.15em; color: var(--md-default-fg-color--light); margin-bottom: 2em;">
Open core under Apache-2.0 — free for any use, including production.<br/>
Advanced &amp; Labs under BUSL-1.1 — free to evaluate; a commercial licence unlocks production.
</div>

---

## :material-rocket-launch: Free Pilot — Try Before You Buy

!!! success "30 days. Full Pro features. No credit card."

    Test Director-AI on **your actual documents** before committing.

    - :material-database-search: Bring your own knowledge base — we run grounded scoring against your data
    - :material-phone: Weekly 30-min call with the maintainer to review results
    - :material-check-decagram: Honest assessment: if it doesn't fit your use case, we tell you
    - :material-lock-open: No lock-in. No credit card. No strings.

[Request Free Pilot :material-flask:](mailto:director.class.ai@anulum.li?subject=Director-AI%20Free%20Pilot%20Request&body=Company:%0AUse%20case:%0AApprox.%20LLM%20call%20volume/day:%0A){ .md-button .md-button--primary }

---

## :material-tag-heart: Founding Member Program

!!! tip "Limited: 10 spots remaining"
    Founding Members get **permanent pricing lock** at 40–50% off standard rates, plus direct access to the maintainer. Once 10 spots fill, prices move to standard tier. No deadline — lock your rate today.

---

## :material-currency-usd: Commercial License Tiers

=== ":material-account: Indie — CHF 49/mo"

    **For:** Solo developers, internal tools, side projects.

    - 1 production deployment
    - Email support (48h response)
    - Same-day updates
    - All features included

    **Founding Member: CHF 29/mo** (40% off, locked permanently)

    [Buy Indie :material-cart:](https://polar.sh/checkout/polar_c_gmuEUUV0VUIsmnm8ZzPJnmrYISFg5oXzualxI11Lgxn){ .md-button .md-button--primary }

=== ":material-account-group: Pro — CHF 199/mo"

    **For:** Teams shipping LLM features to production.

    - Unlimited deployments
    - Slack priority support (4h response)
    - 99.5% SLA
    - All features included

    **Founding Member: CHF 99/mo** (50% off, locked permanently)

    [Buy Pro :material-cart:](https://polar.sh/checkout/polar_c_kLbfbCFJhyubFxzax8JNLw8WZU8T4IveHPlMo0kpxOZ){ .md-button .md-button--primary }

=== ":material-infinity: Perpetual — CHF 999"

    **For:** One-time purchase, no subscription.

    - Equivalent to Indie tier
    - 12 months of updates included
    - No recurring payments, ever
    - All features included

    [Buy Perpetual :material-license:](https://polar.sh/checkout/polar_c_VW7ClxyB6axih6mu9NqWlh6OgeAVJl4GVDgFW0QePtZ){ .md-button .md-button--primary }

=== ":material-office-building: Enterprise"

    **For:** Regulated industries, multi-tenant SaaS, high volume.

    - Dedicated support engineer + private Slack/Teams channel
    - Custom SLA (99.9%+ uptime)
    - On-prem / air-gapped deployment support
    - SOC2 / HIPAA compliance documentation
    - Custom NLI model fine-tuning on your domain data
    - Quarterly architecture review calls
    - Roadmap input: vote on features that matter to you

    [Contact Us :material-email:](mailto:director.class.ai@anulum.li?subject=Director-AI%20Enterprise%20Inquiry){ .md-button .md-button--primary }

---

## :material-compare: What's the Difference?

The **Apache-2.0 core** is free for everyone, in production, with no strings. The
commercial tiers add a production licence for the **BUSL-1.1 Advanced & Labs**
capabilities, plus support and SLAs.

| | :material-open-source-initiative: Free | :material-account: Indie | :material-account-group: Pro | :material-office-building: Enterprise |
|---|:---:|:---:|:---:|:---:|
| **Core scorer + streaming halt** (Apache-2.0) | :material-check: | :material-check: | :material-check: | :material-check: |
| **NLI + RAG backends** (Apache-2.0) | :material-check: | :material-check: | :material-check: | :material-check: |
| **6 framework integrations** (Apache-2.0) | :material-check: | :material-check: | :material-check: | :material-check: |
| **REST / gRPC server** (Apache-2.0) | :material-check: | :material-check: | :material-check: | :material-check: |
| **ONNX + Rust kernel** (Apache-2.0) | :material-check: | :material-check: | :material-check: | :material-check: |
| **Core in production / closed-source** | :material-check: | :material-check: | :material-check: | :material-check: |
| | | | | |
| **Advanced & Labs** (BUSL-1.1) | Eval only | :material-check: | :material-check: | :material-check: |
| **Advanced & Labs in production** | — | :material-check: | :material-check: | :material-check: |
| **Deployments** | Core: unlimited | 1 prod | Unlimited | Unlimited |
| **Support** | GitHub Issues | Email (48h) | Slack (4h) | Dedicated engineer |
| **SLA** | — | — | 99.5% | 99.9% |
| **Custom fine-tuning** | — | — | — | :material-check: |

!!! quote "Our philosophy"
    The core guardrail is genuinely open: Apache-2.0, free in production, no
    disclosure obligation. We monetise the *advanced and labs* tier — the deeper
    capabilities under BUSL-1.1 — and the support around it. Evaluate everything
    for free; pay only when an advanced capability goes to production.

---

## :material-api: Usage-Based API (Coming Q2 2026)

For teams that prefer pay-per-use over fixed licensing:

| Volume | Per 1,000 checks | |
|--------|:---:|---|
| First 5,000/month | **Free** | No credit card needed |
| 5,001 — 50,000 | CHF 2.50 | ~CHF 0.0025 per check |
| 50,001 — 500,000 | CHF 1.50 | Volume discount |
| 500,000+ | Custom | [Contact us](mailto:director.class.ai@anulum.li) |

Self-hosted deployments use the license tiers above.

---

## :material-shield-key: Polar Deployment Wiring

Director-AI validates commercial self-hosted licenses through Polar when
`DIRECTOR_LICENSE_KEY` and `DIRECTOR_AI_POLAR_ORG_ID` are configured. The
runtime never requires storing a raw organization access token for public
customer-portal validation. Add `DIRECTOR_AI_POLAR_ACCESS_TOKEN` only on
trusted server deployments that need server-side activation, deactivation,
customer portal sessions, or webhook reconciliation.

Required production checks:

```bash
director-ai license polar-env
director-ai license polar-env --json
```

Use the JSON form in deployment smoke checks or CI jobs; it emits only
`ready`, `errors`, and `warnings` so raw licence keys and Polar access tokens
are not printed.

Core variables:

| Variable | Purpose |
|---|---|
| `DIRECTOR_LICENSE_KEY` | Customer license key presented by the deployment. |
| `DIRECTOR_AI_POLAR_ORG_ID` | Polar organization UUID used for license validation. |
| `DIRECTOR_AI_POLAR_ACTIVATION_ID` | Optional activation binding for deployments with activation limits. |
| `DIRECTOR_AI_POLAR_INCREMENT_USAGE` | Optional integer usage increment sent during validation. |
| `DIRECTOR_AI_POLAR_CONDITIONS` | Optional JSON object for Polar validation conditions, such as major version or edition. |
| `DIRECTOR_AI_POLAR_ACCESS_TOKEN` | Server-only organization access token for server API calls. Never expose it in client code or public logs. |
| `DIRECTOR_AI_POLAR_WEBHOOK_SECRET` | Base64 Standard Webhooks secret for validating Polar webhook deliveries. |

Supported operational surfaces:

- License validation through Polar customer-portal or server endpoints.
- Activation and deactivation helpers for deployments using activation limits.
- Usage tracking via Polar validation `increment_usage`.
- Customer portal session creation from a server-side customer id or external customer id.
- Standard Webhooks HMAC verification over the raw body plus `webhook-id` and `webhook-timestamp` headers.

Webhook handlers must verify the raw request body before parsing JSON, reject
stale timestamps, and use `webhook-id` as the idempotency key in the deployment's
queue or database. Return a 2xx response only after the event has been accepted
for durable processing.

---

## :material-open-source-initiative: Open Core — Apache-2.0

The Director-AI **core** — the guardrail engine, 5-tier scoring (rules →
embeddings → NLI), SDK guard, FastAPI middleware, REST/gRPC server, injection
detection, streaming halt, and the agent/MCP preflight guard — is licensed under
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0).

**You can:** use it freely for anything, including production and closed-source
products. Self-host without restrictions. Modify and redistribute. No
source-disclosure obligation.

## :material-flask: Advanced & Labs — BUSL-1.1

The advanced capabilities (under `core/<advanced>/`, `enterprise/`, `voice/`,
`ui/`, `experimental/`, `compliance/`, `agentic/`) are **source-available** under
[BUSL-1.1](https://mariadb.com/bsl11/).

**You can:** read the source, evaluate, and use it for free in any
**non-production** setting. Each file automatically converts to Apache-2.0 on its
change date.

**The obligation:** production and hosted/SaaS use of the advanced tier requires
a commercial licence (the tiers above).

---

## :material-frequently-asked-questions: FAQ

??? question "Can I use Director-AI in academic research?"
    Yes. The Apache-2.0 core is free for any use. The BUSL-1.1 advanced tier is
    free for non-production research and evaluation; a published deployment that
    serves external users on the advanced tier would need a commercial licence.

??? question "Can I use it in my SaaS product?"
    The Apache-2.0 core: yes, freely, including closed-source SaaS. The BUSL-1.1
    advanced tier in a hosted/production service requires a commercial licence.

??? question "Can I use the free version for internal tools?"
    The Apache-2.0 core: yes, without restriction. The BUSL-1.1 advanced tier is
    free for non-production internal use; production use needs a commercial licence.

??? question "Do I need a license for evaluation or prototyping?"
    No. The core is Apache-2.0, and the advanced tier is free to evaluate under
    BUSL-1.1. Buy a commercial licence when an advanced capability ships to production.

??? question "What about contributions?"
    Contributors retain copyright. Contributions to the Apache-2.0 core are
    accepted under Apache-2.0 per [CONTRIBUTING.md](https://github.com/anulum/director-ai/blob/main/CONTRIBUTING.md).

---

<div style="text-align: center; margin: 2em 0;">

**Ready to get started?**

[Request Free Pilot :material-flask:](mailto:director.class.ai@anulum.li?subject=Director-AI%20Free%20Pilot%20Request&body=Company:%0AUse%20case:%0AApprox.%20LLM%20call%20volume/day:%0A){ .md-button .md-button--primary }
&nbsp;&nbsp;
[Buy License :material-cart:](https://polar.sh/checkout/polar_c_gmuEUUV0VUIsmnm8ZzPJnmrYISFg5oXzualxI11Lgxn){ .md-button }
&nbsp;&nbsp;
[Contact Us :material-email:](mailto:director.class.ai@anulum.li?subject=Director-AI%20Inquiry){ .md-button }

<small>Email: [director.class.ai@anulum.li](mailto:director.class.ai@anulum.li) · Web: [anulum.li](https://www.anulum.li)</small>

</div>
