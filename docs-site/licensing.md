<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- DIRECTOR-AI — docs-site/licensing.md -->

# Licensing

<div class="hero-subtitle" style="font-size: 1.15em; color: var(--md-default-fg-color--light); margin-bottom: 2em;">
The licence model. Open core under Apache-2.0 — free for any use, including production.<br/>
Advanced &amp; Labs under LicenseRef-BUSL-1.1-ANULUM-TDM — non-production use is subject to the permanent Use Limitation; production requires a commercial licence.<br/>
Looking for tiers and prices? See the <a href="pricing.md">Pricing page</a>.
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

## :material-tag-heart: Commercial Plans

Pro and Full are recurring annual subscriptions. The canonical prices,
checkout links, invoice options, and current terms live on the dedicated
**[Pricing page](pricing.md)**.

---

## :material-currency-usd: Tiers & prices

Commercial tiers, the usage-based API, and the side-by-side comparison live on
the dedicated **[Pricing page](pricing.md)** — one source of truth for prices.

In short: `director-ai-lite` is the free PyPI adoption package; the
**Apache-2.0 core** in `director-ai` is free for everyone, in production, with
no strings; the commercial tiers add production entitlement for the **LicenseRef-BUSL-1.1-ANULUM-TDM
Advanced & Labs** capabilities plus support; Director-Class AI is a separate
action-control and evidence product with scoped terms, not a Director-AI wheel.

| Route | Licence status | Support | Production use |
|---|---|---|---|
| **Director-Lite** | Free package; Apache-2.0 adoption surface | Community / self-support | Allowed, including closed-source products |
| **Director-AI Pro self-host** | Commercial licence for LicenseRef-BUSL-1.1-ANULUM-TDM advanced tier | Priority support, 99.5 % support target | Allowed for self-hosted production and internal platforms |
| **Director-AI Full self-host** | Commercial licence for LicenseRef-BUSL-1.1-ANULUM-TDM Advanced plus Labs/Research | Direct engineering support | Allowed for the complete self-hosted capability set |
| **Director-Class AI** | Separate BUSL action-control product with negotiated terms | Defined in the accepted order form | High-impact agent action control, approval routing, and evidence under the scoped deployment terms |

[See pricing :material-arrow-right:](pricing.md){ .md-button .md-button--primary }

Self-hosted deployments use the licence tiers above.

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

## :material-flask: Advanced & Labs — LicenseRef-BUSL-1.1-ANULUM-TDM

The advanced capabilities (under `core/<advanced>/`, `enterprise/`, `voice/`,
`ui/`, `experimental/`, `compliance/`, `agentic/`) are **source-available** under
[modified licence](https://github.com/anulum/director-ai/blob/main/LICENSES/LicenseRef-BUSL-1.1-ANULUM-TDM.txt).

Non-production use is subject to the complete modified terms. Text and data
mining and collecting, training, fine-tuning or evaluating machine-learning or
artificial-intelligence models require the Licensor's prior written permission.
This does not restrict running the software for its documented purpose.
The Use Limitation is permanent and survives the Change Date unless withdrawn
in writing. Original Change Date (the fourth anniversary of each version's first
public distribution) and Change License (`Apache-2.0`) parameters remain unchanged
pending the Licensor's later relicensing decision.

**The obligation:** production and hosted/SaaS use of the advanced tier requires
a commercial licence (the tiers above).

---

## :material-frequently-asked-questions: FAQ

??? question "Can I use Director-AI in academic research?"
    The Apache-2.0 core is free for any use. Advanced/Labs non-production research
    is subject to the permanent Use Limitation: text and data mining and
    machine-learning uses need prior written permission. Running the software
    for its documented purpose is excepted. A production deployment of the
    advanced tier needs a commercial licence.

??? question "Can I use it in my SaaS product?"
    The Apache-2.0 core: yes, freely, including closed-source SaaS. The LicenseRef-BUSL-1.1-ANULUM-TDM
    advanced tier in a hosted/production service requires a commercial licence.

??? question "Can I use the free version for internal tools?"
    The Apache-2.0 core: yes, without restriction. Advanced/Labs non-production
    internal use is subject to the permanent Use Limitation described above;
    production use needs a commercial licence.

??? question "Do I need a license for evaluation or prototyping?"
    The core is Apache-2.0. Advanced/Labs evaluation and prototyping are subject
    to the permanent Use Limitation described above. Text and data mining and
    machine-learning uses need prior written permission; running the software
    for its documented purpose is excepted. Production use needs a commercial
    licence.

??? question "What about contributions?"
    Contributors retain copyright. Contributions to the Apache-2.0 core are
    accepted under Apache-2.0 per [CONTRIBUTING.md](https://github.com/anulum/director-ai/blob/main/CONTRIBUTING.md).

---

<div style="text-align: center; margin: 2em 0;">

**Ready to get started?**

[Request Free Pilot :material-flask:](mailto:director.class.ai@anulum.li?subject=Director-AI%20Free%20Pilot%20Request&body=Company:%0AUse%20case:%0AApprox.%20LLM%20call%20volume/day:%0A){ .md-button .md-button--primary }
&nbsp;&nbsp;
[Request USD Checkout :material-cart:](mailto:director.class.ai@anulum.li?subject=Director-AI%20USD%20Checkout){ .md-button }
&nbsp;&nbsp;
[Contact Us :material-email:](mailto:director.class.ai@anulum.li?subject=Director-AI%20Inquiry){ .md-button }

<small>Email: [director.class.ai@anulum.li](mailto:director.class.ai@anulum.li) · Web: [anulum.li](https://www.anulum.li)</small>

</div>

The permanent limitation applies to the Advanced/Labs Licensed Work; the
Apache-2.0 core retains its existing terms. The original legal-review notice
and conversion fields remain unchanged; their presence does not end the
permanent limitation.
