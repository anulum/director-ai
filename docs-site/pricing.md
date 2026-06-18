# Pricing

<div class="hero-subtitle" style="font-size: 1.2em; color: var(--md-default-fg-color--light); margin-bottom: 1.5em;">
The core is free — forever, in production, no strings.<br/>
Pay only when you put the advanced tier into production, or when you want support behind it.
</div>

!!! tip "The wedge"
    Director-AI is the only guardrail that severs the token stream **before** a
    hallucination finishes generating — a seatbelt that deploys before the crash,
    not an incident report after it. Detection is too late; you pay for prevention.

---

## :material-gift: Free — Apache-2.0 core

**CHF 0. Forever. Including production and closed-source.**

The open core is genuinely free, not a crippled trial:

- :material-check: Coherence guardrail engine + 5-tier scoring (rules → embeddings → NLI)
- :material-check: Token-level **streaming halt**
- :material-check: SDK guard, FastAPI middleware, REST/gRPC server
- :material-check: Prompt-injection detection, agent/MCP preflight guard
- :material-check: ONNX + Rust-accelerated compute paths
- :material-check: 6 framework integrations (LangChain, LlamaIndex, LangGraph, Haystack, CrewAI, OpenAI)
- :material-check: Self-hosted, air-gapped, no telemetry, no licence key

No source-disclosure obligation. Use it in a closed-source commercial product
without paying anyone. [Get started](quickstart.md){ .md-button }

---

## :material-rocket-launch: Free pilot — try the advanced tier on your data

!!! success "30 days · full advanced features · no credit card"
    - :material-database-search: Bring your own knowledge base — grounded scoring against *your* documents
    - :material-phone: Weekly 30-min call with the maintainer to review results
    - :material-check-decagram: Honest assessment — if it does not fit, we tell you
    - :material-lock-open: No lock-in, no card, no strings

    [Request a free pilot :material-flask:](mailto:director.class.ai@anulum.li?subject=Director-AI%20Free%20Pilot&body=Company:%0AUse%20case:%0AApprox.%20LLM%20calls/day:%0A){ .md-button .md-button--primary }

---

## :material-currency-usd: Commercial tiers — advanced tier in production

The **Advanced & Labs** capabilities are source-available under BUSL-1.1: free to
read and evaluate, free in non-production. A commercial licence unlocks them in
production and hosted/SaaS deployments, and adds support and SLAs.

!!! tip "Founding Member — 10 spots, permanent lock"
    Founding Members keep **40–50 % off, locked permanently**, plus direct access
    to the maintainer. Once 10 spots fill, prices move to the standard tier.

=== ":material-account: Indie — CHF 49/mo"

    **For:** solo developers, internal tools, side projects.

    - 1 production deployment of the advanced tier
    - Email support (48 h response)
    - Same-day updates
    - Every feature included

    **Founding Member: CHF 29/mo** (40 % off, locked)

    [Buy Indie :material-cart:](https://polar.sh/checkout/polar_c_gmuEUUV0VUIsmnm8ZzPJnmrYISFg5oXzualxI11Lgxn){ .md-button .md-button--primary }

=== ":material-account-group: Pro — CHF 199/mo"

    **For:** teams shipping LLM features to production.

    - Unlimited deployments
    - Slack priority support (4 h response)
    - 99.5 % SLA
    - Every feature included

    **Founding Member: CHF 99/mo** (50 % off, locked)

    [Buy Pro :material-cart:](https://polar.sh/checkout/polar_c_kLbfbCFJhyubFxzax8JNLw8WZU8T4IveHPlMo0kpxOZ){ .md-button .md-button--primary }

=== ":material-infinity: Perpetual — CHF 999"

    **For:** one-time purchase, no subscription.

    - Equivalent to the Indie tier
    - 12 months of updates included
    - No recurring payments, ever

    [Buy Perpetual :material-license:](https://polar.sh/checkout/polar_c_VW7ClxyB6axih6mu9NqWlh6OgeAVJl4GVDgFW0QePtZ){ .md-button .md-button--primary }

=== ":material-office-building: Enterprise — talk to us"

    **For:** regulated industries, multi-tenant SaaS, high volume.

    - Dedicated support engineer + private Slack/Teams channel
    - Custom SLA (99.9 %+)
    - On-prem / air-gapped deployment support
    - SOC 2 / HIPAA compliance documentation
    - Custom NLI fine-tuning on your domain data
    - Quarterly architecture reviews + roadmap input

    [Contact us :material-email:](mailto:director.class.ai@anulum.li?subject=Director-AI%20Enterprise){ .md-button .md-button--primary }

---

## :material-api: Usage-based API _(coming Q2 2026)_

For teams that prefer pay-per-use over fixed licensing:

| Volume / month | Per 1,000 checks | |
|---|:---:|---|
| First 5,000 | **Free** | no card needed |
| 5,001 – 50,000 | CHF 2.50 | ~CHF 0.0025 / check |
| 50,001 – 500,000 | CHF 1.50 | volume discount |
| 500,000+ | Custom | [contact us](mailto:director.class.ai@anulum.li) |

Self-hosted deployments use the licence tiers above.

---

## :material-compare: How we compare

The measured differences, from the committed
[competitor benchmark](benchmarks.md) — not marketing claims.

| | Director-AI | NeMo Guardrails | GuardrailsAI | SelfCheckGPT |
|---|:---:|:---:|:---:|:---:|
| **Token-level streaming halt** _(experimental, under calibration)_ | :material-flask: | :material-close: | :material-close: | :material-close: |
| **GPU NLI latency / pair** | **~0.9 ms** | LLM-bound | LLM-bound | 5–10 s |
| **Local guard benchmark p50 / p95** | **0.124 / 0.200 ms** | 0.818 / 1.418 ms config-load + LLM | 0.659 / 0.996 ms parse + LLM | not run in this harness |
| **Offline / local** | :material-check: | :material-close: | :material-close: | :material-close: |
| **Self-hosted core price** | **CHF 0** | free | free | free |
| **AggreFact balanced acc.** | 75.6 % (0.4 B) | N/A | N/A | N/A |

The local guard benchmark was produced on 2026-06-18 on host `aaarthuus`
(Ubuntu 24.04.4, Linux 6.17, ASRock H510 Pro BTC+, 11th Gen Intel Core
i5-11600K, 12 logical CPUs) and is stored in
`benchmarks/results/competitor_guard_latency.json` with
`non_isolated_local_regression` metadata. Prevention happens during generation,
on a 0.4 B model, on commodity hardware — the others score after the fact, or
need a second large LLM in the loop.

---

## :material-scale-balance: Is it worth it?

In a scientific, legal, medical, or financial pipeline, a single fabricated
number or citation can invalidate everything downstream. The cost of catching it
**before** it ships — at sub-millisecond NLI cost on a small model — is a rounding
error next to the cost of one bad answer reaching a customer, a regulator, or a
court. The commercial tiers price the advanced safeguards and the support that
stands behind them; the core that does the catching is free.

---

## :material-heart: Prefer to support directly?

Not ready for a licence but the project helps you? Donations directly fund
development and are genuinely appreciated:

- :material-credit-card: [PayPal](https://www.paypal.com/donate?hosted_button_id=4X5F6DNT934HY)
- :material-cellphone: [TWINT](https://go.twint.ch/1/e/tw?tw=acq.lJTAypb8SL2s8vPg7fL0ubi2C220ajOH0BEQn1aKfEJIiIakLpt8jlEv8XdQ9tCp.)
- :material-bank: **Bank transfer** — IBAN (CHF) `CH14 8080 8002 1898 7544 1` · IBAN (EUR) `CH66 8080 8002 8173 6061 8`
- :material-bitcoin: **Crypto** — BTC `bc1qg48gdmrjrjumn6fqltvt0cf0w6nvs0wggy37zd` · ETH `0xd9b07F617bEff4aC9CAdC2a13Dd631B1980905FF` · LTC `ltc1q886tmvtlnj86kmg2urd8f5td3lmfh32xtpdrut`
- :material-github: [GitHub Sponsors](https://github.com/sponsors/anulum)

---

## :material-frequently-asked-questions: Questions

??? question "What exactly is free?"
    The entire **Apache-2.0 core** — including the streaming halt, scoring, server,
    and integrations — for any use, including closed-source production. Only the
    **BUSL-1.1 advanced & labs** tier needs a commercial licence for production.

??? question "Do I need a licence to evaluate?"
    No. The core is Apache-2.0; the advanced tier is free to read and evaluate
    under BUSL-1.1. Buy a licence when an advanced capability ships to production.

??? question "What happens to BUSL files over time?"
    Each converts automatically to Apache-2.0 on its change date (the fourth
    anniversary). Today's advanced tier becomes tomorrow's open core.

For the full legal model, see [Licensing](licensing.md).

---

<div style="text-align: center; margin: 2em 0;">

**Ready?**

[Free pilot :material-flask:](mailto:director.class.ai@anulum.li?subject=Director-AI%20Free%20Pilot){ .md-button .md-button--primary }
&nbsp;&nbsp;
[Buy a licence :material-cart:](https://polar.sh/checkout/polar_c_gmuEUUV0VUIsmnm8ZzPJnmrYISFg5oXzualxI11Lgxn){ .md-button }
&nbsp;&nbsp;
[Talk to us :material-email:](mailto:director.class.ai@anulum.li?subject=Director-AI%20Inquiry){ .md-button }

<small>[director.class.ai@anulum.li](mailto:director.class.ai@anulum.li) · [anulum.li](https://www.anulum.li)</small>

</div>
