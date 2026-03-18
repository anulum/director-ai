# Licensing

Director-AI is dual-licensed. Choose the license that fits your use case.

## Open Source — AGPL-3.0

The entire Director-AI package (core, server, integrations, CLI) is licensed under
the [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.html).

**You can:**

- Use it freely for research, personal projects, and open-source software
- Self-host without restrictions
- Modify and redistribute under AGPL terms

**The copyleft obligation:**

If you modify Director-AI or include it in a networked service, you must release
your source code under AGPL-3.0. This applies to SaaS products that expose
Director-AI functionality to users over a network.

## Commercial License

A proprietary license removes the AGPL copyleft obligation. You need a commercial
license if:

- You embed Director-AI in a **proprietary product** and don't want to open-source it
- You run a **SaaS** that uses Director-AI without disclosing source code
- Your legal team requires a **non-copyleft license**

### Founding Member Program

!!! tip "Limited: 10 Founding Member spots"
    Founding Members get **permanent pricing lock** + direct access to the maintainer.
    Once 10 spots fill, prices move to standard tier. No time limit — lock your rate now.

**Every tier includes:** full source access, all features, all integrations.
The only differences are deployment scope, support level, and update cadence.

### Free Pilot (30 days)

Test Director-AI on your actual data before committing. No credit card, no lock-in.

- Full Pro features for 30 days
- Bring your own documents — we run grounded scoring against your KB
- Weekly 30-min call with the maintainer to review results
- Honest assessment: if Director-AI doesn't fit your use case, we tell you

<a href="mailto:protoscience@anulum.li?subject=Director-AI%20Free%20Pilot%20Request&body=Company:%0AUse%20case:%0AApprox.%20LLM%20call%20volume/day:%0A" class="cta-button">Request Free Pilot</a>

### Commercial License Tiers

| Tier | Monthly | Yearly | Target |
|------|---------|--------|--------|
| **Indie** | $49 | $490 | Solo developers, internal tools, side projects |
| **Pro** | $199 | $1,990 | Teams shipping LLM features to production |
| **Enterprise** | Custom | Custom | Regulated industries, multi-tenant SaaS, high volume |

**Founding Member pricing (first 10 customers):**

| Tier | Founding Monthly | Founding Yearly | Savings |
|------|-----------------|-----------------|---------|
| **Indie** | $29 | $290 | 40% off, locked permanently |
| **Pro** | $99 | $990 | 50% off, locked permanently |
| **Enterprise** | Custom | Custom | Priority roadmap input |

**Perpetual license:** $999 one-time (Indie equivalent, 12 months updates included).

### Feature Matrix

| Feature | AGPL (Free) | Indie | Pro | Enterprise |
|---------|:-----------:|:-----:|:---:|:----------:|
| Core scorer + streaming halt | Yes | Yes | Yes | Yes |
| NLI + RAG backends | Yes | Yes | Yes | Yes |
| VerifiedScorer (5-signal) | Yes | Yes | Yes | Yes |
| Framework integrations (6) | Yes | Yes | Yes | Yes |
| REST / gRPC server | Yes | Yes | Yes | Yes |
| ONNX export | Yes | Yes | Yes | Yes |
| Rust FFI kernel | Yes | Yes | Yes | Yes |
| CLI tools | Yes | Yes | Yes | Yes |
| AGPL source disclosure required | **Yes** | No | No | No |
| Multi-tenant (`TenantRouter`) | Yes | Yes | Yes | Yes |
| Policy engine + audit logging | Yes | Yes | Yes | Yes |
| Deployments | Unlimited | 1 prod | Unlimited | Unlimited |
| Support | GitHub Issues | Email (48h) | Slack (4h) | Dedicated engineer |
| Updates | Same day | Same day | Same day | Same day + preview |
| SLA | — | — | 99.5% | 99.9% |
| On-prem / air-gapped | Yes (AGPL) | Yes | Yes | Yes |
| SOC2 / HIPAA guidance | — | — | — | Yes |
| Custom model fine-tuning | — | — | — | Yes |

### Usage-Based Option (API)

For teams that prefer pay-per-use over fixed licensing:

| Volume | Price per 1,000 checks | Notes |
|--------|----------------------|-------|
| First 5,000/month | Free | No credit card needed |
| 5,001 — 50,000/month | $2.50 per 1K | ~$0.0025 per check |
| 50,001 — 500,000/month | $1.50 per 1K | Volume discount |
| 500,000+/month | Custom | Contact us |

Usage-based pricing requires the hosted API (coming Q2 2026).
Self-hosted deployments use the license tiers above.

### Why Not Just Use AGPL?

AGPL-3.0 gives you everything for free. The commercial license exists for one reason:
**if you can't or won't open-source your code.**

If your product is open-source, or your use is internal-only, AGPL is the right choice.
If you're building a proprietary SaaS that calls Director-AI, you need a commercial license.

We don't gate features behind the commercial license. We gate the *obligation to disclose source code*.

### Get Started

<a href="mailto:protoscience@anulum.li?subject=Director-AI%20Free%20Pilot%20Request" class="cta-button">Start Free Pilot</a>&nbsp;&nbsp;
<a href="mailto:protoscience@anulum.li?subject=Director-AI%20License%20Inquiry" class="cta-button cta-button--secondary">Buy License</a>

### Enterprise Includes

- Dedicated support engineer with private Slack/Teams channel
- Custom SLA (99.9%+ uptime guarantee)
- On-prem and air-gapped deployment support
- SOC2 / HIPAA compliance documentation and audit support
- Priority bug fixes and feature requests
- Quarterly architecture review calls
- Custom NLI model fine-tuning on your domain data
- Roadmap input: vote on features that matter to your use case

### Contact

- Email: [sales@anulum.li](mailto:sales@anulum.li)
- Web: [anulum.li/contact](https://www.anulum.li/contact.html)

## FAQ

**Can I use Director-AI in academic research?**

Yes. AGPL-3.0 permits research use. If you publish code that includes Director-AI,
release it under AGPL-3.0 or a compatible license.

**Can I use it in my SaaS product?**

Under AGPL-3.0, you must open-source the parts of your service that use Director-AI.
If you prefer not to, purchase a commercial license.

**Can I use the free version for internal tools?**

Yes, as long as you comply with AGPL-3.0 — internal use within an organization is
fine. The copyleft trigger is *distribution* or *network interaction* with external users.

**Do I need a license for evaluation or prototyping?**

No. Evaluate freely under AGPL-3.0. Purchase a commercial license when you ship
to production.

**What about contributions?**

Contributors retain copyright. By contributing, you agree to license your contribution
under AGPL-3.0 per the [CONTRIBUTING.md](https://github.com/anulum/director-ai/blob/main/CONTRIBUTING.md).
