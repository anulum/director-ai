<!--
SPDX-License-Identifier: Apache-2.0
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-Class AI — Applications and market map
-->

# Applications And Market Map

Director-AI is a factual-coherence control plane for LLM applications. It sits
between generated output and consequence: a user-visible answer, a streamed
token path, a tool call, an agent handoff, a stored record, or an audit event.

Use this page when you need to explain the product quickly to a buyer,
developer, evaluator, operator, or partner.

## What The Software Is

Director-AI is not a chatbot, prompt template, content-moderation list, or
model provider. It is a guardrail runtime that checks whether generated text is
grounded enough to proceed.

The runtime combines:

- governed facts from inline rules, document ingestion, or vector retrieval;
- configurable scorers, including rules, embeddings, NLI, and structured
  verification;
- response-level factual-coherence scoring, with opt-in contradiction-driven
  streaming halt for completed streamed claims;
- SDK, framework, REST, gRPC, inference-server, and voice integration surfaces;
- tenant-safe evidence, audit, metrics, and forensics records.

The smallest useful path is `director-ai-lite`, which exposes a three-line
guard facade for free-tier adoption. The full `director-ai` package is
the open-core runtime. Director-Class AI is the commercial implementation and
evidence programme around governed customer deployments; it is not a separate
wheel.

## Who Needs It

| Audience | Primary problem | Director-AI value |
|---|---|---|
| Product owner | A wrong answer damages trust or triggers rework | Define which answers need fact gates and what evidence a pilot must collect |
| Application developer | LLM output reaches users before review | Add `guard()`, SDK middleware, or a REST proxy at the output boundary |
| RAG engineer | Retrieval context is stale, noisy, or incomplete | Score answers against governed facts and retain retrieval evidence |
| Runtime/platform team | Multiple apps need one shared control layer | Deploy REST/gRPC, auth, metrics, audit, and rollout controls once |
| Evaluation lead | Model, prompt, or KB changes need regression gates | Run batch scoring, thresholds, false-positive review, and benchmark cards |
| Governance/security team | Incidents need reviewable evidence without raw data exposure | Use tenant-safe events, compliance reports, and guardrail forensics |

## Application Lanes

Pick one lane for the first pilot. Each lane has a different success signal.

| Lane | Protected workflow | First proof |
|---|---|---|
| Customer support | Refunds, warranty, policy, entitlement, and account answers | One known-good answer approved and one unsupported policy answer rejected |
| Enterprise knowledge assistant | Private-document answers and summaries | Verdict includes retrieved chunks and a traceable rejection reason |
| Regulated review | Medical, legal, finance, or research drafts | Unsupported claims route to human review with evidence and threshold rationale |
| Streaming assistant | User-visible token stream | Contradictory completed claim halts before the stream continues |
| Agent workflow | Tool output, chain step, or handoff | Unsafe step rejects or routes before downstream action |
| Evaluation pipeline | Prompt/response datasets and model updates | Batch report shows threshold, false positives, and false negatives |
| Platform deployment | Shared guardrail service | Auth, metrics, logs, rollback, and runbook evidence are visible |

## Market Value

The market value is control over factual risk. Director-AI can reduce:

- unsupported customer-facing claims;
- manual review load for routine factual checks;
- regressions from model, prompt, or knowledge-base changes;
- incidents caused by hallucinated streamed output;
- duplicated guardrail work across LLM providers and application teams.

It can increase:

- buyer confidence that LLM output remains tied to governed facts;
- operational visibility into why an answer was allowed, rejected, halted, or
  routed;
- reuse of one guard policy across SDK, REST, gRPC, voice, agent, and
  inference-server deployments;
- quality of procurement, security, and compliance evidence.

Director-AI does not remove the need for domain experts, governance, access
control, or legal review. It gives those functions a concrete enforcement point
inside LLM output flow.

## What Ships Publicly

| Surface | Public package/docs | Commercial extension boundary |
|---|---|---|
| Director-Lite | `director-ai-lite`, three-line guard facade, free onboarding | None required for free-tier use |
| Director-AI | `director-ai`, open-core SDK, scorers, APIs, integrations, evidence packet, docs | Paid Pro use for Advanced & Labs production deployment |
| Director-Class AI | Promoted in docs and PyPI positioning | Customer-specific deployment, sector packs, evidence reviews, tuning, and SLA |

This table is the public-vs-commercial *boundary*; for the full commercial tier
ladder and prices (Director-Lite, Director-AI, Pro, Full, Director-Class AI) see
**[Pricing](../pricing.md)**.

The public repository contains the core software and general evidence surfaces.
Customer-specific sector packs, tuning datasets, deployment recipes, acceptance
criteria, and performance claims must be validated against the customer's own
governed data before they are used commercially.

## First Evidence Packet

Before any serious pilot discussion, produce a small evidence packet:

```bash
pip install "director-ai[nli]"
director-ai evidence --emit evidence/
director-ai verify-evidence evidence/
```

That packet should prove:

1. a governed fact loaded successfully;
2. a grounded answer passed;
3. a hallucinated or contradictory answer failed;
4. the decision record has a digest and can be verified;
5. the operator can explain what happened without exposing secrets.

## Documentation Path

| Need | Read next |
|---|---|
| Product and tier boundary | [Product Overview](product-overview.md) |
| Buyer value and budget language | [Market Value and Positioning](market-value-and-positioning.md) |
| First local run | [Quickstart](../quickstart.md) |
| Guided notebook path | [Notebook Gallery](../notebook-gallery.md) |
| API selection | [API Reference](../api/index.md) |
| Pilot checklist | [Evaluation Onboarding](onboarding.md) |
| Production operation | [Production Guide](../deployment/production.md) |
