<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Commercial license available -->
<!-- Copyright © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- Copyright © Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Director-Class AI - Guardrail Landscape -->

# Guardrail Landscape

Last reviewed: 2026-06-02.

LLM guardrails are not one product category. They cover several different risk
surfaces, and a system that performs well in one category may not protect
another category at all. Director-AI is positioned as a factual-coherence and
evidence-control layer, not as a replacement for moderation, access control, or
human review.

## Guardrail Categories

| Category | Main question | Typical tools | Director-AI fit |
|---|---|---|---|
| Content safety | Is the prompt or answer unsafe, abusive, or policy-violating? | Safety classifiers, provider moderation, human review queues | Complementary |
| Prompt-injection defence | Is the user or retrieved content trying to redirect the model? | Intent classifiers, prompt hardening, retrieval sanitisation, tool gating | Partial coverage through injection detection and policy gates |
| Grounded factuality | Is the answer supported by the supplied evidence? | NLI, claim checking, RAG faithfulness scoring, verifier models | Primary lane |
| Retrieval integrity | Did the retrieval layer expose the right evidence to the model? | Access control, document provenance, vector-store checks, citation audits | Supported through knowledge and evidence APIs |
| Structured-output control | Does the answer satisfy a schema or machine contract? | JSON schema validation, type checks, deterministic validators | Supported |
| Tool and agent control | Should the model be allowed to call this tool or hand off this trace? | Least-privilege tool policies, approval gates, trajectory review | Supported through guard control and trajectory surfaces |
| Audit and governance | Can an operator reconstruct what happened and why? | Telemetry, compliance reports, immutable evidence packs | Primary enterprise lane |

The practical production pattern is layered. A content-safety classifier should
not be expected to prove factual grounding. A factuality checker should not be
expected to replace data-loss prevention or tool permissioning. A prompt filter
should not be treated as a complete agent-security boundary.

## Where Director-AI Competes

Director-AI is most competitive when the protected workflow needs:

- document-grounded answers;
- private knowledge-base evidence;
- streamed output that may need to halt before completion;
- reusable guard policies across local, REST, SDK, and proxy integrations;
- inspectable scores, evidence, and audit events;
- customer-specific acceptance evidence before deployment.

It is less suitable as the only control for:

- generic user-generated-content moderation;
- image or video safety review;
- broad social-platform trust and safety;
- endpoint data-loss prevention;
- authorisation for high-risk tools.

Those controls can still sit beside Director-AI in a complete deployment.

## Current Public Benchmark Position

The live LLM-AggreFact leaderboard lists FactCG-DeBERTa-L at 75.6 percent
per-dataset mean balanced accuracy. Director-AI's default scorer uses the same
FactCG-DeBERTa-v3-Large model family and ships the surrounding guard,
streaming, evidence, API, and deployment surfaces.

The repository also carries a local reproducibility packet reporting 75.8
percent at threshold 0.46:

`benchmarks/results/aggrefact_yaxili96_FactCG-DeBERTa-v3-Large.json`

That packet is close to the public leaderboard row but is not yet a separate
official Director-AI leaderboard entry. A submission email was sent on
2026-06-02. Until the maintainers acknowledge it, the conservative public
statement is:

> Director-AI's default FactCG scorer is near the top public
> LLM-AggreFact factuality results while adding production guardrail,
> streaming, evidence, and deployment features that the leaderboard does not
> measure.

## Feature Comparison

| Capability | Content-safety classifiers | Programmable rail frameworks | Cloud managed guardrails | Director-AI |
|---|---|---|---|---|
| Harm and policy moderation | Primary | Depends on configured model | Primary | Complementary |
| Grounded factuality scoring | Usually limited | Depends on configured checks | Limited to service-specific features | Primary |
| Private RAG evidence links | Usually no | Possible with integration work | Platform-specific | Yes |
| Streaming contradiction halt | Usually no | Usually post-call or flow-level | Often complete-response only | Opt-in, contradiction-driven |
| Local/offline scoring path | Often yes for open models | Depends on model provider | No | Yes |
| API/proxy/SDK deployment | Varies | Yes | Managed service APIs | Yes |
| Customer-specific evidence packs | Usually no | Operator-built | Platform-specific | Yes |
| Deterministic structured checks | Usually no | Possible | Platform-specific | Yes |
| Tool and handoff guard surfaces | Usually no | Flow-dependent | Platform-specific | Yes |
| Commercial audit narrative | Usually moderation-centred | Operator-built | Platform-provided | Evidence-centred |

## Claim Boundaries

Keep these statements separate in external material:

1. Official LLM-AggreFact leaderboard rows.
2. Local reproducibility packets stored in this repository.
3. Tuned-threshold replay results, which are useful for deployment but not the
   same as the default leaderboard convention.
4. End-to-end HaluEval and RAG guardrail metrics, which measure a different
   operational surface.
5. Streaming false-halt diagnostics, which are not hallucination catch-rate
   claims.
6. Customer-specific acceptance results, which require customer data and
   customer approval criteria.
7. Contradiction-driven streaming-halt results, which prove the replacement
   streaming signal for contradiction-scoped claims but should not be reworded as
   coverage of every unsupported addition.

## Sources To Recheck Before External Claims

- LLM-AggreFact leaderboard: <https://llm-aggrefact.github.io/>
- OWASP Top 10 for LLM Applications: <https://owasp.org/www-project-top-10-for-large-language-model-applications/>
- NVIDIA NeMo Guardrails documentation: <https://docs.nvidia.com/nemo-guardrails/>
- Amazon Bedrock Guardrails automated reasoning documentation: <https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-automated-reasoning-checks.html>
- Llama Guard model documentation: <https://huggingface.co/meta-llama/Llama-Guard-4-12B>
