# How Director-AI compares

Director-AI is unusual: it is a **real-time runtime guardrail** *and* a **CI eval
gate** in one tool. Most alternatives are one or the other. Its defining feature
— a **token-level streaming halt** that stops output *before* a hallucination
finishes generating — is, to our knowledge, not offered by any of the tools
below.

!!! note "About this page"
    Competitor entries are compiled from public vendor materials and third-party
    reviews (as of 2026-06) and are **indicative, not independently benchmarked
    by us**. Director-AI entries are from this repository. Corrections welcome.

## What's free vs commercial

Director-AI is **open core**. The table below is what ships in the free
Apache-2.0 package vs the commercial BUSL-1.1 advanced tier.

| Capability | Free (Apache-2.0 core) | Advanced (BUSL-1.1) |
|---|:---:|:---:|
| Token-level streaming halt | ✅ | |
| 5-tier scoring (rules → embeddings → NLI) | ✅ | |
| RAG grounding + vector store | ✅ | |
| Prompt-injection detection (regex + NLI) | ✅ | |
| PII + toxicity moderation | ✅ | |
| Unified firewall decision | ✅ | |
| Rate limiting, multi-tenant isolation | ✅ | |
| Tamper-evident audit chain + evidence packets | ✅ | |
| CI quality gate + GitHub Action | ✅ | |
| REST / gRPC server, Rust acceleration | ✅ | |
| Reasoning-chain + structured-output verification | ✅ | |
| Streaming repair (corrective halt) | | ✅ |
| Multimodal guard (image / audio / video) | | ✅ |
| Temporal-consistency, swarm coherence | | ✅ |
| Voice guardrail, config UI | | ✅ |
| Customer model factory, threat intel | | ✅ |

The free core is free for any use, including production and closed-source. The
advanced tier is source-available and free to evaluate; production use needs a
commercial licence. See [Pricing](pricing.md) and [Licensing](licensing.md).

## vs real-time guardrails

| | Director-AI | Galileo | GA Guard | NeMo Guardrails | Llama Guard 4 | Future AGI |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Token-level streaming halt** | ✅ | post-hoc | — | — | — | token-prefix |
| Self-host / open weights | ✅ | — | partial | ✅ | ✅ | hosted |
| Offline / air-gapped | ✅ | — | partial | partial | ✅ | — |
| Injection (semantic NLI) | ✅ | ✅ | ✅ | partial | ✅ | ✅ |
| PII / toxicity | ✅ | ✅ | ✅ | partial | ✅ | ✅ |
| Multimodal | ✅ | ✅ | ✅ | — | partial | ✅ |
| Tamper-evident audit | ✅ | partial | partial | — | — | partial |
| Multi-tenant (OSS tier) | ✅ | partial | partial | — | — | partial |
| Swarm / multi-agent guarding | ✅ | partial | — | — | — | — |
| Cloud SaaS | roadmap | ✅ | ✅ | ✅ | n/a | ✅ |
| Licence | Apache-2.0 + BUSL-1.1 | proprietary | proprietary | Apache-2.0 | open weights | proprietary |

## vs eval / observability / red-teaming tools

These are mostly evaluation, observability, or testing tools rather than runtime
guards. Director-AI spans both — runtime guard **and** CI eval.

| | Director-AI | Braintrust | Patronus | Arize | Promptfoo | Giskard | Guardrails AI |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Real-time runtime guard | ✅ | — | partial | — | — | — | ✅ |
| Token-level streaming halt | ✅ | — | — | — | — | — | — |
| CI eval gate | ✅ | ✅ | partial | partial | ✅ | partial | partial |
| Automated red-teaming | ✅ | — | partial | — | ✅ | ✅ | partial |
| Observability / tracing | ✅ | ✅ | partial | ✅ | partial | partial | partial |
| Hallucination / RAG eval | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | partial |
| Self-host / OSS | ✅ | partial | partial | ✅ | ✅ | ✅ | ✅ |

## Adversarial-benchmark numbers (HarmBench + JailbreakBench)

Measured by `benchmarks/jailbreak_detection.py` over the public
[HarmBench](https://github.com/centerforaisafety/HarmBench) (400 behaviours) and
[JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) (100 harmful +
100 benign) sets. Director-AI is a hallucination + prompt-injection guardrail,
not a harmful-content refusal classifier, so each layer is reported separately,
measured on the input it is designed for — no single headline number that would
flatter or mislead.

| Layer | What it measures | Result |
|---|---|---|
| Input guard — raw harmful goals | injection patterns in plain harmful requests | 0.0% (0/500) — a plain harmful request is not an injection; this is the honest scope baseline |
| Input guard — jailbreak-wrapped | same goals wrapped in five canonical jailbreak templates | **59.9%** (1498/2500) |
| └ prefix injection / AIM / base64 | | 100% / 100% / 99.6% |
| └ refusal suppression / DAN | | 0% / 0% — pattern families not yet covered |
| Toxicity moderation — raw harmful | detoxify on the raw goals | 2.0% (10/500) — detoxify targets toxic *language*, not harmful *intent*, so lexically-neutral instructions are true misses for this layer |
| False positives — benign (input guard) | 100 benign JailbreakBench requests | **0.0%** (0/100) |
| False positives — benign (toxicity) | 100 benign JailbreakBench requests | 1.0% (1/100) |

The input guard catches three of the five standard jailbreak families outright
at a 0% benign false-positive rate; refusal-suppression and DAN-style role-play
are not yet covered and are tracked as pattern-coverage work. We publish the
gaps rather than a rounded-up aggregate.

## Where we're honest about the roadmap

We publish what we don't have yet, too: a **cloud SaaS** offering and
**long-context moderation beyond the 512-token model window** are on the roadmap.
Everything in the tables above is in the repository today.
