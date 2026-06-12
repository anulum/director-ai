# How Director-AI compares

Director-AI is unusual: it is a **response-level runtime guardrail** *and* a
**CI eval gate** in one tool, with its hallucination accuracy benchmarked on
LLM-AggreFact. It also ships an **experimental token-level streaming halt** that
re-scores output during generation — a mechanism we deposited early, but which on
our own false-halt benchmark cannot yet separate hallucinated from correct
streaming text without a high false-halt rate. It is opt-in and under
calibration; treat the rows below for it as experimental, not a production claim.

!!! note "About this page"
    Competitor entries are compiled from public vendor materials and third-party
    reviews (as of 2026-06) and are **indicative, not independently benchmarked
    by us**. Director-AI entries are from this repository. Corrections welcome.

## What's free vs commercial

Director-AI is **open core**. The table below is what ships in the free
Apache-2.0 package vs the commercial BUSL-1.1 advanced tier.

| Capability | Free (Apache-2.0 core) | Advanced (BUSL-1.1) |
|---|:---:|:---:|
| Token-level streaming halt (experimental) | 🧪 | |
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
| **Token-level streaming halt** _(experimental)_ | 🧪 | post-hoc | — | — | — | token-prefix |
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
| Token-level streaming halt (experimental) | 🧪 | — | — | — | — | — | — |
| CI eval gate | ✅ | ✅ | partial | partial | ✅ | partial | partial |
| Automated red-teaming | ✅ | — | partial | — | ✅ | ✅ | partial |
| Observability / tracing | ✅ | ✅ | partial | ✅ | partial | partial | partial |
| Hallucination / RAG eval | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | partial |
| Self-host / OSS | ✅ | partial | partial | ✅ | ✅ | ✅ | ✅ |

## Adversarial-benchmark numbers (HarmBench + JailbreakBench)

Measured by `benchmarks/jailbreak_detection.py --with-model` over the public
[HarmBench](https://github.com/centerforaisafety/HarmBench) (400 behaviours),
[JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) (100 harmful +
500 benign incl. an Alpaca sample), and the **real published attack artifacts**
(PAIR, GCG, DSN, random-search; the prompts an independent tester would use).
The guard measured is `LayeredPromptGuard`: the pattern `InputSanitizer` plus a
model stage (ProtectAI `deberta-v3-base-prompt-injection-v2`, Apache-2.0,
enabled with `prompt_guard_model_enabled`). A prompt is blocked if either fires.

We report every family separately — including the ones we are weak on — so the
numbers reproduce under an independent re-run rather than flattering the product.

| Attack family | What it is | Detection |
|---|---|---|
| Canonical templates | prefix / refusal-suppression / DAN / AIM / base64 | **100.0%** (2500/2500) |
| **Real artifacts (aggregate)** | published PAIR/GCG/DSN/random-search prompts | **74.9%** (286/382) |
| └ random-search | optimised black-box | 100% |
| └ DSN | | 89% |
| └ GCG | gradient-optimised suffix | 64% |
| └ PAIR | LLM-crafted persuasion | 40% |
| Held-out evasion (aggregate) | families never used to tune a pattern | 57.2% (1145/2000) |
| └ many-shot / leetspeak | | 100% / 87% |
| └ ROT13 / payload-split | **weak spots, disclosed** | 31% / 11% |
| Raw harmful goals (baseline) | plain harmful requests — not injections | 0.0% (0/500) |
| Toxicity moderation — raw harmful | detoxify; targets toxic *language*, not intent | 2.0% |
| **False positives — benign** | 500 benign (JailbreakBench + Alpaca) | **0.2%** (1/500) |

Without the model stage the pattern guard alone scores 0% on every real artifact
and held-out family — patterns only catch the vocabulary they were written for.
The model stage is what makes the guard hold up against attacks it has not seen,
at a 0.2% benign false-positive rate. ROT13 and payload-splitting remain weak and
are tracked as open work; we publish them rather than rounding the aggregate up.

This stage is **optional, off by default, and still being improved.** The default
classifier is chosen for a near-zero benign false-positive rate: other public
models reach higher recall only by flagging 17-58% of *legitimate* traffic, which
is unusable. A higher-recall, low-FPR option (Meta Prompt Guard 2) is gated and
non-permissive but configurable as an opt-in. See the
[prompt-injection guard guide](guide/prompt-guard.md) for the model bake-off and
roadmap.

## Where we're honest about the roadmap

We publish what we don't have yet, too: a **cloud SaaS** offering and
**long-context moderation beyond the 512-token model window** are on the roadmap.
Everything in the tables above is in the repository today.
