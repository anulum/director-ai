<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Commercial license available -->
<!-- Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Director-Class AI - External Validation Report -->

# External Validation Report

Last reviewed: 2026-06-02.

This report records the public validation state for Director-AI factuality and
guardrail claims. It distinguishes official leaderboard evidence from local
reproducibility packets and deployment-oriented diagnostics.

## Summary

| Surface | Current status | Evidence |
|---|---|---|
| LLM-AggreFact public leaderboard | FactCG-DeBERTa-L is visible at 75.6 percent average balanced accuracy | <https://llm-aggrefact.github.io/> |
| Director-AI submission | Submitted by email on 2026-06-02; awaiting maintainer response | `docs/ROADMAP_STATUS.md` |
| Local global-threshold packet | 75.8 percent average balanced accuracy at threshold 0.46 | `benchmarks/results/aggrefact_yaxili96_FactCG-DeBERTa-v3-Large.json` |
| Tuned-threshold replay | Deployment-oriented replay result; not an official default leaderboard row | `benchmarks/results/aggrefact_sweep_lr_low.json` |
| Validation packet | Commands, required artefacts, and claim boundaries documented | `benchmarks/EXTERNAL_VALIDATION_PACKET.md` |

## LLM-AggreFact Position

LLM-AggreFact measures grounded factuality across 11 datasets and reports
per-dataset mean balanced accuracy. The live public board currently places the
FactCG-DeBERTa-L row close to the top of the visible table, behind larger or
specialised systems and ahead of several large general-purpose models.

Director-AI uses the same default scorer family and adds:

- SDK guard and direct scoring APIs;
- REST, gRPC, middleware, and proxy deployment surfaces;
- RAG evidence integration;
- token-level streaming halt;
- structured verification;
- injection detection;
- audit and compliance artefacts;
- Rust-accelerated compute paths where installed.

The leaderboard measures only the factuality scorer. It does not measure these
control-plane features.

## Local Result Packet

The local result packet reports:

| Field | Value |
|---|---|
| Benchmark | LLM-AggreFact |
| Model | `yaxili96/FactCG-DeBERTa-v3-Large` |
| Threshold | 0.46 |
| Total samples | 29,320 |
| Average balanced accuracy | 75.8 percent |

Primary file:

`benchmarks/results/aggrefact_yaxili96_FactCG-DeBERTa-v3-Large.json`

This packet should be treated as reproducibility evidence for the submitted
Director-AI row, not as an upstream acknowledgement.

## Per-Dataset Local Results

| Dataset | Balanced accuracy |
|---|---:|
| Reveal | 89.14 percent |
| Lfqa | 86.38 percent |
| RAGTruth | 82.18 percent |
| ClaimVerify | 78.13 percent |
| Wice | 76.94 percent |
| TofuEval-MeetB | 74.26 percent |
| AggreFact-XSum | 74.25 percent |
| FactCheck-GPT | 73.02 percent |
| TofuEval-MediaS | 71.85 percent |
| AggreFact-CNN | 68.82 percent |
| ExpertQA | 59.09 percent |

The weakness map is important for deployment planning. ExpertQA and
class-imbalanced summarisation are the main risk areas for the default
threshold.

## External Claim Language

Use this wording until the upstream submission is acknowledged:

> Director-AI's default FactCG scorer is near the top public LLM-AggreFact
> factuality results and ships with production guardrail surfaces for
> streaming halt, RAG evidence, structured verification, APIs, and audit
> evidence. A Director-AI result packet has been submitted for leaderboard
> review.

Do not state that Director-AI has a separate official leaderboard row until the
maintainers add or acknowledge it.

## Non-Comparable Metrics

These results are valuable but should not be merged into the LLM-AggreFact
claim:

- HaluEval end-to-end catch, precision, false-positive rate, and F1;
- local judge benchmark results;
- streaming false-halt diagnostics;
- prompt-injection replication results;
- customer-specific acceptance tests;
- tuned-threshold replay results.

Each belongs to a different operational question and should keep its own
dataset, metric, threshold, and hardware record.

## Next Validation Actions

1. Preserve the submitted packet unchanged unless the maintainer requests a
   different format.
2. If there is no response after the follow-up window, open an upstream pull
   request with the same packet and command record.
3. Add the upstream acknowledgement or pull-request URL to
   `docs/ROADMAP_STATUS.md`.
4. Re-run the external packet before the next public release if the scorer,
   threshold, model revision, or dataset loader changes.
