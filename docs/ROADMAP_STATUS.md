<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Director-Class AI - Roadmap Status Reconciliation -->

# Roadmap Status Reconciliation

Last reconciled: 2026-06-04

This file is the public status index for distinguishing active roadmap work
from completed, blocked, stale, and speculative planning material. The
canonical feature roadmap remains `ROADMAP.md`; this file records what is
still open after comparing the roadmap with current implementation notes,
validation records, and release documentation.

## Source Hierarchy

1. `ROADMAP.md` is the active public engineering roadmap.
2. `docs/ROADMAP_2026_2027.md` is the long-range strategic concept roadmap.
3. Internal planning and audit notes are evidence sources, not automatically
   active backlog. Any unchecked internal note must be reconciled here or in
   `ROADMAP.md` before it is treated as live work.
4. User-facing deployment, migration, and operations checklists are operator
   checklists, not repository TODO items.

## Active Actionable Items

| ID | Item | Status | Next evidence needed |
|---|---|---|---|
| R1 | Independent external security test focused on streaming paths and tenant isolation | Blocked on independent external reviewer | Third-party report or signed internal exception |
| R2 | Lite Scorer v2 training completion and real artefact evaluation | Open | Completed training artefacts, evaluation JSON, model card, and benchmark-claim eligibility check |
| R3 | FrontierFail broader sourced production-failure corpus | Open | More public incident cases with dated evidence, diversity validation, and no synthetic-only benchmark claims |
| R4 | PINT official upstream run and evidence | Open | Official upstream dataset execution, preserved outputs, and claim-guarded benchmark card |
| R5 | Hugging Face Space live deployment push | Open/manual | Published Space URL plus deployment smoke evidence |
| R6 | Polar storefront environment-specific live deployment smoke | Open/environment-specific | Live checkout, portal, webhook, and licence validation smoke records without committed secrets |
| R7 | AggreFact leaderboard submission | Submitted by email on 2026-06-02; awaiting maintainer response | Maintainer response, submission acknowledgement, or upstream pull-request URL |
| R8 | Unified Observability Dashboard (Core) | Partly implemented: OTEL token spans, Langfuse-shaped callback adapter, safety dashboard, and tenant-safe operations report with halt forensics, drift alerts, readiness controls, and compliance-export references are present; live external dashboard telemetry remains open | Archived staging telemetry, dashboard screenshots or hosted URL, and operator sign-off for the exported operations packet |
| R9 | Online KB evolution + provenance | Partly implemented: KB snapshot Merkle roots, HMAC provenance chain, source credibility, protected-claim conflict reports, and local provenance evidence packet are present; operator-owned online feedback loop remains open | Archived live feedback-loop run, signed lineage packet for a tenant KB, and operator sign-off that the exported evidence matches deployed facts |
| R10 | Conformal prediction routing + uncertainty | Partly implemented: `ConformalRoutingPolicy`, `ConformalRoutingDecision`, `ProductionGuard` interval compatibility methods, Rust/Python quantile fallback, dedicated routing tests, and local conformal-routing evidence packet are present | Representative domain calibration packet, archived deployment evidence, and operator sign-off for the live human-review or stronger-model escalation route |
| R11 | Agent trajectory simulation + rollback | Open | Monte-Carlo pre-execution simulation and safe undo hooks with adversarial stress testing evidence |
| R12 | Multi-modal + temporal consistency guard | Open | Vision-NLI streaming checks across frame sequences plus temporal consistency evidence |
| R13 | Privacy-preserving federated learning for signals | Open | MPC/DP aggregation evidence with poisoning-resilient federation and tenant isolation checks |
| R14 | Edge/mobile optimisation path | Open | Quantised NLI + Rust/WASM build and deployment evidence for low-latency local path |
| R15 | Continuous auto-redteam + defence genome loop | Open | Repeating adversarial generation and patch integration cycle with measurable coverage uplift |
| R16 | Formal + symbolic depth expansion | Open | New Lean/Z3/DPLL integration in production paths with regression evidence for math/code/numeric outputs |
| R17 | Deployment hardening + async/multi-tenant corrections | Partly implemented: production scaffold, authenticated monitoring, async ordering regression, tenant poisoning regression, and local sustained-load evidence runner added; external deployment telemetry still open | Sustained staging or production run with archived telemetry plus tenant poisoning evidence packet |

## Future And Strategic Items

| Area | Status | Handling |
|---|---|---|
| Shadow Director validation | Strategic | Keep in `docs/ROADMAP_2026_2027.md`; translate into actionable work only when a measurable validation packet is planned |
| Interventionist Director and intrinsic Backfire Kernel roadmap | Strategic | Keep in long-range roadmap; current shipped Backfire surfaces remain tracked in `ROADMAP.md` |
| Integrated Strange Loop and RLCF | Future research | Do not schedule as current engineering without a concrete research plan, datasets, and acceptance gates |
| Future differentiator programme phases | Partly implemented, partly needs reconciliation | Reconcile phase-by-phase against current public API docs before adding new work |
| Post-2035 cosmic-scale safety vision | Speculative | Not scheduled; do not move into active roadmap until prerequisites are real and testable |

## Completed Queues

The hardening and audit execution queues for dependency closure, model revision
pinning, coverage-gate correction, cyber-physical adapter contracts, router
failure visibility, knowledge deletion semantics, runtime assert removal,
privacy/RNG handling, SQL safety, optional-extras CI, licence reconciliation,
enterprise ingestion, public/internal hygiene, and the later May product-maturity
slices are closed or explicitly addressed in internal evidence records.

Internal roadmap and audit-note reconciliation was completed on 2026-05-22:
the active ignored TODO surface is `docs/internal/TODO_CONSOLIDATED.md`, and
older internal TODO, backlog, and roadmap notes were moved under
`docs/internal/archive/todo_roadmap_2026-05-22/`.

These items should not be re-opened from old unchecked checkboxes unless a fresh
current-code review finds a concrete regression.

## Stale Or Archival Material

Older unchecked internal checklists, old implementation-plan checkboxes, and
retracted benchmark-gap plans are not active backlog by themselves. They require
one of these outcomes during reconciliation:

- confirm the item is already implemented and record the evidence;
- migrate the item into `Active Actionable Items` with a concrete acceptance
  gate;
- mark the source as archival if its premise was superseded or retracted.

## Non-Backlog Checklist Classes

The following classes must not be counted as open repository TODO unless they
are explicitly promoted into `Active Actionable Items`:

- customer deployment checklists;
- migration checklists;
- examples that contain the word TODO as sample policy text;
- generated model/tokenizer outputs;
- historical audit notes whose findings were closed in later evidence records.
