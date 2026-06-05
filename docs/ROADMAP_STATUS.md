<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Director-Class AI - Roadmap Status Reconciliation -->

# Roadmap Status Reconciliation

Last reconciled: 2026-06-05

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
| R8 | Unified Observability Dashboard (Core) | Partly implemented: OTEL token spans, Langfuse-shaped callback adapter, safety dashboard, tenant-safe operations report with halt forensics, drift alerts, readiness controls, compliance-export references, and Customer Model Factory release-gate observability evidence contract are present; live external dashboard telemetry remains open | Archived staging telemetry, dashboard screenshots or hosted URL, and operator sign-off for the exported operations packet |
| R9 | Online KB evolution + provenance | Partly implemented: KB snapshot Merkle roots, HMAC provenance chain, source credibility, protected-claim conflict reports, local provenance evidence packet, and Customer Model Factory release-gate provenance lineage evidence contract are present; operator-owned online feedback loop remains open | Archived live feedback-loop run, signed lineage packet for a tenant KB, and operator sign-off that the exported evidence matches deployed facts |
| R10 | Conformal prediction routing + uncertainty | Partly implemented: `ConformalRoutingPolicy`, `ConformalRoutingDecision`, `ProductionGuard` interval compatibility methods, Rust/Python quantile fallback, dedicated routing tests, local conformal-routing evidence packet, and Customer Model Factory release-gate conformal routing evidence contract are present | Representative domain calibration packet, archived deployment evidence, and operator sign-off for the live human-review or stronger-model escalation route |
| R11 | Agent trajectory simulation + rollback | Partly implemented: Monte-Carlo preflight, predictive pre-halt steering, tenant-safe rollback handles, idempotent rollback hook execution, dedicated rollback tests, local trajectory rollback evidence packet, and Customer Model Factory release-gate trajectory rollback evidence contract are present | Operator-owned live undo backend, adversarial trajectory stress testing against deployment traffic, and sign-off that rollback evidence is attached to incident/change-management records |
| R12 | Multi-modal + temporal consistency guard | Partly implemented: opt-in multimodal adapter, image/audio/video guard decisions, caption and metadata grounding references, temporal frame consistency halts, Rust/Python hash-bag fallback, dedicated tests, local multimodal temporal evidence packet, and Customer Model Factory release-gate multimodal temporal evidence contract are present | External Vision-NLI or equivalent benchmark evidence, real video/frame model validation, and operator sign-off for deployment-specific modality coverage |
| R13 | Privacy-preserving federated learning for signals | Partly implemented: DP-noised safety-signal aggregation, tenant/category contribution caps, minimum cohort gate, additive secret sharing, Rust/Python accountant and aggregator fallback, dedicated tests, local federated privacy evidence packet, and Customer Model Factory release-gate federated privacy evidence contract are present | External federation run, malicious-secure aggregation review, deployment-specific poisoning-resilience evidence, and operator sign-off |
| R14 | Edge/mobile optimisation path | Partly implemented: `build_edge_runtime_readiness()` now verifies tracked quantised NLI, ONNX, Rust kernel, WASM source, target-matrix, deployment-doc, and latency-benchmark contracts; `tools/check_wasm_release_package.py` validates generated WASM package metadata and sha256 digests after `wasm-pack` build/test; `tools/run_wasm_browser_worker_smoke.py` runs a real headless Chrome module-Worker smoke against generated `backfire-wasm`; `benchmarks.edge_mobile_evidence` emits local tenant-safe evidence with browser-worker smoke attachment support; and Customer Model Factory release-gate edge/mobile evidence contract is present | Quantised model artefact, mobile or embedded-device smoke evidence, package-publish evidence, archived deployment latency packet, and operator sign-off |
| R15 | Continuous auto-redteam + defence genome loop | Partly implemented: `AutoRedteamDefenceLoop` now runs reviewed repeated adversarial-mining cycles, measures detection uplift against the active defence, promotes candidates through the defence update pipeline, emits a local tenant-safe evidence packet, and Customer Model Factory release-gate auto-redteam defence evidence contract is present | Live nightly run, operator-owned patch/model integration sign-off, external adversarial corpus evidence, rollback plan evidence, and operator sign-off |
| R16 | Formal + symbolic depth expansion | Partly implemented: formal/code adapter production paths now have local evidence for DPLL formula halts/allows, Lean runner contract, Z3 optional-profile gate or actual run when `[formal]` is installed, code-contract ordering, and tenant-safe serialisation | External Lean proof run, actual Z3 packet under `[formal]` in release evidence, and operator-owned math/code/numeric domain contracts |
| R17 | Deployment hardening + async/multi-tenant corrections | Partly implemented: production scaffold, `director-ai production-check` scaffold/secret validator, authenticated monitoring, async ordering regression, tenant poisoning regression, default-scale local sustained-load evidence runner, Customer Model Factory release-gate deployment-hardening evidence contract, and strict release-gate blockers for missing staging telemetry or operator sign-off are present; external deployment telemetry still open | Sustained staging or production run with archived telemetry plus tenant poisoning evidence packet |

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
