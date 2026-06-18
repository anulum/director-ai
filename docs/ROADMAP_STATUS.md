<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Commercial license available -->
<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Director-Class AI - Roadmap Status Reconciliation -->

# Roadmap Status Reconciliation

Last reconciled: 2026-06-18

This file is the public status index for distinguishing active roadmap work
from completed, blocked, stale, and speculative planning material. The
canonical feature roadmap remains `ROADMAP.md`; this file records what is
still open after comparing the roadmap with current implementation notes,
validation records, and release documentation.

## Source Hierarchy

1. `ROADMAP.md` is the active public engineering roadmap.
2. Long-range strategic concepts and the differentiator prioritisation queue are
   tracked internally and are not published here to avoid premature disclosure.
3. Internal planning and audit notes are evidence sources, not automatically
   active backlog. Any unchecked internal note must be reconciled here or in
   `ROADMAP.md` before it is treated as live work.
4. User-facing deployment, migration, and operations checklists are operator
   checklists, not repository TODO items.

## Active Actionable Items

| ID | Item | Status | Next evidence needed |
|---|---|---|---|
| R1 | Independent external security test focused on streaming paths and tenant isolation | Blocked on independent external reviewer | Third-party report or signed internal exception |
| R2 | Lite Scorer v2 training completion and real artefact evaluation | Partly implemented: durable training launcher, run manifest, held-out builder, guarded evaluator, ONNX export runner, evidence recorder, and validator are present; `tools/validate_lite_scorer_v2_plan.py --require-recorded-evidence` now returns non-zero until student, teacher, ONNX, held-out evaluation, quantized latency, model-card, and benchmark-claim review statuses are recorded or validated | Completed training artefacts, evaluation JSON, model card, benchmark-claim review, passing recorded-evidence gate, and operator approval for any scored release claim |
| R3 | FrontierFail broader sourced production-failure corpus | Partly implemented: FrontierFail seed packet now includes synthetic regressions plus public incident intake rows with dated evidence, category/domain/publisher/evidence-reference diversity gates, public metadata requirements, and no public benchmark eligibility claim | Additional public incidents, external corpus-quality review, preserved validation outputs, and a claim-guarded benchmark card |
| R4 | PINT official upstream run and evidence | Open | Official upstream dataset execution, preserved outputs, and claim-guarded benchmark card |
| R5 | Hugging Face Space live deployment push | Open/manual | Published Space URL plus deployment smoke evidence |
| R6 | Polar storefront environment-specific live deployment smoke | Open/environment-specific; pricing is USD-first and request-checkout mail links remain in place until matching USD Polar products are created and smoked | Live USD checkout, portal, webhook, and licence validation smoke records without committed secrets |
| R7 | AggreFact leaderboard submission | Submitted by email on 2026-06-02; awaiting maintainer response | Maintainer response, submission acknowledgement, or upstream pull-request URL |

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
