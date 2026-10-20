<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Commercial license available -->
<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Director-AI - Roadmap Status Reconciliation -->

# Roadmap Status Reconciliation

Source status checked: 2026-10-20

This index records product capabilities and their qualification boundaries.
The source check covers the current checkout and the committed evidence
packets cited below. External inboxes, hosted services and deployment state
were not queried; packet declarations are not fresh runtime measurements.

## Reading the roadmap

1. [ROADMAP.md](../ROADMAP.md) describes public product direction and preserves
   dated release records. Earlier version targets are historical.
2. Implementation, package publication and deployment qualification are separate
   states. A source change does not establish a live service or approved claim.
3. Execution priorities and audit dispositions are maintained privately. They do
   not require publication here to become active work.
4. Deployment and migration checklists describe operator procedures; their
   presence does not establish an unfinished implementation.

## Implemented source with a remaining integration boundary

The advanced event-feedback SDK and CLI preserve source-bound original reviews
and signed requests, check current rights and reconcile durable feedback
acknowledgements. The
[integration contract](../docs-site/integrations/remanentia.md) describes the
required registration and custody checks. This receipt covers event feedback;
it does not authorise an action or confirm storage of a complete workflow
result. Complete autonomous approval, execution, independent verification and
result-custody qualification remains open.

## Qualification and release gates

| ID | Item | Status | Next evidence needed |
|---|---|---|---|
| R1 | Independent external security test focused on streaming paths and tenant isolation | The committed packet declares `run_status=not_run`. | Executed results and the required evidence/redaction review in `security/external_security_test_packet.toml`. Source review alone does not complete independent external testing. |
| R2 | Lite Scorer v2 artefact evaluation and release claim | `benchmarks/lite_scorer_v2_evidence_packet.toml` marks student, teacher, held-out evaluation, ONNX export, quantised latency, model card and claim review as recorded; `public_score_claim=false`. | Validate the exact recorded artefacts and obtain operator approval for any scored release claim, including its wording and release surfaces. |
| R3 | FrontierFail sourced production-failure corpus | `benchmarks/frontierfail_seed_packet.toml` describes seed regressions and incident intake; `public_benchmark_eligible=false`. | External corpus-quality review, preserved validation outputs and a benchmark card that respects the claim boundary. |
| R4 | PINT official upstream evidence | The official-export runner and seed replication packet are present; `benchmarks/pint_replication_packet.toml` sets `public_score_claim=false`. | Official upstream dataset export, execution output from `tools/run_pint_official_export.py`, dataset/result hashes and a benchmark card that respects the claim boundary. |
| R5 | Hugging Face Space live deployment | `demo/hf_space_deployment_smoke.toml` declares `deployment_status=not_published`, `smoke_status=pending` and `public_demo_claim=false`. | Operator-approved publication and dated smoke evidence for the published Space URL. |
| R6 | Polar storefront live deployment | `security/polar_deployment_smoke_packet.toml` declares checkout, portal, webhook and licence-validation checks pending; live-checkout and public-commercial claims remain false. | Dated live USD checkout, portal, webhook and licence-validation smoke records without committed secrets. |
| R7 | AggreFact leaderboard submission | Historical submission record dated 2026-06-02; current maintainer acknowledgement is unverified. | Verify the original submission correspondence and any maintainer response or upstream pull request before assigning a current submission status. |

## Historical closure records

Earlier closure records cover dependency closure, model revision
pinning, coverage-gate correction, cyber-physical adapter contracts, router
failure visibility, knowledge deletion semantics, runtime assert removal,
privacy/RNG handling, SQL safety, optional-extras CI, licence reconciliation,
enterprise ingestion, public/internal hygiene, and the later May product-maturity
slices. Those records describe their original scope and source revision;
they are not a claim that all current project work or runtime acceptance is
complete.

These items should not be re-opened from old unchecked checkboxes unless a fresh
current-code review finds a concrete regression.

## Stale Or Archival Material

Old implementation-plan checkboxes and retracted benchmark-gap plans do not
establish a current product gap. Reconciliation must:

- confirm the item is already implemented and record the evidence;
- retain a verified open product requirement with a concrete acceptance gate;
- mark the source as archival if its premise was superseded or retracted.

## Non-Backlog Checklist Classes

The following classes do not establish open implementation work by themselves:

- customer deployment checklists;
- migration checklists;
- examples that contain the word TODO as sample policy text;
- generated model/tokenizer outputs;
- historical audit notes whose findings were closed in later evidence records.
