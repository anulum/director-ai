<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Director-AI Studio — portal and manifest development -->

# Director-AI Studio — AI-Safety / Streaming Academy (Tier-B portal)

Phase-0 skeleton for the dedicated Director-AI portal: the place to **learn** how
an LLM hallucination guardrail works, **play** with the contradiction-halt live in
your browser, and **audit** its accuracy *and its false-positive cost*. Built to
the fleet pattern (`DEDICATED_STUDIO_PORTALS_PATTERN_*` / `*_AHEAD_OF_SOTA_*`) and
the CEO honesty rulings H1–H4; baseline:
`.coordination/planning/SCPN-STUDIO/AI_SAFETY_STREAMING_ACADEMY_BASELINE_2026-06-24T1911.md`.

> **Status — Phase-0 scaffold, pre-ratification.** This stages the *seven never-redo
> seams* as concrete, reviewable structure. It does **not** yet `npm install`/build
> green here: the shared `@anulum/ui` + `@anulum/portal-kit` workspace libraries are
> published by the SCPN-STUDIO keeper (ratified, not yet released), so this scaffold
> uses a **local Instrument-Deck token mirror** (`src/tokens/`) and a **local honesty
> bridge** (`src/contract/honesty.ts`) that are swapped for the shared packages with
> no structural change once they land. The WASM glue, the contract-consumption layer,
> and the content schema are real and bind to **committed** artefacts.

## The seven seams baked in (research §7 — get these right → never rework)

1. **Contract is the only seam** — `src/contract/manifest.ts` reads the committed
   schema-A `studio_manifest.json` (produced by `tools/emit_studio_manifest.py`),
   forward-tolerant (unknown fields remain available in `extra`). The portal
   never reaches into Python internals.
2. **Shared libraries** — `@anulum/ui` + `@anulum/portal-kit` are the only styling/
   layout source (local mirror in `src/tokens/` until they publish).
3. **MFE independence** — `vite.config.ts` exposes `scpn_director_ai /
   ./DirectorAIStudioPanel` (Module Federation 2.x), singleton-shared react.
4. **Static + WASM** — no backend; the live halt runs the **committed**
   `backfire-wasm` (`WasmStreamingKernel.process_token`) client-side. COOP/COEP
   headers set for future multithreading.
5. **Content as data** — `src/content/schema.ts` + one example lesson binding to a
   real scorer artefact; new science → new content, zero architecture change.
6. **Versioned, drift-guarded** — the consumed manifest is drift-gated in CI
   (`emit_studio_manifest.py --check`); evidence is digest-sealed.
7. **Forward-tolerant consumption** — the manifest reader tolerates unknown verbs/
   fields and degrades honestly.

## Honesty rulings (CEO, 2026-06-24) — enforced in `src/contract/honesty.ts`

- **H1** in-browser reproduction is a **consumer** signal ("reproduced in your
  browser ✓"), never auto-promoted to `verified-at-source`.
- **H2** reproduction is **digest-based**; a content-digest mismatch fires the
  honesty gate loudly as drift.
- **H3** reproduction respects **Exactness** — bit-exact vs tolerance-aware, the UI
  says which.
- **H4** the `@anulum/ui` honesty bridge is the single source of grading;
  `falsified`/`refuted` are first-class.

## Section spine (`learn → play → audit`)

Home · Learn · **Play** (live halt) · Pipeline · Capabilities · Benchmarks ·
Science & Evidence · Red-Team/Robustness · Architecture & Federation · 3D Lab.

## Build (once the shared packages publish)

```bash
npm install
npm run dev          # local dev server
npm run build        # standalone static build + the MFE remoteEntry.js
npm run test         # Vitest unit/component
npm run e2e          # Playwright (home → learn → play → evidence)
```

The full portal build also requires a real `backfire-wasm` package at
`../backfire-kernel/crates/backfire-wasm/pkg`. The manifest checks below do not
build that package or qualify its browser behaviour. The separate kernel smoke
runner is `tools/run_wasm_browser_worker_smoke.py`.

## Manifest contract checks

The manifest reader can be checked independently of the portal's unresolved
React, shared-package and WASM build dependencies. Use the repository's locked
TypeScript compiler and Node 26.13.0:

```bash
npm ci --prefix ../packages/vercel-ai --ignore-scripts
npm run test:manifest
npm run docs:manifest
```

The test command strictly checks the actual TypeScript reader and native JavaScript
tests, then requires 100% native line, branch and function coverage for both.
The tests stage the committed `docs/_generated/studio_manifest.json` with the
actual sync script and consume those bytes through real loopback HTTP. They
also check flat-payload compatibility, future eras and verbs, unknown fields,
malformed fields, HTTP refusal, invalid JSON and a disconnected endpoint.
The existing Vitest and full portal build commands remain separate requirements.

`parseManifest` and `loadManifest` accept the producer's federation envelope:
typed fields come from its own `schema_a` member. Unknown envelope fields are
retained in `manifest.extra`; unknown nested schema fields are retained in
`manifest.extra.schema_a`, keeping equal names at different levels distinct.
Legacy flat schema-A payloads retain their existing `extra` layout. A malformed
`schema_a` uses safe defaults rather than accepting outer identity fields, while
its original value remains unchanged in `manifest.extra.schema_a`.
The browser loader defaults to `/studio_manifest.json`, staged under `public/`
by `npm run sync:manifest`. These checks do not qualify the full portal or WASM
runtime.

## Data sources and build inputs

- `docs/_generated/studio_manifest.json` — schema-A verbs/capabilities (Capabilities).
- `benchmarks/scores/factcg-*.json`, `benchmarks/results/*` — accuracy + FPR (Benchmarks, Evidence).
- `backfire-kernel/crates/backfire-wasm/pkg` — the live halt kernel (Play).
- the sealed evidence packet — digest-verified reproduction (Evidence, per H1–H3).
