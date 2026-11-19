<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Director-AI Studio — portal and manifest development -->

# Director-AI Studio — AI-Safety / Streaming Academy (Tier-B portal)

Studio serves the Director-AI capability manifest and runs the generated
WASM coherence kernel in the browser. The Play route supplies sample coherence
scores and displays the native halt reason and token trace. The Python
contradiction interlock scores completed claims against retrieved facts.
Learn presents the streaming lesson and links to the Play exercise.
It shares the sample configuration with Play. Opening Play loads the generated
binding and WASM binary.

The navigation offers Home, Learn, Play and Capabilities. The planned Pipeline,
Benchmarks, Science & Evidence, Red-Team, Architecture and 3D Lab paths retain
their bookmark addresses and show their availability. Unknown addresses and
planned routes link back to the Studio home.

## Build and runtime boundaries

- The standalone app uses the local Instrument-Deck stylesheet and the local
  honesty contract in `src/contract/honesty.ts`.
- The browser fetches schema-A from the manifest emitted by
  `tools/emit_studio_manifest.py`. Studio displays the producer's declared
  safety tiers and keeps unknown fields in `extra`.
- Vite exposes `scpn_director_ai/DirectorAIStudioPanel` through Module Federation
  and shares React 19.3.1 and React DOM 19.3.1 as singletons. A federated host
  passes the Studio entry URL as `portalUrl` so the panel reads its manifest
  from the producer's origin.
- Each Play run opens and frees a native streaming session. The hard halt
  retains its triggering token in the trace and stops feeding later tokens.
  The trace keeps token text, score, activity and halt reason for each step.
  Returning to Play during initialisation reuses the pending WASM download.
  **Retry WASM** handles a failed asset request or compilation. If the loaded
  kernel fails during a run, **Reload page** fetches a fresh module and keeps
  the current route.
- Play shows a fixed message for load and run failures. Capabilities shows a
  fixed unavailable state for failed requests and invalid JSON.
- Dev and preview send COOP and COEP headers. Serve the built assets and the
  manifest from the same origin for the standalone app.
  Serve the built HTML with its Studio mount element intact.

## Build and test

From the repository root, run `make wasm-build` to generate the web-target
package. Then enter `studio-web` and install the locked Node dependency graph.

```bash
npm ci --ignore-scripts --strict-peer-deps
npm run lint
npm run test
npm run build
npx playwright install chromium
npm run e2e
```

Playwright opens the production build through Vite preview with Chromium's
sandbox enabled. Set `STUDIO_CHROME_EXECUTABLE` to use an installed Chrome
binary. The browser tests cover the capability manifest, native halt,
repeat runs, WASM asset recovery and the remote panel from a separate origin.
The required Rust CI job builds the web-target package from the same checkout,
installs Studio from its lockfile and runs lint, unit tests, the production build
and Playwright using the runner's sandboxed Google Chrome.
If a route bundle fails to load, the navigation stays available. **Reload
page** reloads the current route and its assets.
The separate Web Worker smoke command is `tools/run_wasm_browser_worker_smoke.py`.

## Manifest contract checks

The manifest reader has standalone checks using the SDK's locked
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
Legacy flat schema-A objects retain their existing `extra` layout. A malformed
`schema_a` or a top-level array uses safe defaults for typed fields. The reader
retains the original value in `manifest.extra.schema_a`, including the entire
array in input order when the payload is a top-level array.
The browser loader defaults to `/studio_manifest.json`, staged under `public/`
by `npm run sync:manifest`. It accepts an `AbortSignal`. The Capabilities
route cancels the request when leaving the view. The federated panel cancels
it when removed or when the portal URL changes. Every portal change shows
the loading state until the selected producer responds. These checks do not qualify
the full portal or WASM
runtime.

## Data sources and build inputs

- `docs/_generated/studio_manifest.json` — schema-A verbs/capabilities (Capabilities).
- `benchmarks/scores/factcg-*.json`, `benchmarks/results/*` — accuracy + FPR (Benchmarks, Evidence).
- `backfire-kernel/crates/backfire-wasm/pkg` — the live halt kernel (Play).
- the sealed evidence packet — digest-verified reproduction (Evidence, per H1–H3).
