<!--
SPDX-License-Identifier: Apache-2.0
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-AI — Vercel middleware usage
-->
# @director-ai/vercel-ai

Vercel AI SDK middleware for Director-AI REST review gates.

```ts
import { generateText, wrapLanguageModel } from "ai";
import { createDirectorAiMiddleware } from "@director-ai/vercel-ai";

// `yourModel` is any AI SDK language model instance.
const model = wrapLanguageModel({
  model: yourModel,
  middleware: createDirectorAiMiddleware({
    endpoint: "http://localhost:8080",
    apiKey: process.env.DIRECTOR_API_KEY,
  }),
});

const result = await generateText({
  model,
  prompt: "What is the refund policy?",
});
```

The middleware reviews completed `generateText()` output and buffers
`streamText()` output before releasing it. Buffering is the safe default:
tokens are not sent to the client until `/v1/review` approves the final text.

Set `onReject: "mask"` to return a configured suppression message instead of
throwing `DirectorAiGuardrailError`.

Review responses require Boolean `approved` and a finite `coherence` score
between zero and one, inclusive. Optional `h_logical` and `h_factual` scores
use the same range; optional `warning` must be Boolean. Malformed responses,
HTTP or connection failures, and provider failures stop output release with
`DirectorAiReviewError`. Provider error events and `finishReason: error` statuses
fail before any review request, including when `onReject` is `mask`. Its message
is fixed and contains no original error text or cause. An explicit negative review remains `DirectorAiGuardrailError`,
whose `review` property retains the endpoint's decision and evidence.

Buffering retains the complete stream in memory and waits for it to finish
before review. This middleware does not cancel a provider operation already
in progress or provide an independent execution-authority boundary.

## Contributor checks

Use Node.js 26.13.0, matching the repository CI toolchain, then run:

```bash
npm ci --ignore-scripts
npm test
npm run docs
npm audit --audit-level=low
```

`npm test` builds the package, checks TypeScript and JavaScript test/tool types,
and enforces 100% line, branch and function coverage for the selected SDK
implementation and test/tool sources. Tests exercise the public AI SDK wrapper,
real loopback HTTP requests and readable streams with deterministic protocol
inputs. They do not measure model inference or qualify a deployed review service.
The root CI lint job runs these same checks. TypeDoc rejects missing API
documentation and other validation warnings.

The documentation collector in `tools/quality-ratchet.mjs` emits source-bound
diagnostics for the repository quality tooling; its successful capture alone
does not accept a historical baseline or enforce the whole repository ratchet.

The committed compiler profile uses TypeScript 6.0.3 and TypeDoc 0.28.20.
TypeDoc's declared compiler peer range ends at 6.0.x; compiler-major upgrades
require qualification of both the compiler API and the documentation builder.
The parser collector measures declarations and masking directives. Strict
semantic checking remains the separate `npm run check` and build contract.

## Generated package and documentation verification

From this directory in a source checkout, after the contributor checks above,
compare a real npm package and the generated documentation with fresh native
outputs. Python 3.11 or newer is required for the repository verifier:

```bash
repo_root="$(git rev-parse --show-toplevel)"
artefacts="$(mktemp -d)"
npm pack --offline --json --pack-destination "$artefacts"
PYTHONPATH="$repo_root" python -m tools.verify_typescript_generated \
  --root "$repo_root" \
  --artefacts "$repo_root/packages/vercel-ai" \
  --package "$artefacts/"*.tgz \
  --workspace "$artefacts/verification"
```

`npm pack` runs the original prepack/build hooks. The verifier compares every
`dist` and API-documentation file with a fresh compiler/TypeDoc run, and compares
the archive's complete member names, bytes and modes with the source and fresh
compiled output. It also checks archive framing and input custody under one
deadline. Its workspace must be new and outside the source and supplied outputs.
Keep the original npm command output, archive and verifier receipts together.

The identity binds Git HEAD, safe origin, package version, input/output modes,
installed dependency bytes and the executing verification tools. TypeDoc source
links can depend on the checkout path as well as the revision and origin;
documentation from another checkout is not normalized into acceptance. An absent
origin currently produces a fatal TypeDoc warning. Authenticated URLs, URL query
strings, fragments and whitespace are refused before origin output is recorded.
Source/declaration maps and inline sources need separate recipe qualification.

Successful comparison establishes same-source reproducibility and supplied
package agreement. It does not establish independent publication provenance,
qualify a remote npm release or certify the Studio/gateway integration.
