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

Use Node.js 26.10.0, matching the repository CI toolchain, then run:

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
