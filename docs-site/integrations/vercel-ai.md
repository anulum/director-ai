# Vercel AI SDK

Director-AI ships a TypeScript bridge for the Vercel AI SDK middleware API.
It uses `wrapLanguageModel()` and reviews model output through the Director-AI
REST `/v1/review` endpoint.

```bash
cd packages/vercel-ai
npm install
npm run build
```

```ts
import { generateText, wrapLanguageModel } from "ai";
import { createDirectorAiMiddleware } from "@director-ai/vercel-ai";

// `yourModel` is any AI SDK language model instance.
const guardedModel = wrapLanguageModel({
  model: yourModel,
  middleware: createDirectorAiMiddleware({
    endpoint: "https://director.example.com",
    apiKey: process.env.DIRECTOR_API_KEY,
  }),
});

const result = await generateText({
  model: guardedModel,
  prompt: "What is the refund policy?",
});
```

## Streaming Safety

The middleware buffers `streamText()` text deltas, reviews the final text, and
only releases chunks after Director-AI approves the output. This preserves the
AI SDK stream protocol while avoiding a retroactive block after unsafe tokens
have already reached the client.

## Reject Handling

By default, rejected output raises `DirectorAiGuardrailError` without embedding
the raw model response in the error message.

```ts
createDirectorAiMiddleware({
  endpoint: "https://director.example.com",
  onReject: "throw",
});
```

Use mask mode when the application should return a fixed suppression message:

```ts
createDirectorAiMiddleware({
  endpoint: "https://director.example.com",
  onReject: "mask",
  rejectionMessage: "This answer was blocked by the safety layer.",
});
```

## Contract

| Option | Purpose |
|--------|---------|
| `endpoint` | Director-AI server root, for example `http://localhost:8080` |
| `apiKey` | Optional `X-API-Key` value |
| `reviewPath` | Override path, default `/v1/review` |
| `onReject` | `"throw"` or `"mask"` |
| `rejectionMessage` | Mask-mode replacement text |

The bridge targets AI SDK v6 `LanguageModelV3Middleware`.
