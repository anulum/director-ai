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
