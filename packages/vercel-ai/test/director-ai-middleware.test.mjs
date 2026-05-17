import assert from "node:assert/strict";
import test from "node:test";

import {
  DirectorAiGuardrailError,
  createDirectorAiMiddleware,
  extractDirectorAiPrompt,
} from "../dist/index.js";

test("wrapGenerate reviews generated text and returns approved result", async () => {
  const calls = [];
  const middleware = createDirectorAiMiddleware({
    endpoint: "https://director.example.test",
    apiKey: "test-api-key",
    fetch: async (url, init) => {
      calls.push({ url, body: JSON.parse(init.body), headers: init.headers });
      return new Response(
        JSON.stringify({ approved: true, coherence: 0.91, h_logical: 0, h_factual: 0 }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    },
  });

  const result = await middleware.wrapGenerate({
    params: { prompt: [{ role: "user", content: [{ type: "text", text: "Refund policy?" }] }] },
    doGenerate: async () => ({
      content: [{ type: "text", text: "Refunds are available within 30 days." }],
      finishReason: { unified: "stop", raw: "stop" },
      usage: { inputTokens: 4, outputTokens: 7, totalTokens: 11 },
      warnings: [],
    }),
  });

  assert.equal(result.content[0].text, "Refunds are available within 30 days.");
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "https://director.example.test/v1/review");
  assert.equal(calls[0].headers["x-api-key"], "test-api-key");
  assert.deepEqual(calls[0].body, {
    prompt: "Refund policy?",
    response: "Refunds are available within 30 days.",
  });
});

test("wrapGenerate can mask rejected output without exposing raw response", async () => {
  const middleware = createDirectorAiMiddleware({
    endpoint: "https://director.example.test",
    onReject: "mask",
    rejectionMessage: "Blocked.",
    fetch: async () =>
      new Response(
        JSON.stringify({ approved: false, coherence: 0.2, h_logical: 0.7, h_factual: 0.8 }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
  });

  const result = await middleware.wrapGenerate({
    params: { prompt: "What is the refund policy?" },
    doGenerate: async () => ({
      content: [{ type: "text", text: "Unsafe answer." }],
      finishReason: { unified: "stop", raw: "stop" },
      usage: { inputTokens: 4, outputTokens: 3, totalTokens: 7 },
      warnings: [],
    }),
  });

  assert.deepEqual(result.content, [{ type: "text", text: "Blocked." }]);
});

test("wrapGenerate throws tenant-safe error on rejected review", async () => {
  const middleware = createDirectorAiMiddleware({
    endpoint: "https://director.example.test/",
    fetch: async () =>
      new Response(
        JSON.stringify({ approved: false, coherence: 0.12, h_logical: 0.8, h_factual: 0.9 }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
  });

  await assert.rejects(
    () =>
      middleware.wrapGenerate({
        params: { prompt: "What is the refund policy?" },
        doGenerate: async () => ({ text: "Refunds are never available." }),
      }),
    (error) =>
      error instanceof DirectorAiGuardrailError &&
      error.coherence === 0.12 &&
      !String(error.message).includes("Refunds are never available"),
  );
});

test("wrapStream buffers text deltas and releases chunks only after approval", async () => {
  const released = [];
  const middleware = createDirectorAiMiddleware({
    endpoint: "https://director.example.test",
    fetch: async (_url, init) => {
      const body = JSON.parse(init.body);
      assert.equal(body.response, "Approved answer.");
      assert.deepEqual(released, []);
      return new Response(
        JSON.stringify({ approved: true, coherence: 0.88, h_logical: 0, h_factual: 0 }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    },
  });

  const { stream } = await middleware.wrapStream({
    params: { prompt: "Question?" },
    doStream: async () => ({
      stream: new ReadableStream({
        start(controller) {
          controller.enqueue({ type: "text-start", id: "0" });
          controller.enqueue({ type: "text-delta", id: "0", delta: "Approved " });
          controller.enqueue({ type: "text-delta", id: "0", delta: "answer." });
          controller.enqueue({ type: "text-end", id: "0" });
          controller.close();
        },
      }),
    }),
  });

  for await (const chunk of stream) {
    released.push(chunk);
  }

  assert.deepEqual(
    released.map((chunk) => chunk.type),
    ["text-start", "text-delta", "text-delta", "text-end"],
  );
});

test("wrapStream rejects buffered stream without releasing chunks", async () => {
  const released = [];
  const middleware = createDirectorAiMiddleware({
    endpoint: "https://director.example.test",
    fetch: async () =>
      new Response(
        JSON.stringify({ approved: false, coherence: 0.2, h_logical: 0.7, h_factual: 0.8 }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
  });

  const { stream } = await middleware.wrapStream({
    params: { prompt: "Question?" },
    doStream: async () => ({
      stream: new ReadableStream({
        start(controller) {
          controller.enqueue({ type: "text-delta", id: "0", delta: "Unsafe answer." });
          controller.close();
        },
      }),
    }),
  });

  await assert.rejects(async () => {
    for await (const chunk of stream) {
      released.push(chunk);
    }
  }, DirectorAiGuardrailError);
  assert.deepEqual(released, []);
});

test("wrapStream can mask rejected buffered stream", async () => {
  const middleware = createDirectorAiMiddleware({
    endpoint: "https://director.example.test",
    onReject: "mask",
    rejectionMessage: "Blocked.",
    fetch: async () =>
      new Response(
        JSON.stringify({ approved: false, coherence: 0.2, h_logical: 0.7, h_factual: 0.8 }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
  });

  const { stream } = await middleware.wrapStream({
    params: { prompt: "Question?" },
    doStream: async () => ({
      stream: new ReadableStream({
        start(controller) {
          controller.enqueue({ type: "text-delta", id: "0", delta: "Unsafe answer." });
          controller.close();
        },
      }),
    }),
  });

  const released = [];
  for await (const chunk of stream) {
    released.push(chunk);
  }

  assert.deepEqual(released, [
    { type: "text-start", id: "director-ai-guard" },
    { type: "text-delta", id: "director-ai-guard", delta: "Blocked." },
    { type: "text-end", id: "director-ai-guard" },
  ]);
});

test("extractDirectorAiPrompt supports AI SDK prompt arrays and text blocks", () => {
  assert.equal(
    extractDirectorAiPrompt({
      prompt: [
        { role: "system", content: "System" },
        { role: "user", content: [{ type: "text", text: "Latest question" }] },
      ],
    }),
    "Latest question",
  );
});
