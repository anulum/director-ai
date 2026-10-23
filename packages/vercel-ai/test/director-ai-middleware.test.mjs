// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI — middleware generation and streaming HTTP contracts

import assert from "node:assert/strict";
import test from "node:test";
import { generateText, streamText, wrapLanguageModel } from "ai";

import { DirectorAiGuardrailError, DirectorAiReviewError, createDirectorAiMiddleware } from "../dist/index.js";
import { reviewServer } from "./review-http-server.mjs";

/** @type {import("@ai-sdk/provider").LanguageModelV3CallOptions} Original SDK input. */
const call = { prompt: [{ role: "user", content: [{ type: "text", text: "Refund policy?" }] }] };

/**
 * Provide an SDK-shaped completed response with explicit token usage.
 * @param {string} text Completed response.
 * @param {import("@ai-sdk/provider").LanguageModelV3FinishReason["unified"]} [reason] Provider completion status.
 * @returns {import("@ai-sdk/provider").LanguageModelV3GenerateResult} Provider result.
 */
function generation(text, reason = "stop") {
  return {
    content: [{ type: "text", text }],
    finishReason: { unified: reason, raw: reason },
    usage: {
      inputTokens: { total: 4, noCache: 4, cacheRead: 0, cacheWrite: 0 },
      outputTokens: { total: 7, text: 7, reasoning: 0 },
    },
    warnings: [],
  };
}

/**
 * Emit original provider chunks through an actual readable stream.
 * @param {import("@ai-sdk/provider").LanguageModelV3StreamPart[]} parts Provider chunks.
 * @returns {ReadableStream<import("@ai-sdk/provider").LanguageModelV3StreamPart>} Original stream.
 */
function streamFrom(parts) {
  return new ReadableStream({
    /**
     * Enqueue original provider chunks and finish the fixture stream.
     * @param {ReadableStreamDefaultController<import("@ai-sdk/provider").LanguageModelV3StreamPart>} controller Actual fixture controller.
     * @returns {void} The fixture stream is closed after its original chunks.
     */
    start(controller) {
      for (const part of parts) controller.enqueue(part);
      controller.close();
    },
  });
}

/**
 * Apply production middleware through the AI SDK's public model wrapper.
 * The local provider supplies protocol data; it does not perform AI inference.
 * @param {import("../dist/index.js").DirectorAiMiddlewareOptions} options Review endpoint.
 * @param {string} text Completed provider response.
 * @param {import("@ai-sdk/provider").LanguageModelV3StreamPart[]} parts Provider chunks.
 * @param {import("@ai-sdk/provider").LanguageModelV3FinishReason["unified"]} [reason] Provider completion status.
 * @returns {import("@ai-sdk/provider").LanguageModelV3} SDK-wrapped model.
 */
function guardedModel(options, text, parts, reason = "stop") {
  return wrapLanguageModel({
    model: {
      specificationVersion: "v3",
      provider: "http-contract-fixture",
      modelId: "middleware-contract",
      supportedUrls: {},
      doGenerate: async () => generation(text, reason),
      doStream: async () => ({ stream: streamFrom(parts) }),
    },
    middleware: createDirectorAiMiddleware(options),
  });
}

/** @type {import("@ai-sdk/provider").LanguageModelV3StreamPart[]} Original ordered chunks. */
const approvedParts = [
  { type: "text-start", id: "0" },
  { type: "text-delta", id: "0", delta: "Approved " },
  { type: "text-delta", id: "0", delta: "answer." },
  { type: "text-end", id: "0" },
];

test("wrapGenerate reviews generated text and returns approved result", { timeout: 5000 }, async (t) => {
  const service = await reviewServer(t, { approved: true, coherence: 0.91, h_logical: 0, h_factual: 0 });
  const model = guardedModel({ endpoint: service.endpoint, apiKey: "contract-api-key" },
    "Refunds are available within 30 days.", []);
  const result = await generateText({ model, prompt: "Refund policy?", maxRetries: 0 });
  assert.equal(result.text, "Refunds are available within 30 days.");
  assert.equal(service.requests.length, 1);
  assert.equal(service.requests[0]?.url, "/v1/review");
  assert.equal(service.requests[0]?.method, "POST");
  assert.equal(service.requests[0]?.headers["x-api-key"], "contract-api-key");
  assert.equal(service.requests[0]?.headers["content-type"], "application/json");
  assert.deepEqual(service.requests[0]?.body, {
    prompt: "Refund policy?", response: "Refunds are available within 30 days.",
  });
});

test("wrapGenerate can mask rejected output without exposing raw response", { timeout: 5000 }, async (t) => {
  const service = await reviewServer(t, { approved: false, coherence: 0.2 });
  const result = await guardedModel({
    endpoint: service.endpoint, onReject: "mask", rejectionMessage: "Blocked.",
  }, "Unsafe answer.", []).doGenerate(call);
  assert.deepEqual(result.content, [{ type: "text", text: "Blocked." }]);
  assert.deepEqual(result.usage, generation("").usage);
  assert.deepEqual(service.requests[0]?.body, { prompt: "Refund policy?", response: "Unsafe answer." });
});

test("wrapGenerate throws tenant-safe error on rejected review", { timeout: 5000 }, async (t) => {
  const decision = { approved: false, coherence: 0.12, h_logical: 0.8, h_factual: 0.9,
    warning: true, evidence: { reference: "review-contract" } };
  const service = await reviewServer(t, decision);
  const model = guardedModel({ endpoint: service.endpoint + "/" }, "Refunds are never available.", []);
  await assert.rejects(async () => model.doGenerate(call), /** @param {unknown} error Original rejection. */ (error) => {
    assert.ok(error instanceof DirectorAiGuardrailError);
    assert.equal(error.name, "DirectorAiGuardrailError");
    assert.equal(error.coherence, 0.12);
    assert.deepEqual(error.review, decision);
    assert.ok(!error.message.includes("Refunds are never available"));
    return true;
  });
  assert.equal(service.requests[0]?.url, "/v1/review");
});

test("wrapStream buffers text deltas and releases chunks only after approval", { timeout: 5000 }, async (t) => {
  /** @type {import("@ai-sdk/provider").LanguageModelV3StreamPart[]} Released original chunks. */
  const released = [];
  let observedReleaseCount = -1;
  const service = await reviewServer(t, { approved: true, coherence: 0.88 }, {
    onRequest: () => { observedReleaseCount = released.length; },
  });
  const { stream } = await guardedModel({ endpoint: service.endpoint }, "", approvedParts).doStream(call);
  for await (const chunk of stream) released.push(chunk);
  assert.equal(observedReleaseCount, 0);
  assert.deepEqual(released, approvedParts);
  assert.deepEqual(service.requests[0]?.body, { prompt: "Refund policy?", response: "Approved answer." });
});

test("wrapStream rejects buffered stream without releasing chunks", { timeout: 5000 }, async (t) => {
  const service = await reviewServer(t, { approved: false, coherence: 0.2 });
  const { stream } = await guardedModel({ endpoint: service.endpoint }, "", approvedParts).doStream(call);
  await assert.rejects(stream.getReader().read(), DirectorAiGuardrailError);
});

test("wrapStream can mask rejected buffered stream", { timeout: 5000 }, async (t) => {
  const service = await reviewServer(t, { approved: false, coherence: 0.2 });
  const { stream } = await guardedModel({
    endpoint: service.endpoint, onReject: "mask", rejectionMessage: "Blocked.",
  }, "", approvedParts).doStream(call);
  assert.deepEqual(await Array.fromAsync(stream), [
    { type: "text-start", id: "director-ai-guard" },
    { type: "text-delta", id: "director-ai-guard", delta: "Blocked." },
    { type: "text-end", id: "director-ai-guard" },
  ]);
});

test("default masking replaces both generation and streaming output", { timeout: 5000 }, async (t) => {
  const service = await reviewServer(t, { approved: false, coherence: 0 });
  const model = guardedModel({ endpoint: service.endpoint, onReject: "mask" }, "Unsafe.", approvedParts);
  const result = await model.doGenerate(call);
  assert.deepEqual(result.content, [{ type: "text", text: "Message suppressed by Director-AI." }]);
  const { stream } = await model.doStream(call);
  assert.deepEqual(await Array.fromAsync(stream), [
    { type: "text-start", id: "director-ai-guard" },
    { type: "text-delta", id: "director-ai-guard", delta: "Message suppressed by Director-AI." },
    { type: "text-end", id: "director-ai-guard" },
  ]);
  assert.equal(service.requests.length, 2);
});

test("custom transport performs HTTP at the configured review path", { timeout: 5000 }, async (t) => {
  const service = await reviewServer(t, { approved: true, coherence: 1 });
  /** @type {string[]} Actual delegated fetch URLs. */
  const urls = [];
  /** @type {typeof fetch} Fetch delegation retains actual network I/O. */
  const transport = (input, init) => {
    urls.push(String(input));
    return fetch(input, init);
  };
  const result = await guardedModel({
    endpoint: service.endpoint + "///", reviewPath: "/review/custom", fetch: transport,
  }, "Original.", []).doGenerate(call);
  assert.deepEqual(result.content, [{ type: "text", text: "Original." }]);
  assert.deepEqual(urls, [service.endpoint + "/review/custom"]);
  assert.equal(service.requests[0]?.url, "/review/custom");
  assert.equal(service.requests[0]?.headers["x-api-key"], undefined);
});

/** @type {[string, unknown, {status?: number, raw?: string}][]} Review failure carriers. */
const failures = [
  ["HTTP failure", { approved: true, coherence: 1 }, { status: 503 }],
  ["invalid JSON", null, { raw: '{"SENSITIVE_PROVIDER_DETAIL":' }],
  ["null JSON", null, {}],
  ["primitive JSON", true, {}],
  ["missing approval", { coherence: 0.5 }, {}],
  ["nonboolean approval", { approved: "true", coherence: 0.5 }, {}],
  ["missing coherence", { approved: true }, {}],
  ["nonnumeric coherence", { approved: true, coherence: "0.5" }, {}],
  ["negative coherence", { approved: true, coherence: -0.1 }, {}],
  ["excess coherence", { approved: true, coherence: 1.1 }, {}],
  ["infinite coherence", null, { raw: '{"approved":true,"coherence":1e999}' }],
  ["invalid logical score", { approved: true, coherence: 1, h_logical: "0" }, {}],
  ["invalid factual score", { approved: true, coherence: 1, h_factual: 2 }, {}],
  ["invalid warning", { approved: true, coherence: 1, warning: "false" }, {}],
];
for (const [name, decision, options] of failures) {
  test("review " + name + " prevents generation and stream release", { timeout: 5000 }, async (t) => {
    const service = await reviewServer(t, decision, options);
    const model = guardedModel({ endpoint: service.endpoint }, "Unreviewed.", approvedParts);
    await assert.rejects(async () => model.doGenerate(call), safeReviewFailure);
    const { stream } = await model.doStream(call);
    await assert.rejects(stream.getReader().read(), safeReviewFailure);
    assert.equal(service.requests.length, 2);
  });
}

/**
 * Verify the public error retains no original parser/provider detail or cause.
 * @param {unknown} error Failure received by an SDK caller.
 * @returns {boolean} True after the exact safe error contract is verified.
 */
function safeReviewFailure(error) {
  assert.ok(error instanceof DirectorAiReviewError);
  assert.equal(error.name, "DirectorAiReviewError");
  assert.equal(error.message, "Director-AI could not review model output.");
  assert.equal(Object.hasOwn(error, "cause"), false);
  return true;
}

test("upstream stream failure refuses before any review or release", { timeout: 5000 }, async (t) => {
  const service = await reviewServer(t, { approved: true, coherence: 1 });
  const model = guardedModel({ endpoint: service.endpoint }, "", []);
  const original = new Error("SENSITIVE_PROVIDER_DETAIL");
  let sent = false;
  const middleware = createDirectorAiMiddleware({ endpoint: service.endpoint });
  assert.ok(middleware.wrapStream);
  const { stream } = await middleware.wrapStream({
    model, params: call, doGenerate: model.doGenerate.bind(model, call),
    doStream: async () => ({
      stream: new ReadableStream({
        /**
         * Supply partial text, then fail before a completed response is available.
         * @param {ReadableStreamDefaultController<import("@ai-sdk/provider").LanguageModelV3StreamPart>} controller Actual failing stream controller.
         * @returns {void} Each pull supplies partial chunks or the terminal failure.
         */
        pull(controller) {
          if (!sent) {
            sent = true;
            controller.enqueue({ type: "text-start", id: "partial" });
            controller.enqueue({ type: "text-delta", id: "partial", delta: "Unreviewed partial text." });
          } else {
            controller.error(original);
          }
        },
      }),
    }),
  });
  await assert.rejects(stream.getReader().read(), safeReviewFailure);
  assert.equal(sent, true);
  assert.deepEqual(service.requests, []);
});

for (const mode of ["generate", "stream"]) {
  test("provider " + mode + " setup failure has a fixed caller-safe message", async (t) => {
    const service = await reviewServer(t, { approved: true, coherence: 1 });
    const model = guardedModel({ endpoint: service.endpoint }, "", []);
    const middleware = createDirectorAiMiddleware({ endpoint: service.endpoint });
    const invoke = mode === "generate" ? middleware.wrapGenerate : middleware.wrapStream;
    assert.ok(invoke);
    /** Provider rejection before a generation result or stream is returned. */
    const failure = async () => { throw new Error("SENSITIVE_PROVIDER_DETAIL"); };
    await assert.rejects(async () => invoke({
      model, params: call, doGenerate: failure, doStream: failure,
    }), safeReviewFailure);
    assert.deepEqual(service.requests, []);
  });
}

test("a closed review connection blocks generation and streaming", async (t) => {
  const service = await reviewServer(t, null, { disconnect: true });
  const model = guardedModel({ endpoint: service.endpoint }, "Unreviewed.", approvedParts);
  await assert.rejects(async () => model.doGenerate(call), safeReviewFailure);
  const { stream } = await model.doStream(call);
  await assert.rejects(() => Array.fromAsync(stream), safeReviewFailure);
  assert.equal(service.requests.length, 2);
});

test("empty endpoint refuses middleware construction", () => {
  assert.throws(() => createDirectorAiMiddleware({ endpoint: "" }), {
    name: "TypeError", message: "Director-AI endpoint is required",
  });
});

/** @type {[string, unknown, string][]} Untrusted provider results for rejection controls. */
const untrustedResults = [
  ["null result", null, ""],
  ["primitive result", 7, ""],
  ["legacy text result", { text: "Unreviewed legacy text." }, "Unreviewed legacy text."],
  ["legacy response result", { response: "Unreviewed legacy response." }, "Unreviewed legacy response."],
];
for (const [name, carrier, expectedText] of untrustedResults) {
  test("rejected " + name + " cannot escape the public generation hook", { timeout: 5000 }, async (t) => {
    const service = await reviewServer(t, { approved: false, coherence: 0 });
    const model = guardedModel({ endpoint: service.endpoint }, "", []);
    const middleware = createDirectorAiMiddleware({ endpoint: service.endpoint });
    assert.ok(middleware.wrapGenerate);
    const generate = middleware.wrapGenerate;
    await assert.rejects(async () => generate({
      model, params: call, doStream: model.doStream.bind(model, call),
      doGenerate: async () => /** @type {import("@ai-sdk/provider").LanguageModelV3GenerateResult} */ (carrier),
    }), DirectorAiGuardrailError);
    assert.deepEqual(service.requests[0]?.body, { prompt: "Refund policy?", response: expectedText });
  });
}

test("rejected malformed stream chunks never reach the consumer", { timeout: 5000 }, async (t) => {
  const service = await reviewServer(t, { approved: false, coherence: 0 });
  /** @type {unknown[]} Deliberately invalid native-provider protocol inputs. */
  const carrier = [null, { type: "text-delta", id: "0", delta: 7 }, ...approvedParts];
  const parts = /** @type {import("@ai-sdk/provider").LanguageModelV3StreamPart[]} */ (carrier);
  const { stream } = await guardedModel({ endpoint: service.endpoint }, "", parts).doStream(call);
  await assert.rejects(stream.getReader().read(), DirectorAiGuardrailError);
  assert.deepEqual(service.requests[0]?.body, { prompt: "Refund policy?", response: "Approved answer." });
});

/** @type {("throw" | "mask")[]} Rejection policies must not mask provider failures. */
const protocolPolicies = ["throw", "mask"];
for (const policy of protocolPolicies) {
  test("generateText refuses error completion before review with " + policy, async (t) => {
    const service = await reviewServer(t, { approved: true, coherence: 1 });
    const model = guardedModel({ endpoint: service.endpoint, onReject: policy },
      "Unreviewed partial response.", [], "error");
    await assert.rejects(generateText({ model, prompt: "Question?", maxRetries: 0 }), safeReviewFailure);
    assert.deepEqual(service.requests, []);
  });

  for (const signal of ["event", "finish"]) {
    test("streamText refuses protocol " + signal + " before release with " + policy, async (t) => {
      const service = await reviewServer(t, { approved: true, coherence: 1 });
      const original = new Error("SENSITIVE_PROVIDER_DETAIL");
      /** @type {import("@ai-sdk/provider").LanguageModelV3StreamPart[]} Valid provider failure after partial text. */
      const parts = [
        { type: "stream-start", warnings: [] }, ...approvedParts,
        signal === "event" ? { type: "error", error: original } :
          { type: "finish", finishReason: { unified: "error", raw: "error" }, usage: generation("").usage },
      ];
      const model = guardedModel({ endpoint: service.endpoint, onReject: policy }, "", parts);
      const guarded = await model.doStream(call);
      await assert.rejects(guarded.stream.getReader().read(), safeReviewFailure);
      const result = streamText({ model, prompt: "Question?", maxRetries: 0 });
      await assert.rejects(result.textStream.getReader().read(), safeReviewFailure);
      await assert.rejects(async () => await result.text, safeReviewFailure);
      assert.deepEqual(service.requests, []);
    });
  }
}

test("generateText preserves a normal successful completion", async (t) => {
  const service = await reviewServer(t, { approved: true, coherence: 1 });
  const model = guardedModel({ endpoint: service.endpoint }, "Completed.", []);
  const result = await generateText({ model, prompt: "Question?", maxRetries: 0 });
  assert.equal(result.text, "Completed.");
  assert.equal(result.finishReason, "stop");
  assert.equal(service.requests.length, 1);
});

test("streamText preserves a normal successful finish event", async (t) => {
  const service = await reviewServer(t, { approved: true, coherence: 1 });
  /** @type {import("@ai-sdk/provider").LanguageModelV3StreamPart[]} Complete successful provider protocol. */
  const parts = [{ type: "stream-start", warnings: [] }, ...approvedParts,
    { type: "finish", finishReason: { unified: "stop", raw: "stop" }, usage: generation("").usage }];
  const model = guardedModel({ endpoint: service.endpoint }, "", parts);
  const result = streamText({ model, prompt: "Question?", maxRetries: 0 });
  assert.equal(await result.text, "Approved answer.");
  assert.equal(await result.finishReason, "stop");
  assert.equal(service.requests.length, 1);
});
