// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI — Vercel AI SDK review middleware

import type {
  LanguageModelV3Middleware,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamPart,
} from "@ai-sdk/provider";

/** Decision returned by the configured Director-AI review endpoint. */
export interface DirectorAiReviewResponse {
  /** Whether the endpoint permits release of the completed response. */
  approved: boolean;
  /** Finite coherence score in the inclusive range zero to one. */
  coherence: number;
  /** Optional logical component reported by the review service. */
  h_logical?: number;
  /** Optional factual component reported by the review service. */
  h_factual?: number;
  /** Optional warning state accompanying the endpoint's decision. */
  warning?: boolean;
  /** Endpoint-supplied evidence retained without interpreting its structure. */
  evidence?: unknown;
}

/** Transport and rejection behaviour for the review middleware. */
export interface DirectorAiMiddlewareOptions {
  /** Nonempty review-service base URL; trailing slashes are removed. */
  endpoint: string;
  /** Optional value sent in the review request's `x-api-key` header. */
  apiKey?: string;
  /** Fetch transport; defaults to the runtime's global `fetch`. */
  fetch?: typeof fetch;
  /** Path appended to the base URL; defaults to `/v1/review`. */
  reviewPath?: string;
  /** Throw on rejection by default, or replace output when set to `mask`. */
  onReject?: "throw" | "mask";
  /** Mask text; defaults to `Message suppressed by Director-AI.`. */
  rejectionMessage?: string;
}

/** Rejected review decision, with its score and original decision metadata. */
export class DirectorAiGuardrailError extends Error {
  /** Coherence score from the rejected review. */
  readonly coherence: number;
  /** Complete decision returned by the review endpoint. */
  readonly review: DirectorAiReviewResponse;

  /**
   * Construct a rejection error whose message identifies the coherence score.
   * @param review - Original rejected review decision.
   */
  constructor(review: DirectorAiReviewResponse) {
    super(`Director-AI rejected model output (coherence=${review.coherence})`);
    this.name = "DirectorAiGuardrailError";
    this.coherence = review.coherence;
    this.review = review;
  }
}

/** Review or provider failure whose message contains no remote error text. */
export class DirectorAiReviewError extends Error {
  /** Construct a fixed caller-safe failure without retaining the original cause. */
  constructor() {
    super("Director-AI could not review model output.");
    this.name = "DirectorAiReviewError";
  }
}

/**
 * Review completed generation and buffer streaming chunks until review finishes.
 *
 * Approved output is returned unchanged. Rejected output throws by default or
 * is replaced with mask text. Provider, transport, parsing and validation failures
 * reject with DirectorAiReviewError before output is released. Protocol error
 * events and error completion statuses fail before any review request.
 *
 * @param options - Review endpoint, transport and rejection policy.
 * @returns LanguageModelV3 middleware for the Vercel AI SDK.
 * @throws TypeError if the endpoint is empty.
 */
export function createDirectorAiMiddleware(
  options: DirectorAiMiddlewareOptions,
): LanguageModelV3Middleware {
  const client = new DirectorAiReviewClient(options);
  return {
    specificationVersion: "v3",
    wrapGenerate: async ({ doGenerate, params }) => {
      try {
        const result = await doGenerate();
        if (result?.finishReason?.unified === "error") {
          throw new DirectorAiReviewError();
        }
        const prompt = extractDirectorAiPrompt(params);
        const response = extractGenerateText(result);
        const review = await client.review(prompt, response);
        if (!review.approved) {
          return handleRejectedGenerate(result, review, options);
        }
        return result;
      } catch (error) {
        throw callerSafeError(error);
      }
    },
    wrapStream: async ({ doStream, params }) => {
      try {
        const streamResult = await doStream();
        const prompt = extractDirectorAiPrompt(params);
        return {
          ...streamResult,
          stream: guardedBufferedStream(
            streamResult.stream,
            prompt,
            client,
            options,
          ),
        };
      } catch (error) {
        throw callerSafeError(error);
      }
    },
  };
}

/**
 * Preserve an authored rejection; discard all other providers' error details.
 * @param error - Original failure, whose message and cause are never copied.
 * @returns Deliberate refusal or a new fixed review failure.
 */
function callerSafeError(error: unknown): Error {
  return error instanceof DirectorAiGuardrailError
    ? error
    : new DirectorAiReviewError();
}

/**
 * Extract the last user message, a direct prompt, or the prompt-array fallback.
 *
 * Nonobject parameters produce empty text. Content arrays are joined as text;
 * other content objects use their text/content string or JSON representation.
 *
 * @param params - Original SDK parameters or an unrecognised runtime value.
 * @returns Extracted text, with empty text for absent content.
 */
export function extractDirectorAiPrompt(params: unknown): string {
  const maybeParams = asRecord(params);
  if (!maybeParams) {
    return "";
  }
  const prompt = maybeParams.prompt;
  if (typeof prompt === "string") {
    return prompt;
  }
  if (Array.isArray(prompt)) {
    for (let index = prompt.length - 1; index >= 0; index -= 1) {
      const message = asRecord(prompt[index]);
      if (message?.role === "user") {
        return contentToText(message.content);
      }
    }
    return prompt.map((message) => contentToText(message)).join(" ").trim();
  }
  return contentToText(prompt);
}

/**
 * Consume and review the complete input stream before releasing any chunks.
 * @param stream - Original model stream, including nontext chunks.
 * @param prompt - Extracted user prompt supplied to the review endpoint.
 * @param client - Configured review transport.
 * @param options - Rejection policy and optional mask message.
 * @returns Stream of approved original chunks or rejection mask chunks.
 */
function guardedBufferedStream(
  stream: ReadableStream<LanguageModelV3StreamPart>,
  prompt: string,
  client: DirectorAiReviewClient,
  options: DirectorAiMiddlewareOptions,
): ReadableStream<LanguageModelV3StreamPart> {
  return new ReadableStream<LanguageModelV3StreamPart>({
    /**
     * Buffer original chunks and release them only after the complete review.
     * @param controller - Destination stream controller for approved or masked chunks.
     * @returns Completion after approval, masking, or a reported stream error.
     */
    async start(controller) {
      const buffered: LanguageModelV3StreamPart[] = [];
      let response = "";
      try {
        for await (const chunk of stream) {
          if (chunk?.type === "error" ||
            (chunk?.type === "finish" && chunk.finishReason.unified === "error")) {
            throw new DirectorAiReviewError();
          }
          buffered.push(chunk);
          response += streamPartText(chunk);
        }

        const review = await client.review(prompt, response);
        if (!review.approved) {
          if (options.onReject === "mask") {
            enqueueMaskChunk(controller, options.rejectionMessage);
            controller.close();
            return;
          }
          throw new DirectorAiGuardrailError(review);
        }

        for (const chunk of buffered) {
          controller.enqueue(chunk);
        }
        controller.close();
      } catch (error) {
        controller.error(callerSafeError(error));
      }
    },
  });
}

/**
 * Replace rejected content while preserving result metadata, or throw.
 * @param result - Original completed model result.
 * @param review - Rejected endpoint decision.
 * @param options - Rejection policy and optional mask text.
 * @returns Result with mask content when masking is selected.
 * @throws DirectorAiGuardrailError when masking is not selected.
 */
function handleRejectedGenerate(
  result: LanguageModelV3GenerateResult,
  review: DirectorAiReviewResponse,
  options: DirectorAiMiddlewareOptions,
): LanguageModelV3GenerateResult {
  if (options.onReject !== "mask") {
    throw new DirectorAiGuardrailError(review);
  }
  return {
    ...result,
    content: [
      {
        type: "text",
        text: options.rejectionMessage ?? "Message suppressed by Director-AI.",
      },
    ],
  } satisfies LanguageModelV3GenerateResult;
}

/**
 * Emit one complete text block containing the configured suppression message.
 * @param controller - Guarded stream receiving the mask chunks.
 * @param message - Suppression text, or the default message when omitted.
 */
function enqueueMaskChunk(
  controller: ReadableStreamDefaultController<LanguageModelV3StreamPart>,
  message = "Message suppressed by Director-AI.",
) {
  controller.enqueue({ type: "text-start", id: "director-ai-guard" });
  controller.enqueue({
    type: "text-delta",
    id: "director-ai-guard",
    delta: message,
  });
  controller.enqueue({ type: "text-end", id: "director-ai-guard" });
}

/**
 * Resolve generated text from text, content or response fields in that order.
 * @param result - Completed model result or an unrecognised runtime value.
 * @returns Review text, or empty text for a nonobject result.
 */
function extractGenerateText(result: unknown): string {
  const record = asRecord(result);
  if (!record) {
    return "";
  }
  return contentToText(record.text ?? record.content ?? record.response);
}

/**
 * Extract string text deltas while excluding nontext or malformed chunks.
 * @param chunk - Original stream part received from the model.
 * @returns Delta text, or empty text for other chunk shapes.
 */
function streamPartText(chunk: LanguageModelV3StreamPart): string {
  const record = asRecord(chunk);
  if (!record || record.type !== "text-delta") {
    return "";
  }
  return typeof record.delta === "string" ? record.delta : "";
}

/**
 * Convert strings, nested content arrays and other runtime content into text.
 * @param content - Original content value, without assuming its shape.
 * @returns Text/content string, joined array text, primitive text or object JSON.
 */
function contentToText(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (content == null) {
    return "";
  }
  if (Array.isArray(content)) {
    return content.map((part) => contentToText(part)).join(" ").trim();
  }
  const record = asRecord(content);
  if (!record) {
    return String(content);
  }
  if (typeof record.text === "string") {
    return record.text;
  }
  if (typeof record.content === "string") {
    return record.content;
  }
  return JSON.stringify(record);
}

/**
 * Expose an object value for guarded field access without coercing primitives.
 * @param value - Original unrecognised runtime value.
 * @returns Object record, or null for primitives and null itself.
 */
function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null
    ? (value as Record<string, unknown>)
    : null;
}

/** POST prompt/response pairs and validate the required review decision fields. */
class DirectorAiReviewClient {
  /** Base endpoint with trailing slashes removed. */
  private readonly endpoint: string;
  /** Supplied or global fetch transport. */
  private readonly fetchImpl: typeof fetch;
  /** Optional review-service API key. */
  private readonly apiKey?: string;
  /** Review path appended to the configured base endpoint. */
  private readonly reviewPath: string;

  /**
   * Resolve the configured endpoint, transport, credentials and review path.
   * @param options - Original middleware configuration.
   * @throws TypeError if the configured endpoint is empty.
   */
  constructor(options: DirectorAiMiddlewareOptions) {
    if (!options.endpoint) {
      throw new TypeError("Director-AI endpoint is required");
    }
    this.endpoint = options.endpoint.replace(/\/+$/, "");
    this.fetchImpl = options.fetch ?? fetch;
    this.apiKey = options.apiKey;
    this.reviewPath = options.reviewPath ?? "/v1/review";
  }

  /**
   * Request a decision and retain the endpoint's optional review metadata.
   * @param prompt - Extracted original user prompt.
   * @param response - Completed generation text or joined stream text deltas.
   * @returns Review with Boolean approval and finite scores in zero to one.
   * @throws DirectorAiReviewError if HTTP or required/optional fields are invalid.
   * Transport and JSON failures are normalised at the public middleware boundary.
   */
  async review(
    prompt: string,
    response: string,
  ): Promise<DirectorAiReviewResponse> {
    const headers: Record<string, string> = {
      "content-type": "application/json",
    };
    if (this.apiKey) {
      headers["x-api-key"] = this.apiKey;
    }

    const httpResponse = await this.fetchImpl(this.endpoint + this.reviewPath, {
      method: "POST",
      headers,
      body: JSON.stringify({ prompt, response }),
    });
    if (!httpResponse.ok) {
      throw new DirectorAiReviewError();
    }

    const payload = asRecord(await httpResponse.json());
    if (
      !payload ||
      typeof payload.approved !== "boolean" ||
      !isScore(payload.coherence) ||
      (payload.h_logical !== undefined && !isScore(payload.h_logical)) ||
      (payload.h_factual !== undefined && !isScore(payload.h_factual)) ||
      (payload.warning !== undefined && typeof payload.warning !== "boolean")
    ) {
      throw new DirectorAiReviewError();
    }
    return {
      approved: payload.approved,
      coherence: payload.coherence,
      h_logical: payload.h_logical,
      h_factual: payload.h_factual,
      warning: payload.warning,
      evidence: payload.evidence,
    };
  }
}

/**
 * Validate a finite probability score without coercing remote values.
 * @param value - Original decoded response member.
 * @returns Whether the value is a finite number in the inclusive unit interval.
 */
function isScore(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
}
