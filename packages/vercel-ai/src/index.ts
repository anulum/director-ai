// SPDX-License-Identifier: Apache-2.0
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// Director-AI guardrail middleware for the Vercel AI SDK (LanguageModelV3).

import type {
  LanguageModelV3Middleware,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamPart,
} from "@ai-sdk/provider";

export interface DirectorAiReviewResponse {
  approved: boolean;
  coherence: number;
  h_logical?: number;
  h_factual?: number;
  warning?: boolean;
  evidence?: unknown;
}

export interface DirectorAiMiddlewareOptions {
  endpoint: string;
  apiKey?: string;
  fetch?: typeof fetch;
  reviewPath?: string;
  onReject?: "throw" | "mask";
  rejectionMessage?: string;
}

export class DirectorAiGuardrailError extends Error {
  readonly coherence: number;
  readonly review: DirectorAiReviewResponse;

  constructor(review: DirectorAiReviewResponse) {
    super(`Director-AI rejected model output (coherence=${review.coherence})`);
    this.name = "DirectorAiGuardrailError";
    this.coherence = review.coherence;
    this.review = review;
  }
}

export function createDirectorAiMiddleware(
  options: DirectorAiMiddlewareOptions,
): LanguageModelV3Middleware {
  const client = new DirectorAiReviewClient(options);
  return {
    specificationVersion: "v3",
    wrapGenerate: async ({ doGenerate, params }) => {
      const result = await doGenerate();
      const prompt = extractDirectorAiPrompt(params);
      const response = extractGenerateText(result);
      const review = await client.review(prompt, response);
      if (!review.approved) {
        return handleRejectedGenerate(result, review, options);
      }
      return result;
    },
    wrapStream: async ({ doStream, params }) => {
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
    },
  };
}

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

function guardedBufferedStream(
  stream: ReadableStream<LanguageModelV3StreamPart>,
  prompt: string,
  client: DirectorAiReviewClient,
  options: DirectorAiMiddlewareOptions,
): ReadableStream<LanguageModelV3StreamPart> {
  return new ReadableStream<LanguageModelV3StreamPart>({
    async start(controller) {
      const buffered: LanguageModelV3StreamPart[] = [];
      let response = "";
      try {
        for await (const chunk of stream) {
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
        controller.error(error);
      }
    },
  });
}

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

function extractGenerateText(result: unknown): string {
  const record = asRecord(result);
  if (!record) {
    return "";
  }
  return contentToText(record.text ?? record.content ?? record.response);
}

function streamPartText(chunk: LanguageModelV3StreamPart): string {
  const record = asRecord(chunk);
  if (!record || record.type !== "text-delta") {
    return "";
  }
  return typeof record.delta === "string" ? record.delta : "";
}

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

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null
    ? (value as Record<string, unknown>)
    : null;
}

class DirectorAiReviewClient {
  private readonly endpoint: string;
  private readonly fetchImpl: typeof fetch;
  private readonly apiKey?: string;
  private readonly reviewPath: string;

  constructor(options: DirectorAiMiddlewareOptions) {
    if (!options.endpoint) {
      throw new TypeError("Director-AI endpoint is required");
    }
    this.endpoint = options.endpoint.replace(/\/+$/, "");
    this.fetchImpl = options.fetch ?? fetch;
    this.apiKey = options.apiKey;
    this.reviewPath = options.reviewPath ?? "/v1/review";
  }

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
      throw new Error(
        `Director-AI review request failed with HTTP ${httpResponse.status}`,
      );
    }

    const payload = (await httpResponse.json()) as Partial<DirectorAiReviewResponse>;
    if (typeof payload.approved !== "boolean") {
      throw new Error("Director-AI review response missing boolean approved field");
    }
    if (typeof payload.coherence !== "number") {
      throw new Error("Director-AI review response missing numeric coherence field");
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
