// SPDX-License-Identifier: Apache-2.0
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// Director-AI Studio — live streaming-halt kernel (the COMMITTED backfire-wasm).
//
// Wraps the real wasm-pack output at
// `backfire-kernel/crates/backfire-wasm/pkg` — the SAME kernel the benchmarks
// ran, not a re-implementation. This is the Play section's signature seam: the
// contradiction-driven halt runs client-side, token by token, no server.
//
// Per H1 the in-browser run is a CONSUMER signal; it never sets verification.
// A claim-reproduction (re-deriving a committed number) compares digests via
// `judgeReproduction` (H2) — that path is wired in the Evidence section, not here.

import init, { WasmStreamingKernel } from "backfire-wasm";

let ready: Promise<void> | null = null;

/** Initialise the WASM module once (idempotent); lazy-loaded on the Play route. */
export function initKernel(): Promise<void> {
  ready ??= init().then(() => undefined);
  return ready;
}

/** One token's halt decision, mirrored from the kernel's session payload. */
export interface HaltStep {
  readonly token: string;
  readonly score: number;
  readonly active: boolean;
  readonly session: unknown;
}

/**
 * A live streaming-halt session over the committed kernel.
 *
 * Construct with the guard config (serialised as the kernel expects), then feed
 * tokens via {@link push}; `active` flips to false the moment a completed claim
 * contradicts the grounding and the stream halts. The same logic the science ran.
 */
export class StreamingHaltSession {
  private readonly kernel: WasmStreamingKernel;

  constructor(config: Record<string, unknown>) {
    this.kernel = new WasmStreamingKernel(JSON.stringify(config));
  }

  /** Feed one token + its contradiction score; returns the post-token state. */
  push(token: string, score: number): HaltStep {
    const session = this.kernel.process_token(token, score);
    return { token, score, active: this.kernel.is_active(), session };
  }

  /** Whether the stream is still live (false once it has halted). */
  get active(): boolean {
    return this.kernel.is_active();
  }

  /** The kernel's current session snapshot (for the trajectory view). */
  snapshot(): unknown {
    return this.kernel.get_session();
  }

  /** Release the WASM-owned memory; call when the demo unmounts. */
  dispose(): void {
    this.kernel.free();
  }
}

/** Convenience: init then open a session in one call. */
export async function openSession(
  config: Record<string, unknown>,
): Promise<StreamingHaltSession> {
  await initKernel();
  return new StreamingHaltSession(config);
}
