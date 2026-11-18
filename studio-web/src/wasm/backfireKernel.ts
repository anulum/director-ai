// SPDX-License-Identifier: Apache-2.0
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// Director-AI Studio — adapter for the generated WASM coherence kernel.

import init, { WasmStreamingKernel } from "backfire-wasm";
import wasmUrl from "backfire-wasm/backfire_wasm_bg.wasm?url";
import { DEFAULT_BACKFIRE_CONFIG, type BackfireConfig } from "./backfireConfig";

/** Shared initialisation promise, cleared after a failed asset load. */
let ready: Promise<void> | null = null;

/** Load the bundled WASM asset and check the HTTP response. */
async function loadKernel(): Promise<void> {
  const response = await fetch(wasmUrl);
  if (!response.ok) throw new Error(`WASM request failed (HTTP ${response.status})`);
  await init({ module_or_path: response });
}

/** Reuse a successful or pending load, and allow retries after failure. */
export function initKernel(): Promise<void> {
  ready ??= loadKernel().catch((error: unknown) => {
    ready = null;
    throw error;
  });
  return ready;
}

/** One token's halt decision, mirrored from the kernel's session payload. */
export interface HaltStep {
  /** Text fragment passed to the native session for this step. */
  readonly token: string;
  /** Coherence supplied by the host for this fragment. */
  readonly score: number;
  /** Whether the native session can accept subsequent tokens. */
  readonly active: boolean;
  /** Native halt reason, empty while the session remains active. */
  readonly haltReason: string;
}

/** Read the reason from the generated module's session payload. */
function haltReason(session: unknown): string {
  if (
    typeof session !== "object" ||
    session === null ||
    !("halt_reason" in session) ||
    typeof session.halt_reason !== "string"
  ) {
    throw new TypeError("The WASM session did not contain a halt reason field");
  }
  return session.halt_reason;
}

/** A single stream over the WASM kernel. Create a new session to run again. */
export class StreamingHaltSession {
  /** Owns the WASM allocation for exactly one stream. */
  private readonly kernel: WasmStreamingKernel;

  /** Construct a session after the module has initialised. */
  constructor(config: BackfireConfig = DEFAULT_BACKFIRE_CONFIG) {
    this.kernel = new WasmStreamingKernel(JSON.stringify(config));
  }

  /** Feed a token and its host-supplied coherence score. */
  push(token: string, score: number): HaltStep {
    const session: unknown = this.kernel.process_token(token, score);
    return { token, score, active: this.kernel.is_active(), haltReason: haltReason(session) };
  }

  /** Release the WASM-owned memory when the stream finishes. */
  dispose(): void {
    this.kernel.free();
  }
}
