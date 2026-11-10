// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI — generated Node/WASM streaming measurement

import { createHash } from "node:crypto";
import { readFileSync, realpathSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { performance } from "node:perf_hooks";

/**
 * @typedef {object} StreamingKernel Methods consumed after checking the native object.
 * @property {(token: string, score: number) => unknown} process_token Process one token.
 * @property {() => unknown} get_session Read the native session snapshot.
 * @property {() => void} free Release the native allocation.
 */

/**
 * Check the real runtime instance before calling its generated methods.
 * @param {object} value Object returned by Reflect.construct.
 * @returns {asserts value is StreamingKernel} Checked invocation surface.
 */
function requireKernel(value) {
  if (!["process_token", "get_session", "free"].every(
        (key) => typeof Reflect.get(value, key) === "function")) {
    throw new Error("WASM module does not expose the streaming kernel methods");
  }
}

/**
 * @typedef {object} WasmMeasurement Complete report from the actual generated module.
 * @property {number} tokens Number of tokens consumed by each independent session.
 * @property {number} iterations Number of measured sessions.
 * @property {number[]} timings_ms Unrounded milliseconds for every completed session.
 * @property {string} node Actual executing Node version.
 * @property {string} module Resolved generated JavaScript entry point.
 * @property {Record<string, string>} artefacts SHA-256 of the actual loaded package inputs.
 */

/**
 * Measure the real generated per-token API, excluding module load and construction.
 * @param {string} modulePath Node-target wasm-bindgen JavaScript entry point.
 * @param {number} tokens Positive safe-integer session size.
 * @param {number} iterations Positive safe-integer independent session count.
 * @param {string} configJSON Complete native BackfireConfig JSON.
 * @returns {WasmMeasurement} Raw samples and exact loaded inputs.
 */
export function benchmarkWasm(modulePath, tokens, iterations, configJSON) {
  if (!Number.isSafeInteger(tokens) || tokens < 1 ||
      !Number.isSafeInteger(iterations) || iterations < 1) {
    throw new Error("tokens and iterations must be positive safe integers");
  }
  const entry = realpathSync(modulePath);
  /** Actual module exports are untrusted until the constructor check. @type {unknown} */
  const loaded = createRequire(import.meta.url)(entry);
  if (loaded === null || typeof loaded !== "object" ||
      !("WasmStreamingKernel" in loaded) || typeof loaded.WasmStreamingKernel !== "function") {
    throw new Error("WASM module has no WasmStreamingKernel constructor");
  }
  /** All samples are retained, without rounding or outlier removal. @type {number[]} */
  const timings = [];
  const inputs = Array.from({ length: tokens }, (_unused, index) => `token_${index}`);
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    /** Reflect.construct returns an object; check its native methods. @type {object} */
    const kernel = Reflect.construct(loaded.WasmStreamingKernel, [configJSON]);
    requireKernel(kernel);
    try {
      const started = performance.now();
      for (const token of inputs) kernel.process_token(token, 0.8);
      timings.push(performance.now() - started);
      const session = kernel.get_session();
      if (session === null || typeof session !== "object" ||
          !("tokens" in session) || !Array.isArray(session.tokens) ||
          session.tokens.length !== tokens || !("halted" in session) || session.halted !== false) {
        throw new Error("WASM streaming benchmark did not consume its full workload");
      }
    } finally {
      kernel.free();
    }
  }
  /** Original generated module, declarations, metadata and payload bytes. @type {Record<string, string>} */
  const artefacts = {};
  for (const name of ["package.json", "backfire_wasm.d.ts", "backfire_wasm_bg.wasm"]) {
    artefacts[name] = createHash("sha256").update(readFileSync(join(dirname(entry), name))).digest("hex");
  }
  artefacts["entry"] = createHash("sha256").update(readFileSync(entry)).digest("hex");
  return { tokens, iterations, timings_ms: timings, node: process.version, module: entry, artefacts };
}

/**
 * Run the real command-line argument contract without hiding native failures.
 * @param {string[]} argv Module, tokens, iterations and full configuration JSON.
 * @returns {WasmMeasurement} Actual native measurement packet.
 */
export function main(argv) {
  const [modulePath, tokens, iterations, configJSON, ...extra] = argv;
  if (modulePath === undefined || configJSON === undefined || extra.length !== 0) {
    throw new Error("Expected module, token count, iteration count and configuration JSON");
  }
  return benchmarkWasm(modulePath, Number(tokens), Number(iterations), configJSON);
}

if (import.meta.main) console.log(JSON.stringify(main(process.argv.slice(2))));
