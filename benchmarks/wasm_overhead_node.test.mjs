// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI — real Node/WASM comparison and Python CLI contracts

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync, spawnSync } from "node:child_process";
import { copyFileSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import { benchmarkWasm, main } from "./wasm_overhead_node.mjs";

/** Actual generated Node package produced by the repository's Node build. */
const modulePath = fileURLToPath(
  new URL("../backfire-kernel/crates/backfire-wasm/pkg-nodejs/backfire_wasm.js", import.meta.url));
/** Complete native configuration, including fields outside the streaming gate. */
const config = {
  coherence_threshold: 0.6, hard_limit: 0.5, soft_limit: 0.7, w_logic: 0.6, w_fact: 0.4,
  window_size: 10, window_threshold: 0.55, trend_window: 5, trend_threshold: 0.15,
  history_window: 5, deadline_ms: 50, logit_entropy_limit: 1.2,
};
/** Exact JSON passed through the genuine wasm-bindgen constructor. */
const configJSON = JSON.stringify(config);
/** Only this test-created directory is removed after the native controls finish. */
const scratch = mkdtempSync(join(tmpdir(), "director-wasm-comparison-"));
after(() => rmSync(scratch, { recursive: true }));

test("actual WASM exports complete every measured session", () => {
  const result = benchmarkWasm(modulePath, 3, 2, configJSON);
  assert.equal(result.tokens, 3);
  assert.equal(result.iterations, 2);
  assert.equal(result.timings_ms.length, 2);
  assert.ok(result.timings_ms.every((value) => Number.isFinite(value) && value > 0));
  const entryHash = result.artefacts["entry"];
  const wasmHash = result.artefacts["backfire_wasm_bg.wasm"];
  assert.ok(entryHash);
  assert.ok(wasmHash);
  assert.match(entryHash, /^[a-f0-9]{64}$/);
  assert.match(wasmHash, /^[a-f0-9]{64}$/);
  assert.equal(result.node, process.version);
});

test("the genuine native gate refuses a benchmark that halts at its final token", () => {
  assert.throws(() => benchmarkWasm(modulePath, 3, 1,
    JSON.stringify({ ...config, window_size: 3, window_threshold: 0.9 })), /full workload/);
  assert.throws(() => benchmarkWasm(modulePath, 3, 1, "invalid JSON"));
});

test("invalid workload counts are refused before module execution", () => {
  /** Invalid numeric workloads, with both positions present. @type {Array<[number, number]>} */
  const counts = [
    [0, 2], [-1, 2], [1.5, 2], [Number.NaN, 2], [3, 0], [3, -1], [3, 1.5], [3, Infinity],
  ];
  for (const [tokens, iterations] of counts) {
    assert.throws(() => benchmarkWasm(modulePath, tokens, iterations, configJSON), /safe integers/);
  }
});

test("the native module loader refuses incompatible export shapes", () => {
  for (const [index, source] of [
    "module.exports = null;", "module.exports = String;", "module.exports = {};",
    "module.exports = { WasmStreamingKernel: 7 };",
    "module.exports = { WasmStreamingKernel: String };",
  ].entries()) {
    const broken = join(scratch, `incompatible-${index}.cjs`);
    writeFileSync(broken, source);
    assert.throws(() => benchmarkWasm(broken, 3, 1, configJSON), /WASM module/);
  }
});

test("corrupted generated session returns are refused through the real file consumer", () => {
  const original = readFileSync(modulePath, "utf8");
  const body = "const ret = wasm.wasmstreamingkernel_get_session(this.__wbg_ptr);\n        return ret;";
  assert.equal(original.split(body).length, 2);
  for (const name of ["package.json", "backfire_wasm.d.ts", "backfire_wasm_bg.wasm"]) {
    copyFileSync(join(dirname(modulePath), name), join(scratch, name));
  }
  for (const [index, value] of ["null", "42", "{}", "{ tokens: 0 }",
    "{ tokens: [] }", "{ tokens: new Array(3) }"].entries()) {
    const broken = join(scratch, `corrupted-session-${index}.cjs`);
    writeFileSync(broken, original.replace(body, `return ${value};`));
    assert.throws(() => benchmarkWasm(broken, 3, 1, configJSON), /full workload/);
  }
});

test("CLI arguments and the real Node executable retain native refusals", () => {
  for (const args of [[], [modulePath, "3", "2"], [modulePath, "3", "2", configJSON, "extra"]]) {
    assert.throws(() => main(args), /Expected module/);
  }
  assert.equal(main([modulePath, "3", "2", configJSON]).timings_ms.length, 2);
  const helper = fileURLToPath(new URL("./wasm_overhead_node.mjs", import.meta.url));
  const result = spawnSync(process.execPath, [helper, modulePath, "3", "2", configJSON],
    { encoding: "utf8", timeout: 20000 });
  assert.equal(result.status, 0, result.stderr);
  const value = JSON.parse(result.stdout);
  assert.equal(value.tokens, 3);
  assert.equal(value.timings_ms.length, 2);
  const refused = spawnSync(process.execPath, [helper], { encoding: "utf8", timeout: 20000 });
  assert.equal(refused.status, 1);
  assert.match(refused.stderr, /Expected module/);
});

test("Python CLI crosses all three real runtime boundaries and preserves an existing result", () => {
  const output = join(scratch, "comparison.json");
  const root = fileURLToPath(new URL("../", import.meta.url));
  const args = ["-B", "-m", "coverage", "run", "--rcfile", "benchmarks/streaming_comparison.coveragerc",
    "-m", "benchmarks.wasm_overhead_bench", "--wasm-module", modulePath,
    "--output", output, "--tokens", "3", "--iterations", "2"];
  execFileSync("python", args, { cwd: root, encoding: "utf8", timeout: 20000 });
  /** Actual CLI JSON is checked before its comparison fields are consumed. @type {unknown} */
  const result = JSON.parse(readFileSync(output, "utf8"));
  assert.ok(result !== null && typeof result === "object" && "implementations" in result &&
    "qualification" in result);
  assert.ok(Array.isArray(result.implementations));
  assert.equal(result.implementations.length, 3);
  assert.ok(result.qualification !== null && typeof result.qualification === "object" &&
    "isolated" in result.qualification);
  assert.equal(result.qualification.isolated, false);
  /** Each actual implementation packet remains unknown until its fields are checked. @type {unknown[]} */
  const entries = result.implementations;
  const names = [];
  for (const entry of entries) {
    assert.ok(entry !== null && typeof entry === "object" && "summary" in entry && "timings_ms" in entry);
    assert.ok(entry.summary !== null && typeof entry.summary === "object" && "name" in entry.summary);
    assert.ok(Array.isArray(entry.timings_ms));
    assert.equal(entry.timings_ms.length, 2);
    names.push(entry.summary.name);
  }
  assert.deepEqual(names, ["native_rust_ffi", "python_orchestration", "node_wasm"]);
  assert.ok("host" in result && result.host !== null && typeof result.host === "object" &&
    "native_module" in result.host && typeof result.host.native_module === "string" &&
    "native_module_sha256" in result.host);
  assert.match(result.host.native_module, /\.(so|pyd)$/);
  assert.equal(result.host.native_module_sha256,
    createHash("sha256").update(readFileSync(result.host.native_module)).digest("hex"));
  const original = readFileSync(output);
  const refused = spawnSync("python", args, { cwd: root, encoding: "utf8", timeout: 20000 });
  assert.equal(refused.status, 1);
  assert.match(refused.stderr, /FileExistsError/);
  assert.deepEqual(readFileSync(output), original);
});
