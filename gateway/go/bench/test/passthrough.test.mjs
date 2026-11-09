// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI — native k6 and full strict benchmark consumer contracts

import assert from "node:assert/strict";
import { execFile, execFileSync, spawnSync } from "node:child_process";
import { once } from "node:events";
import { createServer } from "node:http";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import test from "node:test";

/** Actual pinned k6 binary supplied by the caller, without automatic installation or cloud use. */
const k6 = process.env.K6_BINARY;
assert.ok(k6, "K6_BINARY must select the verified native k6 2.3.0 executable");
/** Original maintained benchmark; it executes through the real k6 runtime. */
const source = fileURLToPath(new URL("../passthrough.js", import.meta.url));
/** Root of the real locked compiler and host declarations. */
const bench = fileURLToPath(new URL("../", import.meta.url));
/** Actual compiler executable from this locked dependency graph. */
const compiler = join(bench, "node_modules", "typescript", "bin", "tsc");
/** Retained actual control inputs, diagnostics and native summary within caller TMPDIR. */
const outputs = mkdtempSync(join(tmpdir(), "director-k6-contract-"));
console.log(JSON.stringify({ nativeContractOutputs: outputs }));

assert.match(execFileSync(k6, ["version"], { encoding: "utf8", timeout: 10000 }),
  /^k6 v2\.3\.0 \(commit\/e088784614,/);

test("native k6 resolves original imports and exports the exact schedule", () => {
  const native = spawnSync(k6, ["inspect", "--execution-requirements", source],
    { encoding: "utf8", timeout: 10000 });
  assert.equal(native.status, 0, native.stderr);
  const options = JSON.parse(native.stdout);
  assert.equal(options.vus, 50);
  assert.equal(options.duration, "30s");
  assert.deepEqual(options.thresholds.http_req_failed, ["rate<0.01"]);
  console.log(JSON.stringify({ native: "inspect", stdout: native.stdout, stderr: native.stderr }));
});

test("real strict compiler and native host both reject a nonexistent k6 module", () => {
  const bad = join(outputs, "invalid-host-import.js");
  writeFileSync(bad, readFileSync(source, "utf8").replace("'k6/http'", "'k6/director-nonexistent-host-module'"));
  const config = JSON.parse(readFileSync(join(bench, "tsconfig.json"), "utf8"));
  config.compilerOptions.typeRoots = [join(bench, "node_modules", "@types")];
  config.files = [bad];
  const configPath = join(outputs, "invalid-host-import.json");
  writeFileSync(configPath, JSON.stringify(config));
  const strict = spawnSync(process.execPath, [compiler, "--project", configPath],
    { encoding: "utf8", timeout: 60000 });
  assert.equal(strict.status, 2, strict.stderr);
  assert.match(strict.stdout, /invalid-host-import\.js.*TS2307/);
  const native = spawnSync(k6, ["inspect", bad], { encoding: "utf8", timeout: 10000 });
  assert.equal(native.status, 107, native.stderr);
  assert.match(native.stderr, /k6\/director-nonexistent-host-module/);
  console.log(JSON.stringify({ native: "invalid-host-import", strict, runtime: native }));
});

test("real publisher response types reject an invalid response member", () => {
  const bad = join(outputs, "invalid-response-member.js");
  writeFileSync(bad, readFileSync(source, "utf8").replace("r.status < 500", "r.directorUnknown < 500"));
  const config = JSON.parse(readFileSync(join(bench, "tsconfig.json"), "utf8"));
  config.compilerOptions.typeRoots = [join(bench, "node_modules", "@types")];
  config.files = [bad];
  const configPath = join(outputs, "invalid-response-member.json");
  writeFileSync(configPath, JSON.stringify(config));
  const strict = spawnSync(process.execPath, [compiler, "--project", configPath],
    { encoding: "utf8", timeout: 60000 });
  assert.equal(strict.status, 2, strict.stderr);
  assert.match(strict.stdout, /invalid-response-member\.js.*TS2339/);
  console.log(JSON.stringify({ native: "invalid-response-member", result: strict }));
});

for (const apiKey of ["local-native-contract", ""]) {
  test(`actual native benchmark sends one checked request with key length ${apiKey.length}`, async (context) => {
  /** Actual request observations from the loopback system of record. @type {{url:string|undefined,key:string|string[]|undefined,body:string}[]} */
  const requests = [];
  const server = createServer((request, response) => {
    let body = "";
    request.setEncoding("utf8");
    request.on("data", (chunk) => { body += chunk; });
    request.on("end", () => {
      requests.push({ url: request.url, key: request.headers["x-api-key"], body });
      response.writeHead(200, { "Content-Type": "application/json" });
      response.end(JSON.stringify({ choices: [{ message: { content: "4" } }] }));
    });
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  context.after(async () => await server[Symbol.asyncDispose]());
  const address = /** @type {import("node:net").AddressInfo} */ (server.address());
  const summary = join(outputs, `actual-native-summary-${apiKey.length}.json`);
  const native = await promisify(execFile)(k6,
    ["run", "--vus", "1", "--iterations", "1", "--summary-export", summary, source],
    { encoding: "utf8", timeout: 20000, env: { ...process.env,
      DIRECTOR_URL: `http://127.0.0.1:${address.port}`, DIRECTOR_KEY: apiKey,
      DIRECTOR_VUS: "1", DIRECTOR_DURATION: "1s" } });
  assert.equal(requests.length, 1);
  assert.deepEqual(requests[0], { url: "/v1/chat/completions", key: apiKey || undefined,
    body: JSON.stringify({ model: "gpt-4o-mini", messages: [{ role: "user", content: "What is 2+2?" }], stream: false }) });
  const actualSummary = JSON.parse(readFileSync(summary, "utf8"));
  assert.equal(actualSummary.metrics.http_reqs.count, 1);
  assert.deepEqual(actualSummary.root_group.checks["status is not 5xx"].passes, 1);
  assert.deepEqual(actualSummary.root_group.checks["status is not 5xx"].fails, 0);
  console.log(JSON.stringify({ native: "one-real-http-request", stdout: native.stdout,
    stderr: native.stderr, summary: actualSummary, requests }));
  });
}
