// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI — native TypeScript documentation and command contracts

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync, mkdirSync, symlinkSync } from "node:fs";
import fs from "node:fs";
import { syncBuiltinESMExports } from "node:module";
import { tmpdir } from "node:os";
import { resolve, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { captureDocumentation } from "../tools/quality-ratchet.mjs";

/** Actual native helper command rather than substituted parser output. */
const command = fileURLToPath(new URL("../tools/quality-ratchet.mjs", import.meta.url));
/** Actual SDK root, including when the package is qualified independently. */
const repository = resolve(fileURLToPath(new URL("..", import.meta.url)));

/**
 * Retain exact native input and output for a source-format admission control.
 * Fixture directories remain owned until the external finite runner preserves custody.
 * @param {Record<string, string>} inputs Actual files passed to the native compiler.
 * @returns {string} Newly allocated private runtime directory.
 */
function inputRoot(inputs) {
  const root = mkdtempSync(join(tmpdir(), "quality-typescript-"));
  for (const [path, text] of Object.entries(inputs)) {
    mkdirSync(resolve(root, path, ".."), { recursive: true });
    writeFileSync(join(root, path), text);
  }
  return root;
}

/**
 * Preserve the original native carrier alongside its selected source bytes.
 * @param {string} root Owned source directory.
 * @param {string[]} paths Explicit source inputs.
 * @returns {ReturnType<typeof captureDocumentation>} Original native public result.
 */
function capture(root, paths) {
  const result = captureDocumentation(root, paths);
  writeFileSync(join(root, "original-report.json"), JSON.stringify(result));
  return result;
}

test("native parser qualifies actual SDK production, tests and its own helper", () => {
  const paths = [
    "src/index.ts",
    "test/director-ai-middleware.test.mjs",
    "test/director-ai-prompt.test.mjs",
    "test/review-http-server.mjs",
    "tools/quality-ratchet.mjs",
    "test/quality-ratchet.test.mjs",
  ];
  const report = captureDocumentation(repository, paths);
  assert.equal(report.complete, true);
  assert.deepEqual(report.findings, []);
  assert.equal(Object.keys(report.sources).length, paths.length);
  for (const path of paths) {
    assert.equal(report.sources[path],
      createHash("sha256").update(readFileSync(join(repository, path))).digest("hex"));
  }
});

test("native JSDoc descriptions include explanatory tags but reject type-only tags", () => {
  const root = inputRoot({
    "contract.ts": [
      "/** @type {number} Actual documented input. */",
      "const tagged = 1;",
      "/** @type {number} */",
      "const typeOnly = 2;",
      "/** Descriptive {@link tagged} contract. */",
      "const linked = 3;",
      "/** Group documents both original declarations. */",
      "const first = 4, second = 5;",
      "/** Last declaration remains undocumented by whitespace. */",
      "function local() { /** */ const transform = () => 1; return transform(); }",
    ].join("\n"),
  });
  const report = capture(root, ["contract.ts"]);
  assert.equal(report.complete, true);
  assert.deepEqual(report.findings.map(({ rule, line }) => [rule, line]), [
    ["declaration-documentation", 4], ["declaration-documentation", 10],
  ]);
  assert.ok(report.findings.every(({ column }) => column > 0));
});

test("actual native declaration forms, private members and signatures require documentation", () => {
  const root = inputRoot({
    "contract.ts": [
      "function operation() {}",
      "class Controller { private value = 1; constructor() {} method() {}",
      "get current() { return this.value; } set current(value: number) { this.value = value; } }",
      "interface Request { value: number; method(): void; (): void; new(): Request; }",
      "type Outcome = Request;",
      "enum Status { Ready }",
      "namespace Contract { export const value = 1; }",
      "const callback = function() {};",
      "function loop() { for (const item of [1]) { const data = item; } }",
    ].join("\n"),
  });
  const report = capture(root, ["contract.ts"]);
  assert.equal(report.complete, true);
  assert.equal(report.findings.length, 18);
  assert.ok(report.findings.every(({ rule }) => rule === "declaration-documentation"));
  assert.ok(report.findings.some(({ message }) => message.startsWith("ConstructSignature")));
  assert.ok(report.findings.some(({ message }) => message.startsWith("Constructor")));
});

test("native source comments expose every supported masking form", () => {
  const root = inputRoot({
    "contract.ts": [
      "// @ts-nocheck",
      "/** Documented declaration. */",
      "const value = 1;",
      "// @ts-ignore: retained actual comment",
      "/** Documented operation. */",
      "function operation() { return value; }",
      "// @ts-expect-error retained actual comment",
      "/** @hidden */",
      "/** @ignore */",
      "/**",
      " * @hidden",
      " */",
    ].join("\n"),
  });
  const report = capture(root, ["contract.ts"]);
  assert.equal(report.complete, true);
  assert.deepEqual(report.findings.filter(({ rule }) => rule === "source-directive")
    .map(({ line }) => line).sort((a, b) => a - b), [1, 4, 7, 8, 9, 10]);
});

test("strings, JSX text and prose are not source masking directives", () => {
  const root = inputRoot({
    "contract.tsx": [
      "/** String fixture with literal @ts-ignore text. */",
      "const value = '// @ts-ignore';",
      "/** JSX retains literal documentation-looking text. */",
      "const element = <div>// @ts-nocheck {/* Ordinary comment. */}</div>;",
      "/** Describes the @hidden token without applying it. */",
      "function operation() { return value; }",
      "for (const item of [1, 2]) { const data = item; }",
    ].join("\n"),
  });
  const report = capture(root, ["contract.tsx"]);
  assert.equal(report.complete, true);
  assert.deepEqual(report.findings, []);
});

test("native syntax errors retain original input digest and actual TS source locations", () => {
  const root = inputRoot({ "contract.ts": "/** Original broken declaration. */\nfunction operation( {" });
  const report = capture(root, ["contract.ts"]);
  assert.equal(report.complete, false);
  assert.ok(report.findings.length > 0);
  assert.ok(report.findings.every(({ path, rule, line, column }) =>
    path === "contract.ts" && rule.startsWith("TS") && line === 2 && column > 0));
  assert.equal(report.sources["contract.ts"],
    createHash("sha256").update(readFileSync(join(root, "contract.ts"))).digest("hex"));
});

test("public command preserves native success, findings, syntax refusal and version outcomes", () => {
  const root = inputRoot({
    "clean.ts": "/** Actual source declaration. */\nexport const value = 1;",
    "missing.ts": "export const value = 1;",
    "broken.ts": "function operation(",
  });
  /** Original public command controls and their native exit statuses. */
  const cases = [
    { args: [root, "clean.ts"], status: 0, text: '"findings":[]' },
    { args: [root, "missing.ts"], status: 1, text: "declaration-documentation" },
    { args: [root, "broken.ts"], status: 2, text: '"complete":false' },
    { args: ["--version"], status: 0, text: '"typescript":"6.0.3"' },
    { args: [], status: 2, text: "" },
  ];
  for (const [index, { args, status, text }] of cases.entries()) {
    const native = spawnSync(process.execPath, [command, ...args], { encoding: "utf8", timeout: 15000 });
    writeFileSync(join(root, "command-" + index + ".json"),
      JSON.stringify({ args, status: native.status, stdout: native.stdout, stderr: native.stderr }));
    assert.equal(native.status, status);
    assert.ok(native.stdout.includes(text));
    assert.equal(native.stderr, args.length === 0 ? "Native TypeScript documentation refused.\n" : "");
  }
});

test("noncanonical, empty, duplicate, unreadable and symlink inputs visibly refuse", () => {
  const root = inputRoot({ "contract.ts": "/** Actual source. */\nconst value = 1;",
    "unsupported.txt": "Original non-source input." });
  symlinkSync(join(root, "contract.ts"), join(root, "linked.ts"));
  /** Actual invalid input selections for the source-admission boundary. */
  const selections = [[], ["contract.ts", "contract.ts"], ["./contract.ts"],
    ["absent.ts"], ["linked.ts"], ["."], ["../contract.ts"], [join(root, "contract.ts")],
    ["unsupported.txt"]];
  for (const paths of selections) assert.throws(() => captureDocumentation(root, paths));
});

test("actual input mutation after original native qualification refuses unstable source", (t) => {
  const text = "/** Original admitted source. */\nexport const value = 1;";
  const root = inputRoot({ "contract.ts": text, "original-source.txt": text });
  const original = capture(root, ["contract.ts"]);
  assert.equal(original.complete, true);
  assert.deepEqual(original.findings, []);
  const path = join(root, "contract.ts");
  const originalRead = fs.readFileSync;
  let changed = false;
  t.mock.method(fs, "readFileSync",
    /**
     * Alter actual source bytes only after a real read; never substitute compiler output.
     * @param {Parameters<typeof fs.readFileSync>[0]} selected Original native input.
     * @param {Parameters<typeof fs.readFileSync>[1]} options Original native read options.
     * @returns {ReturnType<typeof fs.readFileSync>} Original bytes read before mutation.
     */
    (selected, options) => {
      const bytes = originalRead(selected, options);
      if (selected === path && !changed) {
        changed = true;
        writeFileSync(path, text + "\n/** Newly changed source. */\nexport const added = 2;");
      }
      return bytes;
    });
  syncBuiltinESMExports();
  try {
    assert.throws(() => captureDocumentation(root, ["contract.ts"]), /source changed/);
    assert.equal(changed, true);
    assert.notEqual(createHash("sha256").update(originalRead(path)).digest("hex"),
      original.sources["contract.ts"]);
  } finally {
    t.mock.restoreAll();
    syncBuiltinESMExports();
  }
});
