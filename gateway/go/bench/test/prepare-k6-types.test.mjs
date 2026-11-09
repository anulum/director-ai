// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI — actual publisher declaration correction contracts

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { main, prepareK6Types } from "../tools/prepare-k6-types.mjs";

/** Actual installed locked package, never a fabricated host definition. */
const packageRoot = fileURLToPath(new URL("../node_modules/@types/k6", import.meta.url));
/** The publisher declaration selected by the production command. */
const declaration = join(packageRoot, "http", "index.d.ts");
/** Exact corrected bytes, already prepared by the package test command. */
const corrected = readFileSync(declaration);
/** Byte-exact original reconstructed by reversing only the named one-line correction. */
const original = Buffer.from(corrected.toString("utf8").replace(
  "responseType?: Exclude<RT, undefined>;", "responseType?: RT;"));
assert.equal(createHash("sha256").update(original).digest("hex"),
  "7217c7460337ab516395be64710b57e22a4ed63b28d0af996d68f828d3f5cd3d");
assert.equal(createHash("sha256").update(corrected).digest("hex"),
  "78b4fff112685fc03aa038e5e76d47631b5ecdc3a05222d344cd7d28fec46d29");

/** Actual public command whose separate CLI status is checked below. */
const command = fileURLToPath(new URL("../tools/prepare-k6-types.mjs", import.meta.url));

test("actual pinned publisher bytes are corrected and survive reopen", (context) => {
  writeFileSync(declaration, original);
  context.after(() => writeFileSync(declaration, corrected));
  assert.deepEqual(prepareK6Types(packageRoot), {
    before: "7217c7460337ab516395be64710b57e22a4ed63b28d0af996d68f828d3f5cd3d",
    after: "78b4fff112685fc03aa038e5e76d47631b5ecdc3a05222d344cd7d28fec46d29", changed: true,
  });
  assert.deepEqual(readFileSync(declaration), corrected);
});

test("exact corrected publisher bytes are idempotent", () => {
  assert.deepEqual(prepareK6Types(packageRoot), {
    before: "78b4fff112685fc03aa038e5e76d47631b5ecdc3a05222d344cd7d28fec46d29",
    after: "78b4fff112685fc03aa038e5e76d47631b5ecdc3a05222d344cd7d28fec46d29", changed: false,
  });
  assert.deepEqual(readFileSync(declaration), corrected);
});

test("changed declaration bytes refuse and remain available for diagnosis", (context) => {
  const corrupted = Buffer.from(corrected);
  corrupted[0] = 120;
  writeFileSync(declaration, corrupted);
  context.after(() => writeFileSync(declaration, corrected));
  assert.throws(() => prepareK6Types(packageRoot), /Unqualified k6 declaration bytes/);
  assert.deepEqual(readFileSync(declaration), corrupted);
});

test("symbolic publisher declaration refuses without altering its target", (context) => {
  const root = mkdtempSync(join(tmpdir(), "director-k6-link-"));
  context.after(() => rmSync(root, { recursive: true }));
  mkdirSync(join(root, "http"));
  symlinkSync(declaration, join(root, "http", "index.d.ts"));
  assert.throws(() => prepareK6Types(root), /ELOOP/);
  assert.deepEqual(readFileSync(declaration), corrected);
});

test("nonregular publisher input refuses without waiting for a pipe writer", (context) => {
  const root = mkdtempSync(join(tmpdir(), "director-k6-pipe-"));
  context.after(() => rmSync(root, { recursive: true }));
  mkdirSync(join(root, "http"));
  execFileSync("mkfifo", [join(root, "http", "index.d.ts")], { timeout: 5000 });
  assert.throws(() => prepareK6Types(root), /must be a regular file/);
});

test("public command accepts default and explicit actual package roots", () => {
  assert.equal(main([]), 0);
  assert.equal(main([packageRoot]), 0);
});

test("public command refuses extra arguments and missing installed inputs", () => {
  assert.equal(main([packageRoot, "unexpected"]), 2);
  assert.equal(main([join(packageRoot, "not-installed")]), 2);
});

test("actual CLI exposes native success and refusal statuses", () => {
  const success = spawnSync(process.execPath, [command], { encoding: "utf8", timeout: 10000 });
  assert.equal(success.status, 0, success.stderr);
  assert.equal(JSON.parse(success.stdout).changed, false);
  const refusal = spawnSync(process.execPath, [command, packageRoot, "extra"],
    { encoding: "utf8", timeout: 10000 });
  assert.equal(refusal.status, 2);
  assert.match(refusal.stderr, /correction refused/);
});
