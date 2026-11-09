// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI — exact optional response parameter declaration correction

import { createHash } from "node:crypto";
import { closeSync, constants, fsyncSync, fstatSync, ftruncateSync,
  openSync, readFileSync, writeSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

/** SHA-256 of the original @types/k6 2.3.1 HTTP declaration, not a replacement host API. */
const originalDigest = "7217c7460337ab516395be64710b57e22a4ed63b28d0af996d68f828d3f5cd3d";
/** SHA-256 after excluding explicit undefined from the refined optional property. */
const correctedDigest = "78b4fff112685fc03aa038e5e76d47631b5ecdc3a05222d344cd7d28fec46d29";

/**
 * Apply the byte-bound correction to the actual locked publisher declaration.
 * @param {string} packageRoot Installed @types/k6 package directory.
 * @returns {{before:string, after:string, changed:boolean}} Original and resulting byte identities.
 */
export function prepareK6Types(packageRoot) {
  const path = join(packageRoot, "http", "index.d.ts");
  const fd = openSync(path, constants.O_RDWR | constants.O_NOFOLLOW | constants.O_NONBLOCK);
  try {
    if (!fstatSync(fd).isFile()) throw new Error("The k6 declaration must be a regular file");
    const original = readFileSync(fd);
    const before = createHash("sha256").update(original).digest("hex");
    if (before === correctedDigest) return { before, after: before, changed: false };
    if (before !== originalDigest) throw new Error("Unqualified k6 declaration bytes");
    const corrected = Buffer.from(original.toString("utf8").replace(
      "responseType?: RT;", "responseType?: Exclude<RT, undefined>;"));
    let offset = 0;
    while (offset < corrected.length) {
      offset += writeSync(fd, corrected, offset, corrected.length - offset, offset);
    }
    ftruncateSync(fd, corrected.length);
    fsyncSync(fd);
    return { before, after: createHash("sha256").update(corrected).digest("hex"), changed: true };
  } finally {
    closeSync(fd);
  }
}

/**
 * Prepare the pinned native-host types or refuse without hiding the failure.
 * @param {string[]} args Optional actual installed package directory.
 * @returns {number} Zero for qualified bytes, two for invalid invocation or input.
 */
export function main(args) {
  try {
    if (args.length > 1) throw new Error("One installed package directory is supported");
    const root = args[0] ?? fileURLToPath(new URL("../node_modules/@types/k6", import.meta.url));
    console.log(JSON.stringify(prepareK6Types(root)));
    return 0;
  } catch {
    console.error("The locked k6 declaration correction refused.");
    return 2;
  }
}

if (import.meta.main) process.exitCode = main(process.argv.slice(2));
