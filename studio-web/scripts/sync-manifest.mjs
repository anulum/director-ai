// SPDX-License-Identifier: Apache-2.0
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// Copy the committed schema-A studio manifest into public/ so the portal serves
// it as a static asset. The producer of record is tools/emit_studio_manifest.py
// (drift-gated in CI); this only stages the generated artefact for the dev server
// and the static build — never an alternative source of truth.

import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

/** Script directory used so staging is independent of the working directory. */
const here = dirname(fileURLToPath(import.meta.url));
/** Committed output of the producer's manifest emitter. */
const src = resolve(here, "..", "..", "docs", "_generated", "studio_manifest.json");
/** Static asset consumed by the standalone portal and its federated panel. */
const dest = resolve(here, "..", "public", "studio_manifest.json");

mkdirSync(dirname(dest), { recursive: true });
copyFileSync(src, dest);
console.log(`synced ${src} -> ${dest}`);
