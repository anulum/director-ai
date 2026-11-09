// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI Studio — committed manifest and real HTTP contracts

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { once } from "node:events";
import { copyFile, mkdir, mkdtemp, readFile, rm } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";

import {
    isEraSupported,
    loadManifest,
    parseManifest,
} from "../src/contract/manifest.ts";

/** Resolve real producer inputs relative to this module, independently of cwd. */
const repository = fileURLToPath(new URL("../../", import.meta.url));
/** Original committed bytes used to verify the staging script without reserialising. */
const original = await readFile(join(repository, "docs/_generated/studio_manifest.json"));
/** Producer envelope used as the reference for retained federation metadata. */
const document = JSON.parse(original.toString("utf8"));
/** The actual nested producer contract, also used for flat-schema compatibility. */
const schema = document.schema_a;
/** Isolated output root for the unchanged staging script, removed after the run. */
const stage = await mkdtemp(join(tmpdir(), "studio-manifest-"));
after(() => rm(stage, { recursive: true }));

/**
 * Serve actual response bytes over loopback without replacing the runtime fetch.
 * @param {import("node:test").TestContext} context Native test lifecycle owner.
 * @param {string | Buffer} body Bytes returned by the actual HTTP server.
 * @param {number} [status] Deliberate HTTP response status.
 * @returns {Promise<string>} Absolute URL on an ephemeral loopback port.
 */
async function serve(context, body, status = 200) {
    const server = createServer((_request, response) => {
        response.writeHead(status, { "content-type": "application/json" });
        response.end(body);
    });
    context.after(() => new Promise((resolve) => {
        server.closeAllConnections();
        server.close(() => resolve(undefined));
    }));
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    const address = server.address();
    assert.ok(address && typeof address !== "string");
    return `http://127.0.0.1:${address.port}/studio_manifest.json`;
}

test("the actual staging script and HTTP loader preserve the committed schema", async (context) => {
    const script = join(stage, "studio-web/scripts/sync-manifest.mjs");
    const source = join(stage, "docs/_generated/studio_manifest.json");
    await mkdir(dirname(script), { recursive: true });
    await mkdir(dirname(source), { recursive: true });
    await copyFile(join(repository, "studio-web/scripts/sync-manifest.mjs"), script);
    await copyFile(join(repository, "docs/_generated/studio_manifest.json"), source);
    execFileSync(process.execPath, [script], { cwd: stage });
    const staged = await readFile(join(stage, "studio-web/public/studio_manifest.json"));
    assert.deepEqual(staged, original);
    const manifest = await loadManifest(await serve(context, staged));
    assert.equal(manifest.studio, "director-ai");
    assert.equal(manifest.contract_era, "v1");
    assert.equal(manifest.verbs.length, 8);
    assert.deepEqual(manifest.verbs, parseManifest(schema).verbs);
    assert.equal(manifest.content_digest, schema.content_digest);
    assert.equal(isEraSupported(manifest), true);
    assert.deepEqual(manifest.extra.architecture_map, document.architecture_map);
    assert.deepEqual(parseManifest(document), manifest);
});

test("flat manifests retain the existing public parser and HTTP result", async (context) => {
    const manifest = await loadManifest(await serve(context, JSON.stringify(schema)));
    assert.deepEqual(manifest, parseManifest(schema));
    assert.equal(manifest.studio, "director-ai");
    assert.equal(manifest.ui_module?.federation, "module-federation-2");
});

test("outer and nested unknown metadata survive independently of name collisions", async (context) => {
    const nested = { ...schema, future_field: { depth: "nested" }, architecture_map: "nested" };
    const envelope = { ...document, schema_a: nested, future_field: { depth: "outer" }, studio: "outer" };
    const manifest = await loadManifest(await serve(context, JSON.stringify(envelope)));
    assert.equal(manifest.studio, schema.studio);
    assert.deepEqual(manifest.extra, {
        ...envelope,
        schema_a: { future_field: { depth: "nested" }, architecture_map: "nested" },
    });
    const flat = parseManifest(nested);
    assert.deepEqual(flat.extra, { future_field: { depth: "nested" }, architecture_map: "nested" });
});

test("future contract eras and verbs remain visible without source mutation", () => {
    const future = { ...schema, contract_era: "v2", verbs: [...schema.verbs, { verb: "future", timing: { class: "future" } }] };
    const before = JSON.stringify(future);
    const manifest = parseManifest({ ...document, schema_a: future });
    assert.equal(manifest.contract_era, "v2");
    assert.equal(isEraSupported(manifest), false);
    assert.equal(manifest.verbs.at(-1)?.verb, "future");
    assert.equal(manifest.verbs.at(-1)?.timing.class, "future");
    assert.equal(JSON.stringify(future), before);
});

test("JSON metadata names remain own data rather than changing extension prototypes", async (context) => {
    const future = JSON.parse('{"__proto__":{"metadata":"retained"},"constructor":{"name":"future"}}');
    const flat = { ...schema, ...future };
    for (const wrapped of [false, true]) {
        const payload = wrapped ? { ...document, schema_a: flat } : flat;
        const manifest = await loadManifest(await serve(context, JSON.stringify(payload)));
        const extra = wrapped ? manifest.extra.schema_a : manifest.extra;
        assert.ok(extra && typeof extra === "object");
        assert.equal(Object.getPrototypeOf(extra), Object.prototype);
        assert.equal(Object.hasOwn(extra, "__proto__"), true);
        assert.equal(Object.hasOwn(extra, "constructor"), true);
        assert.deepEqual(Object.getOwnPropertyDescriptor(extra, "__proto__")?.value, future.__proto__);
        assert.deepEqual(Object.getOwnPropertyDescriptor(extra, "constructor")?.value, future.constructor);
        assert.equal(manifest.studio, schema.studio);
    }
});

test("malformed schema members retain original data and never inherit outer authority", async (context) => {
    for (const raw of [null, false, 7, "not-an-object", undefined, [], ["x", "y"]]) {
        const manifest = parseManifest({ ...document, studio: "wrong-outer-studio", schema_a: raw });
        assert.equal(manifest.studio, "unknown");
        assert.equal(manifest.contract_era, "unknown");
        assert.deepEqual(manifest.verbs, []);
        assert.equal(isEraSupported(manifest), false);
        assert.equal(Object.hasOwn(manifest.extra, "schema_a"), true);
        assert.strictEqual(manifest.extra.schema_a, raw);
        // Undefined is an API-only member: JSON would omit it entirely.
        if (raw !== undefined) {
            const payload = { ...document, studio: "wrong-outer-studio", schema_a: raw };
            const loaded = await loadManifest(await serve(context, JSON.stringify(payload)));
            assert.equal(loaded.studio, "unknown");
            assert.equal(loaded.contract_era, "unknown");
            assert.deepEqual(loaded.verbs, []);
            assert.equal(Object.hasOwn(loaded.extra, "schema_a"), true);
            assert.deepEqual(loaded.extra.schema_a, raw);
        }
    }
    for (const raw of [null, false, 7, "not-an-object", undefined, []]) {
        const manifest = parseManifest(raw);
        assert.equal(manifest.studio, "unknown");
        assert.deepEqual(manifest.verbs, []);
        assert.deepEqual(manifest.evidence_types, []);
        assert.equal(manifest.ui_module, undefined);
    }
});

test("actual schema fields preserve typed lists, fidelity and safe fallback values", () => {
    const manifest = parseManifest({
        ...schema,
        evidence_types: [schema.evidence_types[0], null, 3],
        verbs: [
            ...schema.verbs,
            { verb: "future", fidelity: "exact", timing: null, produces: ["artifact", 4], backends: false },
            { verb: 42, safety_tier: null, side_effect: null, timing: { class: null }, fidelity: 3 },
            null,
        ],
        ui_module: { remote_entry: null, exposes: ["./panel", 9] },
    });
    assert.deepEqual(manifest.evidence_types, [schema.evidence_types[0]]);
    assert.deepEqual(manifest.verbs.at(-3), {
        verb: "future", safety_tier: "research", side_effect: "read-only",
        timing: { class: "batch" }, produces: ["artifact"], backends: [], fidelity: "exact",
    });
    assert.deepEqual(manifest.verbs.at(-2), {
        verb: "unknown", safety_tier: "research", side_effect: "read-only",
        timing: { class: "batch" }, produces: [], backends: [],
    });
    assert.deepEqual(manifest.verbs.at(-1), manifest.verbs.at(-2));
    assert.deepEqual(manifest.ui_module, { remote_entry: "", exposes: ["./panel"], federation: "module-federation-2" });
    assert.equal(parseManifest({ verbs: "invalid", ui_module: null }).ui_module?.remote_entry, "");
});

test("a non-success HTTP status is refused before parsing an otherwise valid body", async (context) => {
    await assert.rejects(loadManifest(await serve(context, original, 503)), {
        message: "studio_manifest.json unavailable (HTTP 503)",
    });
});

test("malformed actual HTTP JSON does not produce a usable manifest", async (context) => {
    await assert.rejects(loadManifest(await serve(context, "{broken-json")));
});

test("a closed actual HTTP endpoint refuses rather than returning an empty manifest", async () => {
    const server = createServer();
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    const address = server.address();
    assert.ok(address && typeof address !== "string");
    const closed = once(server, "close");
    server.close();
    await closed;
    await assert.rejects(loadManifest(`http://127.0.0.1:${address.port}/studio_manifest.json`));
});
