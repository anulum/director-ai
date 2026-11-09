// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI Studio — schema-A manifest consumption

// The portal consumes the committed, versioned `studio_manifest.json` produced by
// `tools/emit_studio_manifest.py` and NEVER reaches into Python internals. Parsing
// is forward-tolerant: unknown verbs remain in the capability list and unknown
// fields remain available in extra rather than being silently discarded.

/** Producer-declared assurance tier; consuming a manifest does not certify it. */
export type SafetyTier = "research" | "certified" | "production";
/** Whether a verb observes, simulates or affects external hardware. */
export type SideEffect = "read-only" | "simulated" | "live-hardware";
/** Scheduling class advertised by the producer for a capability. */
export type TimingClass = "batch" | "interactive" | "realtime";

/** A producer capability with conservative defaults for absent metadata. */
export interface Verb {
    /** Stable capability identifier advertised by the producer. */
    readonly verb: string;
    /** Producer assurance claim; absent declarations default to research. */
    readonly safety_tier: SafetyTier;
    /** Declared external effects, defaulting to read-only consumption. */
    readonly side_effect: SideEffect;
    /** Producer scheduling metadata; absent timing defaults to batch. */
    readonly timing: {
        /** Scheduling class, preserving future string values at the boundary. */
        readonly class: TimingClass;
    };
    /** Evidence formats this capability can emit. */
    readonly produces: readonly string[];
    /** Runtime implementations declared for this capability. */
    readonly backends: readonly string[];
    /** Optional producer fidelity label, absent unless it is a string. */
    readonly fidelity?: string;
}

/** Module-federation entry and exposed components advertised by the studio. */
export interface UiModule {
    /** Remote component entry URL, resolved by the consuming portal. */
    readonly remote_entry: string;
    /** Module-federation export identifiers available at the entry URL. */
    readonly exposes: readonly string[];
    /** Federation protocol label, defaulting to module-federation-2. */
    readonly federation: string;
}

/** Schema-A capabilities and preserved extension metadata consumed by the UI. */
export interface StudioManifest {
    /** Contract era used to decide whether to display a compatibility warning. */
    readonly contract_era: string;
    /** Producer's protocol revision, retained without compatibility promotion. */
    readonly protocol_version: string;
    /** Transport profile advertised by the producer. */
    readonly transport_profile: string;
    /** Studio identity from schema-A, never from outer envelope metadata. */
    readonly studio: string;
    /** Producer release stamp, or 0+unknown when absent. */
    readonly studio_version: string;
    /** Required platform SDK range advertised by the producer. */
    readonly platform_sdk: string;
    /** Capability enumeration convention, defaulting to language-agnostic. */
    readonly enumeration: string;
    /** Recognised string-valued evidence format identifiers. */
    readonly evidence_types: readonly string[];
    /** Producer capabilities, including future identifiers and timing labels. */
    readonly verbs: readonly Verb[];
    /** Optional remote UI declaration, present when supplied by the producer. */
    readonly ui_module?: UiModule;
    /** Producer digest, preserved here without claiming independent verification. */
    readonly content_digest: string;
    /** Forward-tolerance: any field the producer adds after this build lands here. */
    readonly extra: Readonly<Record<string, unknown>>;
}

/** The contract era this portal build was written against. */
export const SUPPORTED_ERA = "v1";

/** Schema-A fields consumed below; every other member remains extension data. */
const KNOWN_KEYS = new Set([
    "contract_era",
    "protocol_version",
    "transport_profile",
    "studio",
    "studio_version",
    "platform_sdk",
    "enumeration",
    "evidence_types",
    "verbs",
    "ui_module",
    "content_digest",
]);

/** Retain object members; scalar or null inputs supply no manifest fields. */
function asRecord(value: unknown): Record<string, unknown> {
    return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

/** Read a producer string without coercing a malformed scalar into an identity. */
function asString(value: unknown, fallback = ""): string {
    return typeof value === "string" ? value : fallback;
}

/** Keep string elements in producer order, defaulting malformed lists to empty. */
function asStringList(value: unknown): string[] {
    return Array.isArray(value) ? value.filter((v): v is string => typeof v === "string") : [];
}

/** Parse a capability while defaulting absent or ill-typed metadata conservatively. */
function parseVerb(raw: unknown): Verb {
    const r = asRecord(raw);
    const timing = asRecord(r["timing"]);
    const verb: Verb = {
        verb: asString(r["verb"], "unknown"),
        safety_tier: asString(r["safety_tier"], "research") as SafetyTier,
        side_effect: asString(r["side_effect"], "read-only") as SideEffect,
        timing: { class: asString(timing["class"], "batch") as TimingClass },
        produces: asStringList(r["produces"]),
        backends: asStringList(r["backends"]),
    };
    const fidelity = r["fidelity"];
    return typeof fidelity === "string" ? { ...verb, fidelity } : verb;
}

/**
 * Parse a federation envelope or a flat schema-A payload into a typed manifest.
 *
 * Forward-tolerant by construction: missing scalars fall back to safe defaults.
 * Unknown envelope fields stay in `extra`; unknown nested schema fields stay in
 * `extra.schema_a`. Flat payloads retain their existing `extra` shape. The own
 * `schema_a` field selects the envelope even when malformed, so outer fields
 * cannot impersonate missing nested fields. Malformed nested values remain
 * unchanged in `extra.schema_a`. Unexpected eras remain available
 * for the UI to warn. Parsing never mutates its input.
 *
 * @param raw Decoded JSON from the manifest producer or a legacy flat payload.
 * @returns Typed fields with safe defaults and unmodified unknown metadata.
 */
export function parseManifest(raw: unknown): StudioManifest {
    const envelope = asRecord(raw);
    const wrapped = Object.hasOwn(envelope, "schema_a");
    const nested = envelope["schema_a"];
    const nestedIsObject = nested !== null && typeof nested === "object" && !Array.isArray(nested);
    const r = wrapped ? asRecord(nestedIsObject ? nested : undefined) : envelope;
    const uiRaw = r["ui_module"];
    const ui = asRecord(uiRaw);
    const extra = Object.fromEntries(
        Object.entries(r).filter(([key]) => !KNOWN_KEYS.has(key)),
    );
    const verbsRaw = r["verbs"];
    const manifest: StudioManifest = {
        contract_era: asString(r["contract_era"], "unknown"),
        protocol_version: asString(r["protocol_version"], "0"),
        transport_profile: asString(r["transport_profile"], "local-first"),
        studio: asString(r["studio"], "unknown"),
        studio_version: asString(r["studio_version"], "0+unknown"),
        platform_sdk: asString(r["platform_sdk"]),
        enumeration: asString(r["enumeration"], "language-agnostic"),
        evidence_types: asStringList(r["evidence_types"]),
        verbs: Array.isArray(verbsRaw) ? verbsRaw.map(parseVerb) : [],
        content_digest: asString(r["content_digest"]),
        extra: wrapped ? { ...envelope, schema_a: nestedIsObject ? extra : nested } : extra,
    };
    if (uiRaw !== undefined) {
        return {
            ...manifest,
            ui_module: {
                remote_entry: asString(ui["remote_entry"]),
                exposes: asStringList(ui["exposes"]),
                federation: asString(ui["federation"], "module-federation-2"),
            },
        };
    }
    return manifest;
}

/**
 * Compare the producer's era with the contract understood by this build.
 * @param manifest Parsed producer contract; unsupported eras remain inspectable.
 * @returns Whether this build recognises the advertised era.
 */
export function isEraSupported(manifest: StudioManifest): boolean {
    return manifest.contract_era === SUPPORTED_ERA;
}

/**
 * Fetch the staged manifest and parse its envelope or legacy flat schema.
 * @param url Public manifest URL, relative to the document in a browser.
 * @returns Producer capabilities with the same extension custody as parsing.
 * @throws Error When HTTP fails; fetch and JSON decoding failures also propagate.
 */
export async function loadManifest(
    url = "/studio_manifest.json",
): Promise<StudioManifest> {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`studio_manifest.json unavailable (HTTP ${response.status})`);
    }
    return parseManifest(await response.json());
}
