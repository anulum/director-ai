// SPDX-License-Identifier: Apache-2.0
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// Director-AI Studio — schema-A contract-consumption layer (the only seam).
//
// The portal consumes the committed, versioned `studio_manifest.json` produced by
// `tools/emit_studio_manifest.py` and NEVER reaches into Python internals. Parsing
// is forward-tolerant: unknown verbs/fields are preserved and rendered at the
// boundary, never thrown on — an additive contract change (a new verb, a new
// field) must never break a deployed portal (research §4.1 / §7.7).

export type SafetyTier = "research" | "certified" | "production";
export type SideEffect = "read-only" | "simulated" | "live-hardware";
export type TimingClass = "batch" | "interactive" | "realtime";

export interface Verb {
  readonly verb: string;
  readonly safety_tier: SafetyTier;
  readonly side_effect: SideEffect;
  readonly timing: { readonly class: TimingClass };
  readonly produces: readonly string[];
  readonly backends: readonly string[];
  readonly fidelity?: string;
}

export interface UiModule {
  readonly remote_entry: string;
  readonly exposes: readonly string[];
  readonly federation: string;
}

export interface StudioManifest {
  readonly contract_era: string;
  readonly protocol_version: string;
  readonly transport_profile: string;
  readonly studio: string;
  readonly studio_version: string;
  readonly platform_sdk: string;
  readonly enumeration: string;
  readonly evidence_types: readonly string[];
  readonly verbs: readonly Verb[];
  readonly ui_module?: UiModule;
  readonly content_digest: string;
  /** Forward-tolerance: any field the producer adds after this build lands here. */
  readonly extra: Readonly<Record<string, unknown>>;
}

/** The contract era this portal build was written against. */
export const SUPPORTED_ERA = "v1";

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

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function asStringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((v): v is string => typeof v === "string") : [];
}

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
 * Parse a raw `studio_manifest.json` payload into a typed manifest.
 *
 * Forward-tolerant by construction: missing scalars fall back to safe defaults,
 * unknown top-level fields are collected into `extra` (rendered at the boundary,
 * never dropped silently), and an unexpected `contract_era` is preserved so the
 * UI can warn rather than crash. It never throws on a well-formed JSON object.
 */
export function parseManifest(raw: unknown): StudioManifest {
  const r = asRecord(raw);
  const uiRaw = r["ui_module"];
  const ui = asRecord(uiRaw);
  const extra: Record<string, unknown> = {};
  for (const key of Object.keys(r)) {
    if (!KNOWN_KEYS.has(key)) extra[key] = r[key];
  }
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
    extra,
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

/** Whether the manifest's era matches the era this portal build supports. */
export function isEraSupported(manifest: StudioManifest): boolean {
  return manifest.contract_era === SUPPORTED_ERA;
}

/** Load + parse the committed manifest from a URL (defaults to the public copy). */
export async function loadManifest(
  url = "/studio_manifest.json",
): Promise<StudioManifest> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`studio_manifest.json unavailable (HTTP ${response.status})`);
  }
  return parseManifest(await response.json());
}
