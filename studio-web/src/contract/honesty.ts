// SPDX-License-Identifier: Apache-2.0
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// Director-AI Studio — honesty bridge (LOCAL MIRROR of @anulum/ui honesty.ts).
//
// H4 (CEO ruling, locked): this is the SINGLE source of claim grading; the portal
// never forms a second opinion. Until `@anulum/ui` publishes, this local mirror
// stands in with the canonical palette; swapping the import is the only change.
// `falsified`/`refuted` are first-class statuses, never hidden.

/** Canonical claim statuses (LOCK-4). `falsified`/`refuted` are first-class. */
export type ClaimStatus =
  | "validated"
  | "bounded"
  | "roadmap"
  | "refuted"
  | "falsified"
  | "unknown";

/** The canonical honesty palette — one set, never a second opinion (H4). */
export const HONESTY_PALETTE: Readonly<Record<ClaimStatus, string>> = {
  validated: "#34d399", // evidence meets the claim
  bounded: "#fbbf24", // true within a stated boundary
  roadmap: "#fbbf24", // declared, not yet evidenced
  refuted: "#f87171", // evidence contradicts the claim
  falsified: "#f87171", // promoted negative result (surfaced, not buried)
  unknown: "#94a3b8", // no admissible evidence
};

/** Map a producer status string onto a canonical status (fail-closed to unknown). */
export function gradeStatus(raw: string): ClaimStatus {
  switch (raw) {
    case "validated":
    case "reference-validated":
      return "validated";
    case "bounded-model":
    case "bounded-support":
    case "bounded":
      return "bounded";
    case "roadmap":
    case "toolchain-gated":
      return "roadmap";
    case "refuted":
      return "refuted";
    case "falsified":
      return "falsified";
    default:
      return "unknown";
  }
}

/** Exactness of a reproduction (H3): a tolerance-match is never labelled bit-exact. */
export type Exactness = "bit-exact" | "tolerance-aware";

/** Outcome of an in-browser reproduction — a CONSUMER signal only (H1). */
export interface ReproductionResult {
  /** True only when the WASM output digest equals the committed claim digest (H2). */
  readonly matches: boolean;
  /** Whether the match was exact or within tolerance; the UI states which (H3). */
  readonly exactness: Exactness;
  /** Committed claim digest (the producer-verified source of truth). */
  readonly claimDigest: string;
  /** Digest the kernel produced in the visitor's browser this run. */
  readonly reproducedDigest: string;
}

/**
 * Judge an in-browser reproduction against a committed claim digest.
 *
 * H1: the result is a *consumer* signal ("reproduced in your browser ✓"); it is
 * NEVER promoted to producer verification (`verified-at-source`) — a visitor's
 * browser cannot be hub-attested. H2: a digest mismatch is drift and must be
 * surfaced loudly. H3: the caller passes the claim's declared `exactness`; a
 * tolerance-aware claim is never rendered as "bit-exact reproduced".
 */
export function judgeReproduction(
  claimDigest: string,
  reproducedDigest: string,
  exactness: Exactness,
): ReproductionResult {
  return {
    matches: claimDigest === reproducedDigest,
    exactness,
    claimDigest,
    reproducedDigest,
  };
}

/** Human label for a reproduction — consumer-side wording only (H1). */
export function reproductionLabel(result: ReproductionResult): string {
  if (!result.matches) return "reproduction MISMATCH — drift detected";
  return result.exactness === "bit-exact"
    ? "reproduced in your browser ✓ (bit-exact)"
    : "reproduced in your browser ✓ (within tolerance)";
}
