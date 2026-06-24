// SPDX-License-Identifier: Apache-2.0
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// Director-AI Studio — contract-consumption tests (forward-tolerance is the point).

import { describe, expect, it } from "vitest";
import { isEraSupported, parseManifest } from "./manifest";

const VALID = {
  contract_era: "v1",
  protocol_version: "1",
  transport_profile: "local-first",
  studio: "director-ai",
  studio_version: "3.16.0",
  platform_sdk: ">=0.1,<0.2",
  enumeration: "language-agnostic",
  evidence_types: ["studio.streaming-halt.v1"],
  verbs: [
    {
      verb: "halt",
      safety_tier: "research",
      side_effect: "read-only",
      timing: { class: "realtime" },
      produces: ["studio.streaming-halt.v1"],
      backends: ["python", "rust"],
    },
  ],
  ui_module: {
    remote_entry: "/studio/remoteEntry.js",
    exposes: ["./DirectorAIStudioPanel"],
    federation: "module-federation-2",
  },
  content_digest: "sha256:abc",
};

describe("parseManifest", () => {
  it("parses a well-formed manifest", () => {
    const m = parseManifest(VALID);
    expect(m.studio).toBe("director-ai");
    expect(m.verbs).toHaveLength(1);
    expect(m.verbs[0]?.timing.class).toBe("realtime");
    expect(m.ui_module?.exposes).toEqual(["./DirectorAIStudioPanel"]);
    expect(isEraSupported(m)).toBe(true);
  });

  it("is forward-tolerant: unknown top-level fields go to extra, never thrown", () => {
    const m = parseManifest({ ...VALID, future_field: { x: 1 } });
    expect(m.extra["future_field"]).toEqual({ x: 1 });
  });

  it("tolerates a future era without crashing", () => {
    const m = parseManifest({ ...VALID, contract_era: "v2" });
    expect(isEraSupported(m)).toBe(false);
    expect(m.contract_era).toBe("v2");
  });

  it("falls back safely on missing/garbage fields", () => {
    const m = parseManifest({ verbs: "not-an-array" });
    expect(m.studio).toBe("unknown");
    expect(m.verbs).toEqual([]);
    expect(m.evidence_types).toEqual([]);
  });

  it("drops non-string verb fields to safe defaults", () => {
    const m = parseManifest({ ...VALID, verbs: [{ verb: 42 }] });
    expect(m.verbs[0]?.verb).toBe("unknown");
    expect(m.verbs[0]?.safety_tier).toBe("research");
  });
});
