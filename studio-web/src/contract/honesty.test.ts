// SPDX-License-Identifier: Apache-2.0
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// Director-AI Studio — honesty bridge tests (H1–H4 enforced, not decorative).

import { describe, expect, it } from "vitest";
import {
  HONESTY_PALETTE,
  gradeStatus,
  judgeReproduction,
  reproductionLabel,
} from "./honesty";

describe("gradeStatus (H4 single source of grading)", () => {
  it("maps producer statuses onto canonical statuses", () => {
    expect(gradeStatus("reference-validated")).toBe("validated");
    expect(gradeStatus("bounded-support")).toBe("bounded");
    expect(gradeStatus("toolchain-gated")).toBe("roadmap");
    expect(gradeStatus("falsified")).toBe("falsified");
    expect(gradeStatus("refuted")).toBe("refuted");
  });

  it("fails closed to unknown for an unrecognised status", () => {
    expect(gradeStatus("totally-made-up")).toBe("unknown");
  });

  it("keeps falsified/refuted first-class in the palette", () => {
    expect(HONESTY_PALETTE.falsified).toBe("#f87171");
    expect(HONESTY_PALETTE.validated).toBe("#34d399");
    expect(HONESTY_PALETTE.bounded).toBe("#fbbf24");
  });
});

describe("judgeReproduction (H1/H2/H3)", () => {
  it("matches only on equal digests (H2)", () => {
    const ok = judgeReproduction("sha256:a", "sha256:a", "bit-exact");
    expect(ok.matches).toBe(true);
    const drift = judgeReproduction("sha256:a", "sha256:b", "bit-exact");
    expect(drift.matches).toBe(false);
  });

  it("labels a mismatch loudly as drift (H2)", () => {
    const drift = judgeReproduction("sha256:a", "sha256:b", "tolerance-aware");
    expect(reproductionLabel(drift)).toContain("MISMATCH");
  });

  it("never claims bit-exact for a tolerance match (H3)", () => {
    const tol = judgeReproduction("sha256:a", "sha256:a", "tolerance-aware");
    expect(reproductionLabel(tol)).toContain("within tolerance");
    expect(reproductionLabel(tol)).not.toContain("bit-exact");
  });

  it("uses consumer-side wording, never 'verified' (H1)", () => {
    const ok = judgeReproduction("sha256:a", "sha256:a", "bit-exact");
    expect(reproductionLabel(ok)).toContain("reproduced in your browser");
    expect(reproductionLabel(ok).toLowerCase()).not.toContain("verified-at-source");
  });
});
