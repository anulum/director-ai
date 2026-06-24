// SPDX-License-Identifier: Apache-2.0
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// Director-AI Studio — example Learn lesson (content-as-data), grounded in the
// committed contradiction-halt. Demonstrates the content schema binding to a real
// artefact + a runnable WASM cell; the brainstorm reviews the pattern, not prose.

import type { Lesson } from "../schema";

export const streamingHaltLesson: Lesson = {
  id: "streaming-contradiction-halt",
  title: "How the contradiction-driven streaming halt works",
  track: "learn",
  // The lesson teaches from the real engine the science ran.
  artefact: "backfire-kernel/crates/backfire-wasm/pkg",
  summary:
    "A completed streamed claim is scored for contradiction against the retrieved " +
    "grounding facts; when P(contradiction) crosses the threshold the stream halts " +
    "on that token. A correct-but-unsupported claim is neutral, not a contradiction, " +
    "so it does not halt — only a claim that contradicts a governed fact does.",
  cells: [
    {
      kind: "katex",
      source:
        "P_{\\mathrm{halt}} = \\max_{f \\in \\text{facts}} P(\\text{contradiction} \\mid f, \\text{claim})",
    },
    {
      // Runs the committed WasmStreamingKernel live in the browser (Play seam).
      kind: "wasm-halt",
      source: JSON.stringify({ threshold: 0.65 }),
    },
  ],
  // Fact-checked against the shipped mechanism + its honest evidence caveat.
  verifiedAgainst:
    "src/director_ai/core/runtime/contradiction_halt.py; " +
    "benchmarks/results/streaming_contradiction_halt_base.json (not a sole production gate)",
};
