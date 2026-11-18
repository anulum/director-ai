// SPDX-License-Identifier: Apache-2.0
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// Director-AI Studio — example Learn lesson (content-as-data), grounded in the
// committed contradiction-halt. Demonstrates the content schema binding to a real
// artefact + a runnable WASM cell; the brainstorm reviews the pattern, not prose.

import type { Lesson } from "../schema";
import { DEFAULT_BACKFIRE_CONFIG } from "../../wasm/backfireConfig";

/** Teach the distinction between the Python claim check and WASM score gate. */
export const streamingHaltLesson: Lesson = {
  id: "streaming-contradiction-halt",
  title: "Streaming coherence and contradiction halts",
  track: "learn",
  artefact: "backfire-kernel/crates/backfire-wasm/pkg",
  summary:
    "The Python contradiction interlock checks completed claims against retrieved " +
    "facts. The browser exercise runs the WASM coherence kernel with supplied scores " +
    "to show how a token below the hard limit stops the stream.",
  cells: [
    {
      kind: "katex",
      source:
        "P_{\\mathrm{halt}} = \\max_{f \\in \\text{facts}} P(\\text{contradiction} \\mid f, \\text{claim})",
    },
    {
      kind: "wasm-halt",
      source: JSON.stringify(DEFAULT_BACKFIRE_CONFIG),
    },
  ],
  // Fact-checked against the shipped mechanism + its honest evidence caveat.
  verifiedAgainst:
    "src/director_ai/core/runtime/contradiction_halt.py; " +
    "benchmarks/results/streaming_contradiction_halt_base.json (not a sole production gate)",
};
