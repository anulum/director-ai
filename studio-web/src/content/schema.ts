// SPDX-License-Identifier: Apache-2.0
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// Director-AI Studio — content-as-data schema (seam §4.5 / §7.5).
//
// Lessons and sections are DATA, not code: structured content that BINDS to a
// committed artefact at build/run time. New science → new content + new artefact,
// zero portal-architecture change. A lesson may carry runnable cells (the WASM
// kernel / a future Pyodide cell) and always cites the artefact it teaches from.

import type { Exactness } from "../contract/honesty";

/** A runnable cell embedded in a lesson — the reproducibility unit. */
export interface RunnableCell {
  /** Selects how the lesson presents or runs the cell source. */
  readonly kind: "wasm-halt" | "katex" | "pyodide" | "chart";
  /** KaTeX source, a kernel config, or a chart artefact path — kind-dependent. */
  readonly source: string;
  /** For reproduction cells: the committed claim digest + its declared exactness. */
  readonly claim?: {
    /** Expected digest from the committed claim. */
    readonly digest: string;
    /** Precision level declared by the producer of the claim. */
    readonly exactness: Exactness;
  };
}

/** One lesson in a learning path — content bound to a committed artefact. */
export interface Lesson {
  /** Stable content identifier within a learning path. */
  readonly id: string;
  /** Heading displayed above the lesson summary. */
  readonly title: string;
  /** The section spine slot this lesson belongs to. */
  readonly track: "learn" | "play" | "audit";
  /** Repo-relative path of the artefact this lesson is grounded in. */
  readonly artefact: string;
  /** Short prose (MDX would expand this; kept as data for i18n-readiness). */
  readonly summary: string;
  /** Cells presented in author-provided order. */
  readonly cells: readonly RunnableCell[];
  /** Fact-check provenance: the source the content is verified against. */
  readonly verifiedAgainst: string;
}

/** A named learning path — an ordered list of lessons. */
export interface LearningPath {
  /** Stable identifier for the sequence of lessons. */
  readonly id: string;
  /** Display name of the learning path. */
  readonly title: string;
  /** Lessons in the order a reader follows them. */
  readonly lessons: readonly Lesson[];
}
