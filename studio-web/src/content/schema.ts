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
  readonly kind: "wasm-halt" | "katex" | "pyodide" | "chart";
  /** KaTeX source, a kernel config, or a chart artefact path — kind-dependent. */
  readonly source: string;
  /** For reproduction cells: the committed claim digest + its exactness (H2/H3). */
  readonly claim?: { readonly digest: string; readonly exactness: Exactness };
}

/** One lesson in a learning path — content bound to a committed artefact. */
export interface Lesson {
  readonly id: string;
  readonly title: string;
  /** The section spine slot this lesson belongs to. */
  readonly track: "learn" | "play" | "audit";
  /** Repo-relative path of the artefact this lesson is grounded in. */
  readonly artefact: string;
  /** Short prose (MDX would expand this; kept as data for i18n-readiness). */
  readonly summary: string;
  readonly cells: readonly RunnableCell[];
  /** Fact-check provenance: the source the content is verified against. */
  readonly verifiedAgainst: string;
}

/** A named learning path — an ordered list of lessons. */
export interface LearningPath {
  readonly id: string;
  readonly title: string;
  readonly lessons: readonly Lesson[];
}
