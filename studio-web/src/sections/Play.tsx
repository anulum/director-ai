// SPDX-License-Identifier: Apache-2.0
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// Director-AI Studio — Play section: the signature live streaming-halt.
//
// Runs the COMMITTED backfire-wasm kernel client-side. The visitor feeds tokens
// with contradiction scores and watches the stream HALT on the offending token —
// the same logic the science ran, no server (H1: a consumer-side demo, never
// promoted to verification). Lazy-loaded on this route only (perf budget).

import { useEffect, useRef, useState } from "react";
import { type HaltStep, StreamingHaltSession, openSession } from "../wasm/backfireKernel";

// A bundled sample: a benign claim, then one that contradicts the grounding.
const SAMPLE: ReadonlyArray<{ token: string; score: number }> = [
  { token: "Paris", score: 0.02 },
  { token: " is", score: 0.03 },
  { token: " the", score: 0.02 },
  { token: " capital", score: 0.04 },
  { token: " of", score: 0.03 },
  { token: " Germany", score: 0.88 }, // contradicts grounding → halt
];

export function Play(): React.JSX.Element {
  const sessionRef = useRef<StreamingHaltSession | null>(null);
  const [steps, setSteps] = useState<HaltStep[]>([]);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    openSession({ threshold: 0.65 })
      .then((session) => {
        if (cancelled) {
          session.dispose();
          return;
        }
        sessionRef.current = session;
        setReady(true);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
      sessionRef.current?.dispose();
      sessionRef.current = null;
    };
  }, []);

  function run(): void {
    const session = sessionRef.current;
    if (!session) return;
    const collected: HaltStep[] = [];
    for (const { token, score } of SAMPLE) {
      const step = session.push(token, score);
      collected.push(step);
      if (!step.active) break; // halted
    }
    setSteps(collected);
  }

  if (error) return <p style={{ color: "var(--status-voided)" }}>WASM error: {error}</p>;

  const halted = steps.length > 0 && !steps[steps.length - 1]?.active;
  return (
    <section>
      <h2>Play — live contradiction halt</h2>
      <p>
        The committed <code>backfire-wasm</code> kernel runs in your browser. Reproduced
        in your browser ✓ — a consumer-side demo, not a verification claim.
      </p>
      <button type="button" disabled={!ready} onClick={run}>
        {ready ? "Stream the sample answer" : "Loading kernel…"}
      </button>
      <ol style={{ fontFamily: "var(--font-mono)" }}>
        {steps.map((s, i) => (
          <li
            key={i}
            style={{ color: s.active ? "var(--text)" : "var(--status-voided)" }}
          >
            {JSON.stringify(s.token)} score={s.score.toFixed(2)}{" "}
            {s.active ? "" : "← HALT"}
          </li>
        ))}
      </ol>
      {halted && (
        <p style={{ color: "var(--status-voided)" }}>
          Stream halted: a completed claim contradicted the grounding.
        </p>
      )}
    </section>
  );
}
