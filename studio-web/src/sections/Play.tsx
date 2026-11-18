// SPDX-License-Identifier: Apache-2.0
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// Director-AI Studio — play the WASM coherence halt with sample scores.

import { useEffect, useState } from "react";
import { DEFAULT_BACKFIRE_CONFIG } from "../wasm/backfireConfig";
import { type HaltStep, initKernel, StreamingHaltSession } from "../wasm/backfireKernel";

/** Fixed input sequence that crosses the default hard limit near the end. */
const SAMPLE: ReadonlyArray<{
  /** Text fragment supplied to the native session. */
  token: string;
  /** Host-supplied coherence for this fragment. */
  score: number;
}> = [
  { token: "Paris", score: 0.98 },
  { token: " is", score: 0.97 },
  { token: " the", score: 0.98 },
  { token: " capital", score: 0.96 },
  { token: " of", score: 0.97 },
  { token: " Germany", score: 0.12 },
  { token: ".", score: 0.99 },
];

/** Distinguish asset failures from failures that require a fresh browser module. */
type KernelStatus = "loading" | "ready" | "load-error" | "run-error";

/** Render a retryable browser demo and an independent trace for each run. */
export function Play(): React.JSX.Element {
  const [steps, setSteps] = useState<HaltStep[]>([]);
  const [status, setStatus] = useState<KernelStatus>("loading");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let current = true;
    void initKernel().then(
      () => { if (current) setStatus("ready"); },
      () => { if (current) setStatus("load-error"); },
    );
    return () => { current = false; };
  }, [attempt]);

  /** Initialise again after an asset load or compilation error. */
  function retry(): void {
    setStatus("loading");
    setAttempt((previous) => previous + 1);
  }

  /** Feed the sample to a new native session and release its memory. */
  function run(): void {
    try {
      const session = new StreamingHaltSession();
      const collected: HaltStep[] = [];
      try {
        for (const { token, score } of SAMPLE) {
          const step = session.push(token, score);
          collected.push(step);
          if (!step.active) break;
        }
      } finally {
        session.dispose();
      }
      setSteps(collected);
    } catch {
      setSteps([]);
      setStatus("run-error");
    }
  }

  const reason = steps.at(-1)?.haltReason;
  const reloadRequired = status === "run-error";
  return (
    <section>
      <h2>Play — WASM coherence halt</h2>
      <p>
        Run the native kernel in your browser with sample coherence scores.
        A score below {DEFAULT_BACKFIRE_CONFIG.hard_limit.toFixed(2)} triggers a hard halt.
      </p>
      {status === "load-error" || reloadRequired ? (
        <>
          <p role="alert">{reloadRequired ? "The WASM sample could not run." : "The WASM kernel could not load."}</p>
          {reloadRequired ? (
            <button type="button" onClick={() => window.location.reload()}>Reload page</button>
          ) : (
            <button type="button" onClick={retry}>Retry WASM</button>
          )}
        </>
      ) : (
        <button type="button" disabled={status === "loading"} onClick={run}>
          {status === "ready" ? "Run the sample" : "Loading kernel…"}
        </button>
      )}
      {steps.length > 0 && (
        <ol aria-label="Token trace" style={{ fontFamily: "var(--font-mono)" }}>
          {steps.map((step, index) => (
            <li key={index} style={{ color: step.active ? "var(--text)" : "var(--status-voided)" }}>
              {JSON.stringify(step.token)} score={step.score.toFixed(2)} {step.active ? "" : "← HALT"}
            </li>
          ))}
        </ol>
      )}
      {reason && (
        <p role="status">Stream halted: {reason}</p>
      )}
    </section>
  );
}
