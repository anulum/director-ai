// SPDX-License-Identifier: Apache-2.0
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// Director-AI Studio — app shell + the section spine (learn → play → audit).
// Routes are code-split; the WASM Play route is lazy so it loads only on demand
// (perf budget: < 200KB JS on first paint, WASM on the Play route only).

import { Suspense, lazy } from "react";
import { NavLink, Route, Routes } from "react-router-dom";

const Capabilities = lazy(() =>
  import("./sections/Capabilities").then((m) => ({ default: m.Capabilities })),
);
const Play = lazy(() => import("./sections/Play").then((m) => ({ default: m.Play })));

// The adaptable spine (pattern §2.4); each repo keeps the order of engagement.
const SECTIONS: ReadonlyArray<{ path: string; label: string }> = [
  { path: "/", label: "Home" },
  { path: "/learn", label: "Learn" },
  { path: "/play", label: "Play" },
  { path: "/pipeline", label: "Pipeline" },
  { path: "/capabilities", label: "Capabilities" },
  { path: "/benchmarks", label: "Benchmarks" },
  { path: "/evidence", label: "Science & Evidence" },
  { path: "/redteam", label: "Red-Team" },
  { path: "/architecture", label: "Architecture" },
  { path: "/lab", label: "3D Lab" },
];

function Placeholder({ title }: { title: string }): React.JSX.Element {
  return (
    <section>
      <h2>{title}</h2>
      <p style={{ color: "var(--text-muted)" }}>
        Phase-{">"}0 content lands here, bound to a committed artefact (content-as-data).
      </p>
    </section>
  );
}

function Home(): React.JSX.Element {
  return (
    <section>
      <h1>AI-Safety / Streaming Academy</h1>
      <p>
        Learn how an LLM hallucination guardrail works, play with the contradiction
        halt live in your browser, and audit its accuracy <em>and</em> its
        false-positive cost.
      </p>
    </section>
  );
}

export default function App(): React.JSX.Element {
  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "var(--space-4)" }}>
      <nav style={{ display: "flex", gap: "var(--space-3)", flexWrap: "wrap" }}>
        {SECTIONS.map((s) => (
          <NavLink key={s.path} to={s.path}>
            {s.label}
          </NavLink>
        ))}
      </nav>
      <main>
        <Suspense fallback={<p>Loading…</p>}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/learn" element={<Placeholder title="Learn" />} />
            <Route path="/play" element={<Play />} />
            <Route path="/pipeline" element={<Placeholder title="Pipeline" />} />
            <Route path="/capabilities" element={<Capabilities />} />
            <Route path="/benchmarks" element={<Placeholder title="Benchmarks" />} />
            <Route path="/evidence" element={<Placeholder title="Science & Evidence" />} />
            <Route path="/redteam" element={<Placeholder title="Red-Team" />} />
            <Route path="/architecture" element={<Placeholder title="Architecture" />} />
            <Route path="/lab" element={<Placeholder title="3D Lab" />} />
          </Routes>
        </Suspense>
      </main>
    </div>
  );
}
