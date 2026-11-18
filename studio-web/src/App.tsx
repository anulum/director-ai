// SPDX-License-Identifier: Apache-2.0
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// Director-AI Studio — app shell + the section spine (learn → play → audit).
// Routes are code-split; the WASM Play route is lazy so it loads only on demand
// (perf budget: < 200KB JS on first paint, WASM on the Play route only).

import { Suspense, lazy } from "react";
import { Link, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { RouteLoadBoundary } from "./sections/RouteLoadBoundary";

/** Load the manifest view only when the visitor opens Capabilities. */
const Capabilities = lazy(() =>
  import("./sections/Capabilities").then((m) => ({ default: m.Capabilities })),
);
/** Load the lesson without starting the WASM exercise. */
const Learn = lazy(() => import("./sections/Learn").then((m) => ({ default: m.Learn })));
/** Defer the kernel and exercise until the visitor opens Play. */
const Play = lazy(() => import("./sections/Play").then((m) => ({ default: m.Play })));

/** Available views in the standalone portal's navigation order. */
const SECTIONS: ReadonlyArray<{
  /** Browser pathname matched by the router. */
  path: string;
  /** Visible link name. */
  label: string;
  /** View rendered at this path. */
  view: React.ComponentType;
}> = [
  { path: "/", label: "Home", view: Home },
  { path: "/learn", label: "Learn", view: Learn },
  { path: "/play", label: "Play", view: Play },
  { path: "/capabilities", label: "Capabilities", view: Capabilities },
];

/** Planned routes retain their bookmark addresses and titles. */
const PLANNED_SECTIONS = [
  { path: "/pipeline", label: "Pipeline" },
  { path: "/benchmarks", label: "Benchmarks" },
  { path: "/evidence", label: "Science & Evidence" },
  { path: "/redteam", label: "Red-Team" },
  { path: "/architecture", label: "Architecture" },
  { path: "/lab", label: "3D Lab" },
] as const;

/** Explain an unavailable route and give the visitor a way back. */
function RouteNotice({ title, children }: {
  /** Heading that identifies the route status. */
  title: string;
  /** Reason the route lacks a working view. */
  children: React.ReactNode;
}): React.JSX.Element {
  return (
    <section>
      <h2>{title}</h2>
      <p>{children}</p>
      <Link to="/">Open Studio home</Link>
    </section>
  );
}

/** Introduce the portal's kernel demonstration and manifest view. */
function Home(): React.JSX.Element {
  return (
    <section>
      <h1>AI-Safety / Streaming Academy</h1>
      <p>
        Run the WASM coherence halt with sample scores and explore the capabilities
        declared by the Director-AI manifest.
      </p>
    </section>
  );
}

/** Render persistent navigation and recover independently per pathname. */
export default function App(): React.JSX.Element {
  const { pathname } = useLocation();

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
        <RouteLoadBoundary key={pathname}>
          <Suspense fallback={<p>Loading…</p>}>
            <Routes>
              {SECTIONS.map(({ path, view: View }) => (
                <Route key={path} path={path} element={<View />} />
              ))}
              {PLANNED_SECTIONS.map(({ path, label }) => (
                <Route key={path} path={path} element={
                  <RouteNotice title={label}>This section is planned for a future Studio release.</RouteNotice>
                } />
              ))}
              <Route path="*" element={
                <RouteNotice title="Studio page not found">This address does not match a Studio page.</RouteNotice>
              } />
            </Routes>
          </Suspense>
        </RouteLoadBoundary>
      </main>
    </div>
  );
}
