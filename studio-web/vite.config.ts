// SPDX-License-Identifier: Apache-2.0
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// Director-AI Studio — Vite config: the three never-redo build seams.
//
// One app, three outputs (pattern §2.1): a standalone static build, a
// Module-Federation 2.x remote exposing <DirectorAIStudioPanel> for the
// Institute hub, and a WASM bundle (the committed backfire-wasm kernel) for the
// live in-browser halt. COOP/COEP are set so SharedArrayBuffer-backed WASM
// multithreading is available later without a config change (seam, not feature).

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { federation } from "@module-federation/vite";

// Cross-origin isolation headers — required for SharedArrayBuffer / threaded WASM.
const crossOriginIsolation = {
  name: "cross-origin-isolation",
  configureServer(server: { middlewares: { use: (fn: unknown) => void } }) {
    server.middlewares.use(
      (
        _req: unknown,
        res: { setHeader: (k: string, v: string) => void },
        next: () => void,
      ) => {
        res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
        res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
        next();
      },
    );
  },
};

export default defineConfig({
  plugins: [
    react(),
    federation({
      name: "scpn_director_ai",
      filename: "remoteEntry.js",
      // Thin summary panel for the Institute hub (ratified federation depth):
      // verbs/capabilities/honest-status + headline numbers + "open full portal".
      exposes: {
        "./DirectorAIStudioPanel":
          "./src/federation/DirectorAIStudioPanel.tsx",
      },
      // Singleton-shared react/react-dom — the #1 MFE runtime-error source is a
      // duplicated react; pin one version across the workspace, not strictVersion.
      shared: {
        react: { singleton: true, requiredVersion: false },
        "react-dom": { singleton: true, requiredVersion: false },
      },
    }),
    crossOriginIsolation,
  ],
  // backfire-wasm ships a .wasm asset; keep it out of inlining so it streams.
  assetsInclude: ["**/*.wasm"],
  build: {
    target: "esnext", // top-level await + WASM ESM integration
    modulePreload: { polyfill: false },
    sourcemap: true,
  },
  worker: {
    format: "es",
  },
});
