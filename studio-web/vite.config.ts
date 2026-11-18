// SPDX-License-Identifier: Apache-2.0
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// Director-AI Studio — Vite config: the three never-redo build seams.
//
// One app, three outputs (pattern §2.1): a standalone static build, a
// Module-Federation 2.x remote exposing <DirectorAIStudioPanel> for the
// Institute hub, and a WASM bundle (generated from the backfire-wasm crate) for the
// live in-browser halt. COOP/COEP are set so SharedArrayBuffer-backed WASM
// multithreading is available later without a config change (seam, not feature).

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { federation } from "@module-federation/vite";

/** Apply the same cross-origin isolation to development and production preview. */
const isolationHeaders = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
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
      // Share one React version with a compatible host.
      shared: {
        react: { singleton: true, requiredVersion: "19.3.1" },
        "react-dom": { singleton: true, requiredVersion: "19.3.1" },
      },
    }),
  ],
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
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
