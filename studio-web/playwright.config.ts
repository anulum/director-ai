// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI Studio — exercise the production build in Chromium.

import { defineConfig } from "@playwright/test";

/** Loopback URL shared by the production preview and browser client. */
const origin = "http://127.0.0.1:4173";
/** Optional host Chrome binary for environments with a system browser. */
const executablePath = process.env.STUDIO_CHROME_EXECUTABLE;

export default defineConfig({
  testDir: "./e2e",
  workers: 1,
  use: {
    baseURL: origin,
    launchOptions: { chromiumSandbox: true, ...(executablePath ? { executablePath } : {}) },
  },
  webServer: {
    command: "npm run preview -- --host 127.0.0.1 --port 4173 --strictPort",
    url: origin,
    reuseExistingServer: false,
  },
});
