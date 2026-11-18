// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI Studio — real Module Federation browser consumer for E2E.

import { createInstance } from "@module-federation/runtime";
import * as React from "react";
import * as ReactDOM from "react-dom";
import { createRoot } from "react-dom/client";
import type { DirectorAIStudioPanelProps } from "../../src/federation/DirectorAIStudioPanel";

/** Producer origin supplied by the browser test. */
const portalUrl = new URL(location.href).searchParams.get("portal");
/** Separate host mount for the remotely loaded React panel. */
const root = document.getElementById("panel");
/** Host form for changing the producer while keeping the remote mounted. */
const sourceForm = document.getElementById("portal-source");
/** URL passed into the remote on the next render. */
const sourceInput = sourceForm?.querySelector<HTMLInputElement>("input[name=portal]");
if (!portalUrl || !root || !sourceForm || !sourceInput) {
  throw new Error("The fixture needs a Studio URL, mount element, and producer form");
}

/** Federation host using the same React singleton as the remote. */
const host = createInstance({
  name: "studio_e2e_host",
  remotes: [{
    name: "scpn_director_ai",
    entry: new URL("remoteEntry.js", portalUrl).href,
    type: "module",
  }],
  shared: {
    react: {
      version: React.version,
      lib: () => React,
      shareConfig: { singleton: true, requiredVersion: React.version },
    },
    "react-dom": {
      version: ReactDOM.version,
      lib: () => ReactDOM,
      shareConfig: { singleton: true, requiredVersion: ReactDOM.version },
    },
  },
});

/** Remote module fetched from the producer's published entry point. */
const module = await host.loadRemote<{
  /** Component exposed as the public panel entry. */
  default: React.ComponentType<DirectorAIStudioPanelProps>;
}>(
  "scpn_director_ai/DirectorAIStudioPanel",
);
if (!module) throw new Error("The Studio remote did not expose its panel");
/** Host lifecycle handle used to exercise an actual React unmount. */
const panel = createRoot(root);
panel.render(React.createElement(module.default, { portalUrl }));
sourceInput.value = portalUrl;
sourceForm.addEventListener("submit", (event) => {
  event.preventDefault();
  panel.render(React.createElement(module.default, { portalUrl: sourceInput.value }));
});
/** Host control that removes the panel while a manifest request is pending. */
const removePanel = document.createElement("button");
removePanel.textContent = "Remove panel";
removePanel.addEventListener("click", () => panel.unmount(), { once: true });
root.before(removePanel);
