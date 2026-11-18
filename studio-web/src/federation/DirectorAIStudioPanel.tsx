// SPDX-License-Identifier: Apache-2.0
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// Director-AI Studio — the federated remote panel for the Institute hub.
//
// Ratified federation depth (Q5): a THIN summary — studio identity, verb count,
// the headline honest numbers — plus an "open full portal" link. The hub loads
// this via Module Federation; the heavy portal stays behind the link, so the
// remote is light and host↔remote stay code-independent.

import { useEffect, useState } from "react";
import { type StudioManifest, loadManifest } from "../contract/manifest";

/** Producer location supplied by the embedding federation host. */
export interface DirectorAIStudioPanelProps {
  /** Studio entry URL. The manifest is served alongside that entry. */
  readonly portalUrl?: string;
}

/** Render a federated summary from the Studio producer's manifest. */
export default function DirectorAIStudioPanel({
  portalUrl = "/",
}: DirectorAIStudioPanelProps): React.JSX.Element {
  return (
    <article
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius)",
        padding: "var(--space-3)",
      }}
    >
      <h3>Director-AI — AI-Safety / Streaming Academy</h3>
      <p style={{ color: "var(--text-muted)" }}>
        Explore the declared Director-AI capabilities and open the browser coherence demo.
      </p>
      <ManifestSummary key={portalUrl} portalUrl={portalUrl} />
      <a href={portalUrl}>Open the full portal →</a>
    </article>
  );
}

/** Fetch and render a fresh summary for the lifetime of the selected producer. */
function ManifestSummary({ portalUrl }: Required<DirectorAIStudioPanelProps>): React.JSX.Element {
  const [manifest, setManifest] = useState<StudioManifest | null>();

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;

    /** Load the manifest alongside the configured Studio portal. */
    async function loadSummary(): Promise<void> {
      let nextManifest: StudioManifest | null = null;
      try {
        const portal = new URL(portalUrl, window.location.href);
        nextManifest = await loadManifest(new URL("studio_manifest.json", portal).href, signal);
      } catch {
        // A host can open the portal while its manifest is unavailable.
      }
      if (!signal.aborted) setManifest(nextManifest);
    }

    void loadSummary();
    return () => controller.abort();
  }, [portalUrl]);

  return (
    <>
      {manifest && (
        <dl>
          <dt>verbs</dt>
          <dd>{manifest.verbs.length}</dd>
          <dt>evidence types</dt>
          <dd>{manifest.evidence_types.length}</dd>
          <dt>contract</dt>
          <dd>{manifest.contract_era}</dd>
        </dl>
      )}
      {manifest === undefined && <p>Loading summary…</p>}
      {manifest === null && <p>Summary unavailable.</p>}
    </>
  );
}
