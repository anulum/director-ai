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

export interface DirectorAIStudioPanelProps {
  /** Where the hub should send a visitor who opens the full portal. */
  readonly portalUrl?: string;
}

export default function DirectorAIStudioPanel({
  portalUrl = "/",
}: DirectorAIStudioPanelProps): React.JSX.Element {
  const [manifest, setManifest] = useState<StudioManifest | null>(null);

  useEffect(() => {
    loadManifest().then(setManifest).catch(() => setManifest(null));
  }, []);

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
        Watch an LLM hallucination-halt fire live in your browser — and see the
        honest false-positive cost next to the accuracy.
      </p>
      {manifest ? (
        <dl>
          <dt>verbs</dt>
          <dd>{manifest.verbs.length}</dd>
          <dt>evidence types</dt>
          <dd>{manifest.evidence_types.length}</dd>
          <dt>contract</dt>
          <dd>{manifest.contract_era}</dd>
        </dl>
      ) : (
        <p>Summary unavailable.</p>
      )}
      <a href={portalUrl}>Open the full portal →</a>
    </article>
  );
}
