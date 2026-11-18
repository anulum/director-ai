// SPDX-License-Identifier: Apache-2.0
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// Director-AI Studio — show the producer-declared capabilities in the manifest.

import { useEffect, useState } from "react";
import {
  type StudioManifest,
  type Verb,
  isEraSupported,
  loadManifest,
} from "../contract/manifest";

/** Render a capability with the assurance tier declared by the producer. */
function VerbRow({ verb }: {
  /** One manifest entry as declared by the producer. */
  verb: Verb;
}): React.JSX.Element {
  return (
    <tr>
      <td style={{ fontFamily: "var(--font-mono)" }}>{verb.verb}</td>
      <td>{verb.safety_tier}</td>
      <td>{verb.side_effect}</td>
      <td>{verb.timing.class}</td>
      <td>{verb.backends.join(", ")}</td>
      <td style={{ fontFamily: "var(--font-mono)" }}>{verb.produces.join(", ")}</td>
    </tr>
  );
}

/** Fetch the Studio manifest and show its declared capabilities and era. */
export function Capabilities(): React.JSX.Element {
  const [manifest, setManifest] = useState<StudioManifest | null>(null);
  const [unavailable, setUnavailable] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    void loadManifest(undefined, signal)
      .then(setManifest)
      .catch(() => {
        if (!signal.aborted) setUnavailable(true);
      });
    return () => controller.abort();
  }, []);

  if (unavailable) {
    return <p role="alert" style={{ color: "var(--status-voided)" }}>The Studio capability manifest could not load.</p>;
  }
  if (!manifest) return <p role="status">Loading capabilities…</p>;

  return (
    <section>
      <h2>Capabilities</h2>
      {!isEraSupported(manifest) && (
        <p role="status" style={{ color: "var(--status-boundary)" }}>
          Manifest era {manifest.contract_era} differs from this build — fields shown
          best-effort.
        </p>
      )}
      <p>
        {manifest.studio} {manifest.studio_version} ·{" "}
        <span style={{ fontFamily: "var(--font-mono)" }}>{manifest.content_digest}</span>
      </p>
      <table>
        <thead>
          <tr>
            <th>verb</th>
            <th>Declared safety tier</th>
            <th>side effect</th>
            <th>timing</th>
            <th>backends</th>
            <th>produces</th>
          </tr>
        </thead>
        <tbody>
          {manifest.verbs.map((v) => (
            <VerbRow key={v.verb} verb={v} />
          ))}
        </tbody>
      </table>
    </section>
  );
}
