// SPDX-License-Identifier: Apache-2.0
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// Director-AI Studio — Capabilities section: the manifest verbs explorer with the
// honesty palette. Reads the committed schema-A manifest (the only seam) and
// renders each verb's honest safety tier; an unsupported era warns, never crashes.

import { useEffect, useState } from "react";
import { HONESTY_PALETTE, gradeStatus } from "../contract/honesty";
import {
  type StudioManifest,
  type Verb,
  isEraSupported,
  loadManifest,
} from "../contract/manifest";

// Map a verb's safety tier onto an honest claim status (production→validated).
function tierStatus(verb: Verb): string {
  switch (verb.safety_tier) {
    case "production":
      return "validated";
    case "certified":
      return "validated";
    default:
      return "bounded"; // research is true-within-a-boundary, never "validated"
  }
}

function VerbRow({ verb }: { verb: Verb }): React.JSX.Element {
  const status = gradeStatus(tierStatus(verb));
  return (
    <tr>
      <td style={{ fontFamily: "var(--font-mono)" }}>{verb.verb}</td>
      <td>
        <span style={{ color: HONESTY_PALETTE[status] }}>{verb.safety_tier}</span>
      </td>
      <td>{verb.side_effect}</td>
      <td>{verb.timing.class}</td>
      <td>{verb.backends.join(", ")}</td>
      <td style={{ fontFamily: "var(--font-mono)" }}>{verb.produces.join(", ")}</td>
    </tr>
  );
}

export function Capabilities(): React.JSX.Element {
  const [manifest, setManifest] = useState<StudioManifest | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadManifest()
      .then(setManifest)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error) return <p style={{ color: "var(--status-voided)" }}>{error}</p>;
  if (!manifest) return <p>Loading capabilities…</p>;

  return (
    <section>
      <h2>Capabilities</h2>
      {!isEraSupported(manifest) && (
        <p style={{ color: "var(--status-boundary)" }}>
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
            <th>safety tier</th>
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
