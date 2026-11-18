// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI Studio — streaming lesson entry.

import { Link } from "react-router-dom";
import { streamingHaltLesson } from "../content/lessons/streamingHalt";

/** Present the authored streaming lesson and link to its browser exercise. */
export function Learn(): React.JSX.Element {
  return (
    <section>
      <h2>{streamingHaltLesson.title}</h2>
      <p>{streamingHaltLesson.summary}</p>
      <Link to="/play">Run the WASM sample</Link>
    </section>
  );
}
