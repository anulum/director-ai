// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI Studio — recovery for a route that failed to load.

import { Component, type ReactNode } from "react";

/** Keep the Studio shell usable when the current route fails. */
export class RouteLoadBoundary extends Component<{
  /** Routed content eligible for recovery independently of navigation. */
  children: ReactNode;
}, {
  /** Whether this pathname is showing its recovery view. */
  failed: boolean;
}> {
  /** A newly mounted pathname first attempts its normal view. */
  override state = { failed: false };

  /** Switch the route area to its recovery view after a React render error. */
  static getDerivedStateFromError(): {
    /** React stores this flag after catching a route render failure. */
    failed: boolean;
  } {
    return { failed: true };
  }

  /** Show a fixed message and let the browser load the current route again. */
  override render(): ReactNode {
    if (!this.state.failed) return this.props.children;

    return (
      <section>
        <p role="alert">This Studio page could not load.</p>
        <button type="button" onClick={() => window.location.reload()}>Reload page</button>
      </section>
    );
  }
}
