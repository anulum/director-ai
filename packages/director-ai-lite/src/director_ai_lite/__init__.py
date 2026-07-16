# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Director-Lite package
"""Director-Lite — the streaming-halt guard, standalone and free.

A self-contained distribution with no heavy dependencies and no ``director-ai``
requirement: ``pip install director-ai-lite`` and call :func:`guard` in three
lines. The grounding heuristic and coherence calibration match the full
package's no-model path, so passing the full ``director-ai`` NLI scorer (via the
``[full]`` extra) upgrades accuracy without changing the call site.
"""

from __future__ import annotations

from .guard import StreamGuard, StreamResult, guard, streaming_guard

__version__ = "3.18.1"

__all__ = [
    "StreamGuard",
    "StreamResult",
    "__version__",
    "guard",
    "streaming_guard",
]
