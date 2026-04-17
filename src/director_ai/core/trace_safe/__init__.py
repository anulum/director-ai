# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — TraceSafe package

"""Dynamic mid-trajectory safety oracle (roadmap 2026-2030,
Tier 1 Batch 2 — TraceSafe foundation).

Given a partial agent trace (a list of accumulated tokens or a
text snapshot), TraceSafe classifies it against a curated corpus
of safe and unsafe exemplars and returns a verdict before the
stream finishes. The oracle plugs into :class:`StreamingKernel`
via a standard :class:`TokenTraceCallback` — every Nth emitted
token triggers a snapshot evaluation.

Foundation scope: hash-bag embedding baseline (zero optional
dependencies), centroid distance classifier, configurable
decision bands. Follow-ups tracked separately: sentence-
transformer embedder, spectral clustering for multi-modal unsafe
classes, Rust fast-path for the cosine-distance scan, and
kernel-level integration with
``core.runtime.streaming.StreamingKernel``.
"""

from .embedder import HashBagEmbedder, TraceEmbedder
from .oracle import (
    TraceLabel,
    TraceSafeOracle,
    TraceSample,
    TraceVerdict,
)

__all__ = [
    "HashBagEmbedder",
    "TraceEmbedder",
    "TraceLabel",
    "TraceSafeOracle",
    "TraceSample",
    "TraceVerdict",
]
