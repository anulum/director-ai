# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — TraceSafe package

"""Dynamic mid-trajectory safety oracle.

Given a partial agent trace (a list of accumulated tokens or a
text snapshot), TraceSafe classifies it against a curated corpus
of safe and unsafe exemplars and returns a verdict before the
stream finishes. The oracle plugs into :class:`StreamingKernel`
via a standard :class:`TokenTraceCallback` — every Nth emitted
token triggers a snapshot evaluation.

The :class:`TraceEmbedder` Protocol is the stable boundary:
the :class:`HashBagEmbedder` that ships here is a deterministic
FNV-1a baseline that runs without optional dependencies;
sentence-transformer, cross-encoder, and Rust-accelerated
embedders slot in as drop-in implementations.
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
