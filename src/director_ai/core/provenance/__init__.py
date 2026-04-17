# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — provenance + citation integrity layer

"""Cryptographic provenance for RAG-grounded responses.

Every fact the guardrail cites is packaged as a
:class:`CitationFact` carrying the source id, a SHA-256 content
hash, and a timestamp. Facts from the same response are
accumulated into a :class:`MerkleTree`; the resulting Merkle
root + a caller-supplied HMAC-SHA-256 tag make each response's
citation bundle tamper-evident — any later edit to a fact
breaks the proof.

:class:`SourceCredibility` tracks a per-source trust score with
exponential decay: freshly-published facts weigh the source's
recent credibility more heavily than facts many days old. The
tracker is thread-safe so background tasks can update it while
the verifier runs in the request path.

:class:`ProvenanceVerifier` bundles the Merkle root builder + the
HMAC signer + the credibility tracker into one API so the
scoring layer hands a response to
:meth:`ProvenanceVerifier.verify` and receives a
:class:`ProvenanceVerdict` with the per-fact checks and a
composite trust score.
"""

from .chain import HmacChainError, ProvenanceChain, ProvenanceEntry
from .credibility import SourceCredibility, SourceScore
from .facts import CitationFact, FactVerificationError
from .merkle import MerkleProof, MerkleTree
from .verifier import ProvenanceVerdict, ProvenanceVerifier

__all__ = [
    "CitationFact",
    "FactVerificationError",
    "HmacChainError",
    "MerkleProof",
    "MerkleTree",
    "ProvenanceChain",
    "ProvenanceEntry",
    "ProvenanceVerdict",
    "ProvenanceVerifier",
    "SourceCredibility",
    "SourceScore",
]
