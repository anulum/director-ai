# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — cross-org attestation passports

"""Cross-organisation agent passports with cryptographic proofs.

When organisation **A** hands off a running agent to organisation
**B**, B wants evidence about the agent's past behaviour without
A having to release raw interaction logs. A passport is a signed
bundle of typed :class:`AttestationStatement` claims, each backed
by a cryptographic proof that B's :class:`PassportVerifier` can
check offline.

This package ships one real backend — :class:`CommitmentBackend`
— and a :class:`ZkSnarkBackend` protocol for pluggable zk-SNARK
adapters.

* **CommitmentBackend** is a cryptographic commitment scheme
  (Merkle tree of HMAC-committed samples + challenge-based
  spot-check). It is hiding under the random-oracle assumption
  but does *not* produce succinct zero-knowledge proofs — calling
  it ``zk`` would be dishonest. It is suitable when the two orgs
  already trust a minimal protocol round-trip (commit → receive
  challenge → open revealed indices).
* **ZkSnarkBackend** is a Protocol for real zk-SNARK adapters
  (groth16 via arkworks / gnark / bellman) brought in as
  entry-points or direct subclass. Shipping such a backend is
  out of scope for this package: the Protocol and the verifier
  wiring live here so an operator can slot one in without
  touching :class:`PassportVerifier`.
"""

from __future__ import annotations

from .backends import AttestationBackend, CommitmentBackend, ZkSnarkBackend
from .commitment import (
    CommitmentProof,
    MerkleCommitment,
    commit_samples,
    open_indices,
    verify_opening,
)
from .passport import (
    CrossOrgPassport,
    PassportIssuer,
    PassportVerdict,
    PassportVerifier,
)
from .statements import (
    AttestationStatement,
    DomainExperience,
    MaximumHaltRate,
    MinimumCoherence,
    NoBreakoutEvents,
)

__all__ = [
    "AttestationBackend",
    "AttestationStatement",
    "CommitmentBackend",
    "CommitmentProof",
    "CrossOrgPassport",
    "DomainExperience",
    "MaximumHaltRate",
    "MerkleCommitment",
    "MinimumCoherence",
    "NoBreakoutEvents",
    "PassportIssuer",
    "PassportVerdict",
    "PassportVerifier",
    "ZkSnarkBackend",
    "commit_samples",
    "open_indices",
    "verify_opening",
]
