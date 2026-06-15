# SPDX-License-Identifier: Apache-2.0
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

This package ships three real backends — :class:`CommitmentBackend`,
:class:`SchnorrAttestationBackend`, and :class:`BulletproofRangeBackend` —
plus a :class:`ZkSnarkBackend` protocol for pluggable zk-SNARK adapters.

* **CommitmentBackend** is a cryptographic commitment scheme
  (Merkle tree of HMAC-committed samples + challenge-based
  spot-check). It is hiding under the random-oracle assumption
  but does *not* produce succinct zero-knowledge proofs — calling
  it ``zk`` would be dishonest. It is suitable when the two orgs
  already trust a minimal protocol round-trip (commit → receive
  challenge → open revealed indices).
* **SchnorrAttestationBackend** seals each sample contribution in
  an additively-homomorphic Pedersen commitment and proves the
  aggregate opening with a non-interactive Schnorr proof, so the
  individual sample values stay hidden (perfect hiding) while the
  aggregate and the threshold decision are checkable. It reveals
  the aggregate value itself; hiding that too (revealing only
  "threshold met") needs a zero-knowledge range proof, which is
  the :class:`ZkSnarkBackend` plug-in's job.
* **BulletproofRangeBackend** hides the aggregate too. It commits
  each sample on Ristretto, sums the commitments homomorphically,
  and uses a Bulletproof (dalek ``bulletproofs`` via the
  ``backfire_kernel`` extension) to prove the committed aggregate
  clears the public threshold while revealing neither the values
  nor their sum — only "threshold met". Rust-only (no Python
  fallback); the threshold stays public.
* **ZkSnarkBackend** is a Protocol for real zk-SNARK adapters
  (groth16 via arkworks / gnark / bellman) brought in as
  entry-points or direct subclass. The Protocol and the verifier
  wiring live here so an operator can slot one in without
  touching :class:`PassportVerifier`.
"""

from __future__ import annotations

from .backends import AttestationBackend, CommitmentBackend, ZkSnarkBackend
from .bulletproof_range import (
    BulletproofRangeBackend,
    RangeAttestation,
    min_aggregate_for,
)
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
from .schnorr import (
    AggregateAttestation,
    PedersenParameters,
    SchnorrAttestationBackend,
    SchnorrProof,
    default_parameters,
    pedersen_commit,
)
from .statements import (
    AttestationStatement,
    DomainExperience,
    MaximumHaltRate,
    MinimumCoherence,
    NoBreakoutEvents,
)


def __getattr__(name: str) -> object:
    """Lazily expose ``DEFAULT_PARAMETERS`` so importing this package stays cheap.

    Building the default group runs a primality re-check (a few 2048-bit modular
    exponentiations); deferring it keeps ``from director_ai.core.zk_attestation
    import CommitmentBackend`` fast for callers that never touch the Schnorr group.
    """
    if name == "DEFAULT_PARAMETERS":
        return default_parameters()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_PARAMETERS",
    "AggregateAttestation",
    "AttestationBackend",
    "AttestationStatement",
    "BulletproofRangeBackend",
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
    "PedersenParameters",
    "RangeAttestation",
    "SchnorrAttestationBackend",
    "SchnorrProof",
    "ZkSnarkBackend",
    "commit_samples",
    "min_aggregate_for",
    "open_indices",
    "pedersen_commit",
    "default_parameters",
    "verify_opening",
]
