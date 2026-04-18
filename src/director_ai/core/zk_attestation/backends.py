# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — attestation backends

"""Prover/verifier backends for passport statements.

Two real shapes:

* :class:`CommitmentBackend` — ships with the package. Uses the
  HMAC Merkle commitment primitives to issue proofs that spot-check
  opened samples against the prover's aggregate claim. Honest
  about what it is: commitment + challenge-response, not a true
  zero-knowledge proof.
* :class:`ZkSnarkBackend` — Protocol only. A real groth16 or
  plonk adapter (arkworks-rs, gnark, snarkjs) would implement
  this interface and plug into :class:`PassportVerifier` without
  any change to the passport format.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

try:
    from backfire_kernel import (
        rust_derive_challenge_indices as _rust_derive_challenge_indices,
    )

    _RUST_CHALLENGE_AVAILABLE = True
except ImportError:  # pragma: no cover — optional accelerator
    _RUST_CHALLENGE_AVAILABLE = False

from .commitment import (
    CommitmentProof,
    commit_samples,
    open_indices,
    verify_opening,
)
from .statements import AttestationStatement, HistorySample


@runtime_checkable
class AttestationBackend(Protocol):
    """Abstract prover/verifier pair for one statement.

    ``prove`` is called on the issuing side; it consumes the
    private sample stream and returns an opaque ``proof`` blob
    the passport serialises. ``verify`` is called on the receiving
    side; given the statement and the proof blob it returns
    ``(accepted, reason)``.
    """

    @property
    def kind(self) -> str: ...

    def prove(
        self,
        statement: AttestationStatement,
        samples: Sequence[HistorySample],
    ) -> object: ...

    def verify(
        self,
        statement: AttestationStatement,
        proof: object,
    ) -> tuple[bool, str]: ...


@dataclass
class CommitmentBackend:
    """HMAC-Merkle commitment backend — the default shipped impl.

    Parameters
    ----------
    key : bytes
        HMAC secret — must be at least 32 bytes.
    challenge_size : int
        Number of leaves the verifier opens per proof. Larger =
        tighter bound on cheating, at the cost of revealing more
        raw samples. 32 is a conservative default (probability of
        undetected cheating is ≤ (1 - cheat_fraction)^32).
    rng : object, optional
        Object exposing ``token_bytes(n)``. Defaults to
        :mod:`secrets`; tests inject a deterministic source.
    """

    key: bytes
    challenge_size: int = 32
    rng: object = field(default=None)

    kind: str = field(default="commitment", init=False)

    def __post_init__(self) -> None:
        if len(self.key) < 32:
            raise ValueError("HMAC key must be at least 32 bytes")
        if self.challenge_size <= 0:
            raise ValueError("challenge_size must be positive")

    def prove(
        self,
        statement: AttestationStatement,
        samples: Sequence[HistorySample],
    ) -> CommitmentProof:
        """Commit to *samples*, compute the aggregate, open a random
        challenge subset."""
        if not samples:
            raise ValueError("samples must be non-empty")
        commitment, leaves, blinds = commit_samples(samples, key=self.key, rng=self.rng)
        aggregate = sum(statement.evaluate_sample(s) for s in samples)
        indices = self._pick_challenge(
            seed_material=commitment.root.encode("utf-8"),
            sample_count=len(samples),
            challenge_size=min(self.challenge_size, len(samples)),
        )
        return open_indices(
            indices=indices,
            samples=samples,
            leaves=leaves,
            blinds=blinds,
            aggregate=aggregate,
            commitment=commitment,
        )

    def verify(
        self,
        statement: AttestationStatement,
        proof: object,
    ) -> tuple[bool, str]:
        if not isinstance(proof, CommitmentProof):
            return False, "wrong_proof_type"
        ok, reason = verify_opening(
            proof, key=self.key, per_sample_evaluator=statement.evaluate_sample
        )
        if not ok:
            return False, reason
        # Re-derive the challenge indices from the commitment root
        # — the prover cannot cherry-pick favourable samples because
        # the indices are a deterministic function of the root.
        # The challenge size is taken from the proof itself so a
        # verifier with a different default ``challenge_size`` can
        # still check proofs issued by another deployment; the
        # deterministic derivation means the *set* must still match
        # exactly for ``len(proof.opened)`` indices.
        proof_challenge_size = len(proof.opened)
        if proof_challenge_size > proof.total_samples:
            return False, "opened_larger_than_committed_population"
        expected_indices = self._pick_challenge(
            seed_material=proof.commitment.root.encode("utf-8"),
            sample_count=proof.total_samples,
            challenge_size=proof_challenge_size,
        )
        if set(proof.opened.keys()) != set(expected_indices):
            return False, "challenge_indices_do_not_match_root_derivation"
        if not statement.accepts(proof.aggregate, proof.total_samples):
            return False, "statement_threshold_not_met"
        return True, ""

    @staticmethod
    def _pick_challenge(
        seed_material: bytes,
        sample_count: int,
        challenge_size: int,
    ) -> list[int]:
        """Derive ``challenge_size`` distinct indices from a
        commitment root via incremental HMAC expansion.

        Using HMAC-SHA256 as the PRF makes the challenge derivation
        publicly reproducible (both sides can derive it offline)
        while still being unpredictable to anyone who has not seen
        the root.
        """
        if _RUST_CHALLENGE_AVAILABLE:
            return list(
                _rust_derive_challenge_indices(
                    seed_material, sample_count, challenge_size
                ),
            )
        indices: list[int] = []
        seen: set[int] = set()
        counter = 0
        hmac_key = b"director-ai/zk-attestation/challenge-derive/v1"
        while len(indices) < challenge_size and counter < challenge_size * 16:
            block = hmac.new(
                hmac_key,
                seed_material + counter.to_bytes(8, "big"),
                hashlib.sha256,
            ).digest()
            # 8 * 4-byte slots per block; consume until exhausted
            for offset in range(0, len(block), 4):
                if len(indices) >= challenge_size:
                    break
                chunk = int.from_bytes(block[offset : offset + 4], "big")
                idx = chunk % sample_count
                if idx not in seen:
                    seen.add(idx)
                    indices.append(idx)
            counter += 1
        return indices


@runtime_checkable
class ZkSnarkBackend(Protocol):
    """Plug-in protocol for real zk-SNARK adapters.

    Implementations supply ``kind`` (e.g. ``"groth16"``,
    ``"plonk"``), serialise their proof via ``bytes`` for the
    passport payload, and expose ``prove``/``verify`` with the
    same contract as :class:`AttestationBackend`. Shipping such
    a backend is deliberately out of scope — the protocol exists
    so operators can slot one in without touching the passport
    format or the verifier wiring.
    """

    @property
    def kind(self) -> str: ...

    def prove(
        self,
        statement: AttestationStatement,
        samples: Sequence[HistorySample],
    ) -> bytes: ...

    def verify(
        self,
        statement: AttestationStatement,
        proof_bytes: bytes,
    ) -> tuple[bool, str]: ...


__all__ = [
    "AttestationBackend",
    "CommitmentBackend",
    "ZkSnarkBackend",
]
