# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Pedersen + Schnorr zero-knowledge attestation backend

"""Zero-knowledge attestation of an aggregate over hidden per-sample values.

The shipped :class:`~director_ai.core.zk_attestation.CommitmentBackend` proves a
statement by *opening* a random subset of samples — sound, but it reveals those
raw samples. This backend hides them. Each sample's statement contribution
``v_i`` is sealed in a Pedersen commitment ``C_i = g^{v_i} · h^{r_i} (mod p)`` in a
prime-order group where ``log_g h`` is unknown, so a commitment reveals nothing
about ``v_i`` (perfect hiding) yet binds the prover to it (computational binding
under the discrete-log assumption). Pedersen commitments are additively
homomorphic, so the product of the per-sample commitments is a commitment to the
aggregate ``A = Σ v_i`` with blinding ``R = Σ r_i``:

    ∏ C_i = g^{Σ v_i} · h^{Σ r_i} = g^A · h^R.

The prover reveals only the aggregate ``A`` and a non-interactive Schnorr proof of
knowledge of ``R`` in ``(∏ C_i) · g^{-A} = h^R`` (Fiat-Shamir over SHA-256). The
verifier recomputes the aggregate commitment from the published per-sample
commitments, checks the Schnorr proof, and applies ``statement.accepts(A, n)`` —
learning the aggregate and the threshold decision but **never an individual
sample value or blinding**.

What this hides and what it does not (stated plainly, no overclaim):

* **hidden:** every individual per-sample value and blinding (perfect hiding);
* **revealed:** the aggregate ``A`` itself, and the public statement/threshold;
* **not proven here:** that the committed ``v_i`` are honest evaluations of real
  samples (a circuit/SNARK statement) — compose with the spot-checking
  :class:`CommitmentBackend` or a real SNARK (the :class:`ZkSnarkBackend`
  plug-in) for that. Hiding the aggregate value itself (revealing only "threshold
  met") needs a zero-knowledge range proof (Bulletproofs / SNARK), which remains
  the documented plug-in point.

The default group is a generated 2048-bit safe prime; the parameters are
re-verified (primality of ``p`` and ``q = (p-1)/2``, subgroup membership of the
generators) at construction, so a corrupted constant fails fast rather than
silently weakening the proof.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Sequence
from dataclasses import dataclass, field

from .statements import AttestationStatement, HistorySample

__all__ = [
    "PedersenParameters",
    "SchnorrProof",
    "AggregateAttestation",
    "SchnorrAttestationBackend",
    "DEFAULT_PARAMETERS",
]

# Generated 2048-bit safe prime p = 2q + 1 (q prime); re-verified at construction.
# Produced by a Miller-Rabin safe-prime search (no hand-entered constant).
_DEFAULT_P = 30269240982472018763728173750962776174373478314537892897998398797906686558389116591975819926561028942762695719045746191709915110971902175428321268332643698537996224573874496413750161634512101249182693017639520328139347743572528430904992564596423286485633775438493292230455397406278022185594507980703370931390703022937157366554965046856235594594474131626972162188343747624104284293330970133618963893416155513571585875362322207372139238789926956109892484411173029533585617543420003313815188904771828043011958578149245549115828883349253654477824678491862116983414414867886882103415295798865336754055533402485794772214143
_FIXED_POINT_SCALE = 1_000_000


def _is_probable_prime(n: int, *, rounds: int = 24) -> bool:
    """Miller-Rabin probable-primality test (parameter self-verification)."""
    if n < 2:
        return False
    for small in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % small == 0:
            return n == small
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for _ in range(rounds):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _hash_to_subgroup(seed: bytes, p: int) -> int:
    """Deterministic nothing-up-my-sleeve generator of the order-q subgroup.

    Squaring lands the hash in the quadratic-residue subgroup of order
    ``q = (p-1)/2``; ``log_g`` of the result is unknown to everyone, which is what
    a Pedersen second generator requires.
    """
    counter = 0
    while True:
        digest = hashlib.sha256(seed + counter.to_bytes(8, "big")).digest()
        candidate = (int.from_bytes(digest, "big") % (p - 2)) + 2
        value = pow(candidate, 2, p)
        if value not in (0, 1):
            return value
        counter += 1


@dataclass(frozen=True)
class PedersenParameters:
    """Prime-order group ``(p, q, g, h)`` for Pedersen commitments.

    ``p`` is a safe prime, ``q = (p-1)/2`` is prime, and ``g``, ``h`` generate the
    order-``q`` quadratic-residue subgroup with ``log_g h`` unknown. All of this
    is verified in :meth:`validate`, called at construction.
    """

    p: int
    q: int
    g: int
    h: int

    def validate(self) -> None:
        """Re-derive every structural invariant; raise on any failure."""
        if self.q != (self.p - 1) // 2:
            raise ValueError("q must equal (p-1)/2")
        if not _is_probable_prime(self.p):
            raise ValueError("p is not prime")
        if not _is_probable_prime(self.q):
            raise ValueError("q is not prime (p is not a safe prime)")
        for gen in (self.g, self.h):
            if not 1 < gen < self.p:
                raise ValueError("generator out of range")
            if pow(gen, self.q, self.p) != 1:
                raise ValueError("generator is not in the order-q subgroup")
        if self.g == self.h:
            raise ValueError("g and h must be distinct")


def _build_default_parameters() -> PedersenParameters:
    p = _DEFAULT_P
    q = (p - 1) // 2
    g = pow(2, 2, p)  # 4 = 2^2 is a quadratic residue, generator of order q
    h = _hash_to_subgroup(b"director-ai/zk-attestation/pedersen-h/v1", p)
    params = PedersenParameters(p=p, q=q, g=g, h=h)
    params.validate()
    return params


DEFAULT_PARAMETERS = _build_default_parameters()


def pedersen_commit(value: int, blind: int, params: PedersenParameters) -> int:
    """Return the Pedersen commitment ``g^value · h^blind (mod p)``."""
    g_v = pow(params.g, value % params.q, params.p)
    h_r = pow(params.h, blind % params.q, params.p)
    return (g_v * h_r) % params.p


@dataclass(frozen=True)
class SchnorrProof:
    """Non-interactive Schnorr proof ``(challenge, response)`` over ``h``."""

    challenge: int
    response: int


def _fiat_shamir(params: PedersenParameters, *points: int, context: bytes) -> int:
    """Domain-separated SHA-256 Fiat-Shamir challenge, reduced mod q."""
    hasher = hashlib.sha256()
    hasher.update(b"director-ai/zk-attestation/schnorr/v1\x00")
    for value in (params.p, params.q, params.g, params.h, *points):
        encoded = value.to_bytes((value.bit_length() + 7) // 8 or 1, "big")
        hasher.update(len(encoded).to_bytes(4, "big"))
        hasher.update(encoded)
    hasher.update(len(context).to_bytes(4, "big"))
    hasher.update(context)
    return int.from_bytes(hasher.digest(), "big") % params.q


def prove_blind_knowledge(
    public: int,
    blind: int,
    params: PedersenParameters,
    *,
    context: bytes = b"",
    nonce: int | None = None,
) -> SchnorrProof:
    """Prove knowledge of ``blind`` such that ``public = h^blind (mod p)``."""
    k = secrets.randbelow(params.q - 1) + 1 if nonce is None else nonce % params.q
    commitment = pow(params.h, k, params.p)
    challenge = _fiat_shamir(params, public, commitment, context=context)
    response = (k + challenge * blind) % params.q
    return SchnorrProof(challenge=challenge, response=response)


def verify_blind_knowledge(
    public: int,
    proof: SchnorrProof,
    params: PedersenParameters,
    *,
    context: bytes = b"",
) -> bool:
    """Verify a :func:`prove_blind_knowledge` proof for ``public = h^blind``."""
    if not 0 <= proof.response < params.q:
        return False
    if not 0 <= proof.challenge < params.q:
        return False
    # t = h^s · public^{-e}; accept iff Fiat-Shamir(t) reproduces the challenge.
    public_inv = pow(public, params.p - 2, params.p)
    commitment = (
        pow(params.h, proof.response, params.p)
        * pow(public_inv, proof.challenge, params.p)
    ) % params.p
    expected = _fiat_shamir(params, public, commitment, context=context)
    return expected == proof.challenge


@dataclass(frozen=True)
class AggregateAttestation:
    """Published artefact: per-sample commitments, aggregate, and the proof."""

    commitments: tuple[int, ...]
    aggregate_scaled: int
    scale: int
    proof: SchnorrProof
    statement_kind: str


@dataclass
class SchnorrAttestationBackend:
    """Zero-knowledge backend hiding per-sample values behind Pedersen commitments.

    Parameters
    ----------
    parameters:
        The Pedersen group; defaults to the verified 2048-bit safe-prime group.
    rng:
        Object exposing ``randbelow(n)``; defaults to :mod:`secrets`. Tests inject
        a deterministic source.
    """

    parameters: PedersenParameters = field(default_factory=lambda: DEFAULT_PARAMETERS)
    rng: object = field(default=None)
    kind: str = field(default="schnorr-pedersen", init=False)

    def _randbelow(self, n: int) -> int:
        if self.rng is not None:
            return int(self.rng.randbelow(n))  # type: ignore[attr-defined]
        return secrets.randbelow(n)

    def prove(
        self,
        statement: AttestationStatement,
        samples: Sequence[HistorySample],
        *,
        context: bytes = b"",
    ) -> AggregateAttestation:
        """Commit each sample's contribution and prove the aggregate opening."""
        if not samples:
            raise ValueError("samples must be non-empty")
        q = self.parameters.q
        commitments: list[int] = []
        total_value = 0
        total_blind = 0
        for sample in samples:
            value = round(statement.evaluate_sample(sample) * _FIXED_POINT_SCALE)
            if value < 0:
                raise ValueError("statement contributions must be non-negative")
            blind = self._randbelow(q - 1) + 1
            commitments.append(pedersen_commit(value, blind, self.parameters))
            total_value += value
            total_blind = (total_blind + blind) % q
        # Y = (∏ C_i) · g^{-A} = h^R; prove knowledge of R = total_blind.
        aggregate_commitment = 1
        for c in commitments:
            aggregate_commitment = (aggregate_commitment * c) % self.parameters.p
        residual = (
            aggregate_commitment
            * pow(self.parameters.g, (-total_value) % q, self.parameters.p)
        ) % self.parameters.p
        proof = prove_blind_knowledge(
            residual, total_blind, self.parameters, context=context
        )
        return AggregateAttestation(
            commitments=tuple(commitments),
            aggregate_scaled=total_value,
            scale=_FIXED_POINT_SCALE,
            proof=proof,
            statement_kind=statement.kind,
        )

    def verify(
        self,
        statement: AttestationStatement,
        proof: object,
        *,
        context: bytes = b"",
    ) -> tuple[bool, str]:
        """Check the Schnorr opening and the statement threshold."""
        if not isinstance(proof, AggregateAttestation):
            return False, "wrong_proof_type"
        if proof.statement_kind != statement.kind:
            return False, "statement_kind_mismatch"
        if proof.scale <= 0:
            return False, "non_positive_scale"
        if not proof.commitments:
            return False, "no_commitments"
        if proof.aggregate_scaled < 0:
            return False, "negative_aggregate"
        p, q, g = self.parameters.p, self.parameters.q, self.parameters.g
        for c in proof.commitments:
            if not 0 < c < p:
                return False, "commitment_out_of_range"
        aggregate_commitment = 1
        for c in proof.commitments:
            aggregate_commitment = (aggregate_commitment * c) % p
        residual = (aggregate_commitment * pow(g, (-proof.aggregate_scaled) % q, p)) % p
        if not verify_blind_knowledge(
            residual, proof.proof, self.parameters, context=context
        ):
            return False, "schnorr_proof_invalid"
        aggregate = proof.aggregate_scaled / proof.scale
        if not statement.accepts(aggregate, len(proof.commitments)):
            return False, "statement_threshold_not_met"
        return True, ""
