# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Pedersen + Schnorr ZK attestation tests

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from director_ai.core.zk_attestation.schnorr import (
    DEFAULT_PARAMETERS,
    PedersenParameters,
    SchnorrAttestationBackend,
    SchnorrProof,
    _hash_to_subgroup,
    _is_probable_prime,
    pedersen_commit,
    prove_blind_knowledge,
    verify_blind_knowledge,
)
from director_ai.core.zk_attestation.statements import MinimumCoherence


class _DetRNG:
    """Deterministic randbelow source for reproducible tests."""

    def __init__(self, seed: int):
        self._r = random.Random(seed)

    def randbelow(self, n: int) -> int:
        return self._r.randrange(n)


def _samples(coherence: float, n: int) -> list[dict]:
    return [{"coherence": coherence} for _ in range(n)]


# ── primality self-check ─────────────────────────────────────────────────────


@pytest.mark.parametrize("prime", [2, 3, 5, 97, 7919, 104729])
def test_is_probable_prime_accepts_primes(prime):
    assert _is_probable_prime(prime)


@pytest.mark.parametrize("composite", [0, 1, 4, 9, 100, 7917, 104730, 1763, 2021])
def test_is_probable_prime_rejects_composites(composite):
    # 1763 = 41·43 and 2021 = 43·47 are coprime to the small-prime sieve, so they
    # exercise the Miller-Rabin witness loop, not the trial-division path.
    assert not _is_probable_prime(composite)


# ── parameters ───────────────────────────────────────────────────────────────


def test_default_parameters_validate():
    DEFAULT_PARAMETERS.validate()  # must not raise
    assert DEFAULT_PARAMETERS.p.bit_length() >= 2048
    assert DEFAULT_PARAMETERS.q == (DEFAULT_PARAMETERS.p - 1) // 2


def test_validate_rejects_wrong_q():
    bad = replace(DEFAULT_PARAMETERS, q=DEFAULT_PARAMETERS.q + 2)
    with pytest.raises(ValueError, match="q must equal"):
        bad.validate()


def test_validate_rejects_non_prime_p():
    p = DEFAULT_PARAMETERS.p + 1  # even, not prime
    bad = PedersenParameters(p=p, q=(p - 1) // 2, g=4, h=9)
    with pytest.raises(ValueError):
        bad.validate()


def test_validate_rejects_generator_outside_subgroup():
    # A primitive root (order p-1, not q) is not in the order-q subgroup.
    bad = replace(DEFAULT_PARAMETERS, h=DEFAULT_PARAMETERS.p - 1)
    with pytest.raises(ValueError, match="order-q subgroup"):
        bad.validate()


def test_validate_rejects_non_prime_q():
    # p = 13 is prime but q = 6 is not — a prime that is not a *safe* prime.
    bad = PedersenParameters(p=13, q=6, g=4, h=9)
    with pytest.raises(ValueError, match="q is not prime"):
        bad.validate()


def test_validate_rejects_generator_out_of_range():
    bad = replace(DEFAULT_PARAMETERS, g=1)  # 1 is not a valid generator
    with pytest.raises(ValueError, match="generator out of range"):
        bad.validate()


def test_hash_to_subgroup_retries_on_trivial_square():
    # In the tiny group p = 7 (q = 3) a candidate ≡ p-1 squares to 1 and forces a
    # retry; sweeping seeds exercises that retry path. Every result is a valid
    # order-3 subgroup element.
    for i in range(24):
        h = _hash_to_subgroup(bytes([i]), 7)
        assert pow(h, 3, 7) == 1


def test_validate_rejects_equal_generators():
    bad = replace(DEFAULT_PARAMETERS, h=DEFAULT_PARAMETERS.g)
    with pytest.raises(ValueError, match="distinct"):
        bad.validate()


def test_hash_to_subgroup_is_in_subgroup():
    h = _hash_to_subgroup(b"seed", DEFAULT_PARAMETERS.p)
    assert pow(h, DEFAULT_PARAMETERS.q, DEFAULT_PARAMETERS.p) == 1


# ── Pedersen homomorphism + hiding ───────────────────────────────────────────


def test_pedersen_is_additively_homomorphic():
    p, q = DEFAULT_PARAMETERS.p, DEFAULT_PARAMETERS.q
    v1, r1, v2, r2 = 7, 12345, 11, 67890
    c1 = pedersen_commit(v1, r1, DEFAULT_PARAMETERS)
    c2 = pedersen_commit(v2, r2, DEFAULT_PARAMETERS)
    combined = (c1 * c2) % p
    assert combined == pedersen_commit(v1 + v2, (r1 + r2) % q, DEFAULT_PARAMETERS)


def test_pedersen_hides_value_under_different_blinds():
    c1 = pedersen_commit(5, 111, DEFAULT_PARAMETERS)
    c2 = pedersen_commit(5, 222, DEFAULT_PARAMETERS)
    assert c1 != c2  # same value, different commitments


# ── Schnorr proof of knowledge ───────────────────────────────────────────────


def test_schnorr_completeness():
    p = DEFAULT_PARAMETERS.p
    blind = 999_983
    public = pow(DEFAULT_PARAMETERS.h, blind, p)
    proof = prove_blind_knowledge(public, blind, DEFAULT_PARAMETERS, context=b"ctx")
    assert verify_blind_knowledge(public, proof, DEFAULT_PARAMETERS, context=b"ctx")


def test_schnorr_rejects_wrong_blind():
    p = DEFAULT_PARAMETERS.p
    public = pow(DEFAULT_PARAMETERS.h, 12345, p)
    forged = prove_blind_knowledge(public, 54321, DEFAULT_PARAMETERS)  # wrong witness
    assert not verify_blind_knowledge(public, forged, DEFAULT_PARAMETERS)


def test_schnorr_rejects_tampered_response():
    p = DEFAULT_PARAMETERS.p
    blind = 777
    public = pow(DEFAULT_PARAMETERS.h, blind, p)
    proof = prove_blind_knowledge(public, blind, DEFAULT_PARAMETERS)
    tampered = SchnorrProof(challenge=proof.challenge, response=proof.response ^ 1)
    assert not verify_blind_knowledge(public, tampered, DEFAULT_PARAMETERS)


def test_schnorr_context_binding():
    p = DEFAULT_PARAMETERS.p
    blind = 4242
    public = pow(DEFAULT_PARAMETERS.h, blind, p)
    proof = prove_blind_knowledge(public, blind, DEFAULT_PARAMETERS, context=b"alice")
    assert not verify_blind_knowledge(public, proof, DEFAULT_PARAMETERS, context=b"bob")


def test_schnorr_rejects_out_of_range_response():
    public = pow(DEFAULT_PARAMETERS.h, 5, DEFAULT_PARAMETERS.p)
    bad = SchnorrProof(challenge=1, response=DEFAULT_PARAMETERS.q)  # == q, out of range
    assert not verify_blind_knowledge(public, bad, DEFAULT_PARAMETERS)


def test_schnorr_rejects_out_of_range_challenge():
    public = pow(DEFAULT_PARAMETERS.h, 5, DEFAULT_PARAMETERS.p)
    bad = SchnorrProof(challenge=DEFAULT_PARAMETERS.q, response=1)
    assert not verify_blind_knowledge(public, bad, DEFAULT_PARAMETERS)


# ── backend: completeness + soundness ────────────────────────────────────────


def _backend() -> SchnorrAttestationBackend:
    return SchnorrAttestationBackend(rng=_DetRNG(7))


def test_backend_accepts_satisfying_statement():
    backend = _backend()
    stmt = MinimumCoherence(name="coh", threshold=0.8, samples_min=3)
    proof = backend.prove(stmt, _samples(0.9, 4))
    ok, reason = backend.verify(stmt, proof)
    assert ok and reason == ""


def test_backend_rejects_unsatisfying_statement():
    backend = _backend()
    stmt = MinimumCoherence(name="coh", threshold=0.8, samples_min=3)
    proof = backend.prove(stmt, _samples(0.5, 4))  # mean 0.5 < 0.8
    ok, reason = backend.verify(stmt, proof)
    assert not ok and reason == "statement_threshold_not_met"


def test_backend_hides_values_across_runs():
    backend = _backend()
    stmt = MinimumCoherence(name="coh", threshold=0.8, samples_min=3)
    p1 = backend.prove(stmt, _samples(0.9, 3))
    p2 = backend.prove(stmt, _samples(0.9, 3))
    # identical inputs, different commitments (randomised blinds) — unlinkable
    assert p1.commitments != p2.commitments
    assert p1.aggregate_scaled == p2.aggregate_scaled


def test_backend_rejects_tampered_aggregate():
    backend = _backend()
    stmt = MinimumCoherence(name="coh", threshold=0.8, samples_min=3)
    proof = backend.prove(stmt, _samples(0.9, 4))
    forged = replace(proof, aggregate_scaled=proof.aggregate_scaled + 1)
    ok, reason = backend.verify(stmt, forged)
    assert not ok and reason == "schnorr_proof_invalid"


def test_backend_rejects_tampered_commitment():
    backend = _backend()
    stmt = MinimumCoherence(name="coh", threshold=0.8, samples_min=3)
    proof = backend.prove(stmt, _samples(0.9, 4))
    commits = list(proof.commitments)
    commits[0] = (commits[0] * 2) % DEFAULT_PARAMETERS.p
    forged = replace(proof, commitments=tuple(commits))
    ok, reason = backend.verify(stmt, forged)
    assert not ok and reason == "schnorr_proof_invalid"


def test_backend_rejects_wrong_proof_type():
    backend = _backend()
    stmt = MinimumCoherence(name="coh", threshold=0.8, samples_min=3)
    ok, reason = backend.verify(stmt, object())
    assert not ok and reason == "wrong_proof_type"


def test_backend_rejects_statement_kind_mismatch():
    backend = _backend()
    stmt = MinimumCoherence(name="coh", threshold=0.8, samples_min=3)
    proof = backend.prove(stmt, _samples(0.9, 3))
    forged = replace(proof, statement_kind="something_else")
    ok, reason = backend.verify(stmt, forged)
    assert not ok and reason == "statement_kind_mismatch"


def test_backend_rejects_commitment_out_of_range():
    backend = _backend()
    stmt = MinimumCoherence(name="coh", threshold=0.8, samples_min=3)
    proof = backend.prove(stmt, _samples(0.9, 3))
    forged = replace(proof, commitments=(DEFAULT_PARAMETERS.p,) + proof.commitments[1:])
    ok, reason = backend.verify(stmt, forged)
    assert not ok and reason == "commitment_out_of_range"


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda pr: replace(pr, scale=0), "non_positive_scale"),
        (lambda pr: replace(pr, commitments=()), "no_commitments"),
        (lambda pr: replace(pr, aggregate_scaled=-1), "negative_aggregate"),
    ],
)
def test_backend_field_validation(mutate, expected):
    backend = _backend()
    stmt = MinimumCoherence(name="coh", threshold=0.8, samples_min=3)
    proof = backend.prove(stmt, _samples(0.9, 3))
    ok, reason = backend.verify(stmt, mutate(proof))
    assert not ok and reason == expected


def test_backend_empty_samples_raises():
    backend = _backend()
    stmt = MinimumCoherence(name="coh", threshold=0.8, samples_min=1)
    with pytest.raises(ValueError, match="non-empty"):
        backend.prove(stmt, [])


def test_backend_negative_contribution_raises():
    backend = _backend()
    stmt = MinimumCoherence(name="coh", threshold=0.8, samples_min=1)
    with pytest.raises(ValueError, match="non-negative"):
        backend.prove(stmt, [{"coherence": -0.5}])


def test_backend_default_rng_roundtrip():
    # Default rng (secrets) path — no injected source.
    backend = SchnorrAttestationBackend()
    stmt = MinimumCoherence(name="coh", threshold=0.8, samples_min=3)
    proof = backend.prove(stmt, _samples(0.9, 4))
    ok, reason = backend.verify(stmt, proof)
    assert ok and reason == ""


def test_backend_context_binding():
    backend = _backend()
    stmt = MinimumCoherence(name="coh", threshold=0.8, samples_min=3)
    proof = backend.prove(stmt, _samples(0.9, 3), context=b"tenant-a")
    ok_same, _ = backend.verify(stmt, proof, context=b"tenant-a")
    ok_diff, reason = backend.verify(stmt, proof, context=b"tenant-b")
    assert ok_same
    assert not ok_diff and reason == "schnorr_proof_invalid"
