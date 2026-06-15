# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — property fuzz tests for the Schnorr / Bulletproof backends

"""Property tests for the zero-knowledge attestation backends.

Random inputs probe the two invariants that matter for a proof system:
*completeness* (an honest proof of a true statement always verifies) and
*soundness* (any tampering, or a false statement, never verifies).
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from director_ai.core.zk_attestation.schnorr import (
    DEFAULT_PARAMETERS,
    SchnorrAttestationBackend,
    pedersen_commit,
    prove_blind_knowledge,
    verify_blind_knowledge,
)
from director_ai.core.zk_attestation.statements import MinimumCoherence

_COH = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_SCALARS = st.integers(min_value=0, max_value=10**9)


class _DetRNG:
    def __init__(self, seed: int):
        self._r = random.Random(seed)

    def randbelow(self, n: int) -> int:
        return self._r.randrange(n)


def _samples(values):
    return [{"coherence": v} for v in values]


# ── Pedersen homomorphism ────────────────────────────────────────────────────


@given(v1=_SCALARS, r1=_SCALARS, v2=_SCALARS, r2=_SCALARS)
@settings(max_examples=80, deadline=None)
def test_pedersen_homomorphic_under_random_values(v1, r1, v2, r2):
    p, q = DEFAULT_PARAMETERS.p, DEFAULT_PARAMETERS.q
    left = (
        pedersen_commit(v1, r1, DEFAULT_PARAMETERS)
        * pedersen_commit(v2, r2, DEFAULT_PARAMETERS)
    ) % p
    right = pedersen_commit(v1 + v2, (r1 + r2) % q, DEFAULT_PARAMETERS)
    assert left == right


# ── Schnorr proof of knowledge ───────────────────────────────────────────────


@given(blind=st.integers(min_value=1, max_value=10**12), ctx=st.binary(max_size=24))
@settings(max_examples=40, deadline=None)
def test_schnorr_blind_roundtrip_and_context_binding(blind, ctx):
    public = pow(DEFAULT_PARAMETERS.h, blind, DEFAULT_PARAMETERS.p)
    proof = prove_blind_knowledge(public, blind, DEFAULT_PARAMETERS, context=ctx)
    assert verify_blind_knowledge(public, proof, DEFAULT_PARAMETERS, context=ctx)
    # any other context must reject (Fiat-Shamir binds the transcript)
    other = ctx + b"x"
    assert not verify_blind_knowledge(public, proof, DEFAULT_PARAMETERS, context=other)


# ── Schnorr attestation backend ──────────────────────────────────────────────


@given(
    values=st.lists(_COH, min_size=1, max_size=8),
    seed=st.integers(min_value=1, max_value=10**6),
)
@settings(max_examples=15, deadline=None)
def test_schnorr_backend_matches_statement_and_rejects_tamper(values, seed):
    backend = SchnorrAttestationBackend(rng=_DetRNG(seed))
    stmt = MinimumCoherence(name="c", threshold=0.6, samples_min=1)
    proof = backend.prove(stmt, _samples(values))
    ok, _ = backend.verify(stmt, proof)
    # completeness: verify accepts iff the statement is genuinely satisfied
    mean = sum(values) / len(values)
    assert ok is (mean >= 0.6)
    # soundness: shifting the revealed aggregate breaks the Schnorr opening
    forged = replace(proof, aggregate_scaled=proof.aggregate_scaled + 1)
    assert backend.verify(stmt, forged)[0] is False


# ── Bulletproof range backend ────────────────────────────────────────────────


@given(
    values=st.lists(
        st.floats(min_value=0.8, max_value=1.0, allow_nan=False), min_size=4, max_size=8
    )
)
@settings(max_examples=30, deadline=None)
def test_bulletproof_proves_satisfying_and_rejects_tamper(values):
    pytest.importorskip("backfire_kernel")
    from director_ai.core.zk_attestation.bulletproof_range import (
        BulletproofRangeBackend,
    )

    stmt = MinimumCoherence(name="c", threshold=0.7, samples_min=1)
    assume(sum(values) / len(values) >= 0.7)  # provable region
    backend = BulletproofRangeBackend()
    proof = backend.prove(stmt, _samples(values))
    assert backend.verify(stmt, proof)[0] is True
    forged = replace(proof, commitments=(bytes(32),) + proof.commitments[1:])
    assert backend.verify(stmt, forged)[0] is False


@given(
    values=st.lists(
        st.floats(min_value=0.0, max_value=0.3, allow_nan=False), min_size=4, max_size=8
    )
)
@settings(max_examples=20, deadline=None)
def test_bulletproof_cannot_prove_false_statement(values):
    pytest.importorskip("backfire_kernel")
    from director_ai.core.zk_attestation.bulletproof_range import (
        BulletproofRangeBackend,
    )

    stmt = MinimumCoherence(name="c", threshold=0.7, samples_min=1)
    assume(sum(values) / len(values) < 0.7)  # genuinely unprovable
    with pytest.raises(ValueError):
        BulletproofRangeBackend().prove(stmt, _samples(values))
