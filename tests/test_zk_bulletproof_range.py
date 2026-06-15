# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Bulletproof range-proof attestation tests

from __future__ import annotations

import math
from dataclasses import replace

import pytest

pytest.importorskip("backfire_kernel")

from director_ai.core.zk_attestation.bulletproof_range import (  # noqa: E402
    BulletproofRangeBackend,
    RangeAttestation,
    min_aggregate_for,
)
from director_ai.core.zk_attestation.statements import (  # noqa: E402
    MaximumHaltRate,
    MinimumCoherence,
)


def _samples(coherence: float, n: int) -> list[dict]:
    return [{"coherence": coherence} for _ in range(n)]


_STMT = MinimumCoherence(name="coh", threshold=0.8, samples_min=4)


# ── min_aggregate_for ────────────────────────────────────────────────────────


def test_min_aggregate_for_minimum_coherence():
    scale = 1_000_000
    assert min_aggregate_for(_STMT, 10, scale) == math.ceil(0.8 * 10 * scale)


def test_min_aggregate_for_unsupported_statement_raises():
    stmt = MaximumHaltRate(name="halt", max_rate=0.1, samples_min=4)
    with pytest.raises(NotImplementedError):
        min_aggregate_for(stmt, 10, 1_000_000)


# ── construction validation ──────────────────────────────────────────────────


@pytest.mark.parametrize("bits", [7, 0, 128, 33])
def test_invalid_bits_rejected(bits):
    with pytest.raises(ValueError, match="bits"):
        BulletproofRangeBackend(bits=bits)


def test_invalid_scale_rejected():
    with pytest.raises(ValueError, match="scale"):
        BulletproofRangeBackend(scale=0)


# ── completeness ─────────────────────────────────────────────────────────────


def test_satisfying_statement_proves_and_verifies():
    backend = BulletproofRangeBackend()
    proof = backend.prove(_STMT, _samples(0.9, 8))  # mean 0.9 >= 0.8
    assert isinstance(proof, RangeAttestation)
    ok, reason = backend.verify(_STMT, proof)
    assert ok and reason == ""


def test_exactly_at_threshold_verifies():
    backend = BulletproofRangeBackend()
    proof = backend.prove(_STMT, _samples(0.8, 8))  # mean exactly 0.8
    ok, reason = backend.verify(_STMT, proof)
    assert ok and reason == ""


# ── soundness: false statement is unprovable ────────────────────────────────


def test_unsatisfying_statement_cannot_be_proven():
    backend = BulletproofRangeBackend()
    with pytest.raises(ValueError, match="threshold not met"):
        backend.prove(_STMT, _samples(0.5, 8))  # mean 0.5 < 0.8


# ── hiding: no aggregate disclosed ──────────────────────────────────────────


def test_artifact_does_not_carry_the_aggregate():
    backend = BulletproofRangeBackend()
    proof = backend.prove(_STMT, _samples(0.9, 8))
    assert not hasattr(proof, "aggregate")
    assert not hasattr(proof, "aggregate_scaled")
    # only per-sample commitments, the proof blob, and the public threshold
    assert len(proof.commitments) == 8
    assert all(len(c) == 32 for c in proof.commitments)


def test_commitments_differ_across_runs():
    backend = BulletproofRangeBackend()
    p1 = backend.prove(_STMT, _samples(0.9, 8))
    p2 = backend.prove(_STMT, _samples(0.9, 8))
    assert p1.commitments != p2.commitments  # randomised blinds → unlinkable


# ── soundness: tampering rejected ───────────────────────────────────────────


def test_tampered_commitment_rejected():
    backend = BulletproofRangeBackend()
    proof = backend.prove(_STMT, _samples(0.9, 8))
    commits = list(proof.commitments)
    commits[0] = bytes(32)  # zero point, not the real commitment
    forged = replace(proof, commitments=tuple(commits))
    ok, reason = backend.verify(_STMT, forged)
    assert not ok and reason == "range_proof_invalid"


def test_tampered_proof_bytes_rejected():
    backend = BulletproofRangeBackend()
    proof = backend.prove(_STMT, _samples(0.9, 8))
    forged = replace(proof, proof=proof.proof[:-1] + bytes([proof.proof[-1] ^ 1]))
    ok, reason = backend.verify(_STMT, forged)
    assert not ok and reason == "range_proof_invalid"


def test_context_binding():
    backend = BulletproofRangeBackend()
    proof = backend.prove(_STMT, _samples(0.9, 8), context=b"tenant-a")
    assert backend.verify(_STMT, proof, context=b"tenant-a")[0]
    ok, reason = backend.verify(_STMT, proof, context=b"tenant-b")
    assert not ok and reason == "range_proof_invalid"


# ── verify-side field validation ────────────────────────────────────────────


def test_verify_wrong_proof_type():
    ok, reason = BulletproofRangeBackend().verify(_STMT, object())
    assert not ok and reason == "wrong_proof_type"


def test_verify_statement_kind_mismatch():
    backend = BulletproofRangeBackend()
    proof = backend.prove(_STMT, _samples(0.9, 8))
    forged = replace(proof, statement_kind="other")
    ok, reason = backend.verify(_STMT, forged)
    assert not ok and reason == "statement_kind_mismatch"


def test_verify_threshold_mismatch():
    backend = BulletproofRangeBackend()
    proof = backend.prove(_STMT, _samples(0.9, 8))
    forged = replace(proof, threshold_scaled=proof.threshold_scaled + 1)
    ok, reason = backend.verify(_STMT, forged)
    assert not ok and reason == "threshold_mismatch"


def test_verify_too_few_samples():
    backend = BulletproofRangeBackend()
    # statement requires 4; prove with 8 then claim samples_min higher than count
    proof = backend.prove(_STMT, _samples(0.9, 8))
    forged = replace(proof, samples_min=99)
    ok, reason = backend.verify(_STMT, forged)
    assert not ok and reason == "too_few_samples"


def test_verify_no_commitments():
    backend = BulletproofRangeBackend()
    proof = backend.prove(_STMT, _samples(0.9, 8))
    forged = replace(proof, commitments=())
    ok, reason = backend.verify(_STMT, forged)
    assert not ok and reason == "no_commitments"


def test_verify_invalid_bits():
    backend = BulletproofRangeBackend()
    proof = backend.prove(_STMT, _samples(0.9, 8))
    forged = replace(proof, bits=7)
    ok, reason = backend.verify(_STMT, forged)
    assert not ok and reason == "invalid_bits"


def test_prove_empty_samples_raises():
    with pytest.raises(ValueError, match="non-empty"):
        BulletproofRangeBackend().prove(_STMT, [])


def test_prove_negative_contribution_raises():
    backend = BulletproofRangeBackend()
    with pytest.raises(ValueError, match="non-negative"):
        backend.prove(_STMT, [{"coherence": -0.1}] * 4)
