# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2020-2026 Miroslav Sotek
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Project: Director-Class AI
# File: zk attestation property fuzz tests
# License: AGPL-3.0-or-later; commercial terms available

"""Property tests for cross-org attestation commitments and passports."""

from __future__ import annotations

from dataclasses import dataclass, replace

from hypothesis import given, settings
from hypothesis import strategies as st

from director_ai.core.zk_attestation import (
    CommitmentBackend,
    CommitmentProof,
    CrossOrgPassport,
    MinimumCoherence,
    PassportIssuer,
    PassportVerifier,
    commit_samples,
    open_indices,
    verify_opening,
)

_KEY_A = b"A" * 32
_KEY_B = b"B" * 32
_TEXT = "abcdefghijklmnopqrstuvwxyz0123456789-._:/"
_SAMPLE = st.fixed_dictionaries(
    {
        "coherence": st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
        "halted": st.booleans(),
        "domain": st.sampled_from(("math", "physics", "code", "ops")),
        "duration_seconds": st.integers(min_value=0, max_value=86_400),
        "breakout": st.booleans(),
        "label": st.text(alphabet=_TEXT, min_size=0, max_size=16),
    },
)
_SAMPLES = st.lists(_SAMPLE, min_size=1, max_size=12)
_PASS_SAMPLES = st.lists(
    st.fixed_dictionaries(
        {
            "coherence": st.floats(
                min_value=0.25,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
            "halted": st.just(False),
            "domain": st.sampled_from(("math", "physics", "code", "ops")),
            "duration_seconds": st.integers(min_value=0, max_value=86_400),
            "breakout": st.just(False),
            "label": st.text(alphabet=_TEXT, min_size=0, max_size=16),
        },
    ),
    min_size=1,
    max_size=12,
)
_ID_TEXT = st.text(alphabet=_TEXT, min_size=1, max_size=24)


@dataclass
class _CycleRng:
    seed: int

    def token_bytes(self, n: int) -> bytes:
        out = bytearray(n)
        for i in range(n):
            self.seed = (1664525 * self.seed + 1013904223) & 0xFFFFFFFF
            out[i] = (self.seed >> 24) & 0xFF
        return bytes(out)


def _always_one(_sample: object) -> float:
    return 1.0


def _flipped_mac(mac: str) -> str:
    head = "0" if mac[0] != "0" else "1"
    return head + mac[1:]


@settings(max_examples=60, deadline=None)
@given(data=st.data(), samples=_SAMPLES, seed=st.integers(min_value=1, max_value=9999))
def test_merkle_openings_reject_revealed_sample_changes(data, samples, seed):
    commitment, leaves, blinds = commit_samples(
        samples,
        key=_KEY_A,
        rng=_CycleRng(seed),
    )
    indices = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=len(samples) - 1),
            min_size=1,
            max_size=min(4, len(samples)),
            unique=True,
        ),
    )
    proof = open_indices(
        indices=indices,
        samples=samples,
        leaves=leaves,
        blinds=blinds,
        aggregate=float(len(samples)),
        commitment=commitment,
    )

    ok, reason = verify_opening(proof, key=_KEY_A, per_sample_evaluator=_always_one)
    assert ok, reason

    opened = dict(proof.opened)
    first_idx = next(iter(opened))
    blind_hex, _serialised, path = opened[first_idx]
    opened[first_idx] = (blind_hex, '{"coherence":1.0,"changed":true}', path)
    tampered = CommitmentProof(
        commitment=proof.commitment,
        opened=opened,
        aggregate=proof.aggregate,
        total_samples=proof.total_samples,
    )
    ok, reason = verify_opening(
        tampered,
        key=_KEY_A,
        per_sample_evaluator=_always_one,
    )
    assert not ok
    assert reason.startswith("merkle_mismatch_at_")


@settings(max_examples=60, deadline=None)
@given(samples=_PASS_SAMPLES, agent_id=_ID_TEXT, org_tail=_ID_TEXT)
def test_passport_verifier_accepts_generated_bundles_and_rejects_mac_changes(
    samples,
    agent_id,
    org_tail,
):
    org = f"org://{org_tail}"
    issuer = PassportIssuer(
        key=_KEY_A,
        issuing_org=org,
        default_backend=CommitmentBackend(key=_KEY_A, challenge_size=4),
        clock=lambda: 1_771_234_500,
    )
    verifier = PassportVerifier(
        issuer_keys={org: _KEY_A},
        backends={"commitment": CommitmentBackend(key=_KEY_A, challenge_size=4)},
    )
    passport = issuer.issue(
        agent_id=agent_id,
        samples=samples,
        statements=[MinimumCoherence(name="coherence", threshold=0.0, samples_min=1)],
    )

    verdict = verifier.verify(passport)
    assert verdict.accepted, verdict.failures
    assert verdict.signature_ok

    changed_mac = CrossOrgPassport(
        agent_id=passport.agent_id,
        issuing_org=passport.issuing_org,
        created_at=passport.created_at,
        entries=passport.entries,
        mac=_flipped_mac(passport.mac),
    )
    failed = verifier.verify(changed_mac)
    assert not failed.accepted
    assert not failed.signature_ok
    assert failed.failures == (("_passport", "mac_mismatch"),)


@settings(max_examples=60, deadline=None)
@given(samples=_PASS_SAMPLES, agent_id=_ID_TEXT, org_tail=_ID_TEXT)
def test_passport_verifier_reports_tampered_proof_failures(
    samples,
    agent_id,
    org_tail,
):
    org = f"org://{org_tail}"
    issuer = PassportIssuer(
        key=_KEY_A,
        issuing_org=org,
        default_backend=CommitmentBackend(key=_KEY_A, challenge_size=4),
        clock=lambda: 1_771_234_500,
    )
    verifier = PassportVerifier(
        issuer_keys={org: _KEY_A},
        backends={"commitment": CommitmentBackend(key=_KEY_A, challenge_size=4)},
    )
    passport = issuer.issue(
        agent_id=agent_id,
        samples=samples,
        statements=[MinimumCoherence(name="coherence", threshold=0.0, samples_min=1)],
    )
    proof = passport.entries[0].proof
    assert isinstance(proof, CommitmentProof)
    bad_proof = CommitmentProof(
        commitment=proof.commitment,
        opened=proof.opened,
        aggregate=0.0,
        total_samples=proof.total_samples,
    )
    bad_entry = replace(passport.entries[0], proof=bad_proof)
    changed_passport = CrossOrgPassport(
        agent_id=passport.agent_id,
        issuing_org=passport.issuing_org,
        created_at=passport.created_at,
        entries=(bad_entry,),
        mac=passport.mac,
    )

    verdict = verifier.verify(changed_passport)
    assert verdict.signature_ok
    assert not verdict.accepted
    assert verdict.failures
    assert verdict.failures[0][0] == "coherence"
    assert verdict.failures[0][1].startswith("aggregate_inconsistent")


@settings(max_examples=30, deadline=None)
@given(samples=_PASS_SAMPLES)
def test_commitment_backend_rejects_wrong_verifier_key(samples):
    issuer_backend = CommitmentBackend(key=_KEY_A, challenge_size=4)
    verifier_backend = CommitmentBackend(key=_KEY_B, challenge_size=4)
    statement = MinimumCoherence(name="coherence", threshold=0.0, samples_min=1)
    proof = issuer_backend.prove(statement, samples)

    ok, reason = verifier_backend.verify(statement, proof)
    assert not ok
    assert reason.startswith("merkle_mismatch_at_")
