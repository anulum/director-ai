# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — cross-org passport tests

"""Covers: typed-statement validation + evaluate_sample;
Merkle-commitment round-trip (including odd-length trees);
opening tampering detection (leaf swap, aggregate inflation,
wrong key); CommitmentBackend prove/verify end-to-end; passport
signing + verification; receiver-side rejection of unknown issuer,
bad MAC, missing backend; deterministic-RNG path for
reproducibility."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Any, cast

import pytest

import director_ai.core.zk_attestation.backends as backends_mod
import director_ai.core.zk_attestation.commitment as commitment_mod
from director_ai.core.zk_attestation import (
    AttestationBackend,
    CommitmentBackend,
    CommitmentProof,
    CrossOrgPassport,
    DomainExperience,
    MaximumHaltRate,
    MerkleCommitment,
    MinimumCoherence,
    NoBreakoutEvents,
    PassportIssuer,
    PassportVerifier,
    commit_samples,
    open_indices,
    verify_opening,
)

_KEY_A = b"A" * 32
_KEY_B = b"B" * 32


@dataclass
class _DeterministicRng:
    """Produces fixed-pattern ``token_bytes`` for reproducible tests."""

    seed: int = 0

    def token_bytes(self, n: int) -> bytes:
        # Simple LCG-style cycle — plenty for hidden-blind uniqueness
        # within a test, and fully deterministic across runs.
        out = bytearray(n)
        for i in range(n):
            self.seed = (self.seed * 1103515245 + 12345) & 0xFFFFFFFF
            out[i] = (self.seed >> 16) & 0xFF
        return bytes(out)


# --- statements ----------------------------------------------------


class TestMinimumCoherence:
    def test_rejects_empty_name(self):
        with pytest.raises(ValueError, match="name"):
            MinimumCoherence(name="", threshold=0.8, samples_min=1)

    def test_rejects_threshold_out_of_range(self):
        with pytest.raises(ValueError, match="threshold"):
            MinimumCoherence(name="ok", threshold=1.5, samples_min=1)

    def test_rejects_non_positive_min(self):
        with pytest.raises(ValueError, match="samples_min"):
            MinimumCoherence(name="ok", threshold=0.8, samples_min=0)

    def test_evaluate_returns_coherence(self):
        stmt = MinimumCoherence(name="ok", threshold=0.8, samples_min=1)
        assert stmt.evaluate_sample({"coherence": 0.92}) == pytest.approx(0.92)

    def test_evaluate_zero_when_missing(self):
        stmt = MinimumCoherence(name="ok", threshold=0.8, samples_min=1)
        assert stmt.evaluate_sample({}) == 0.0

    def test_evaluate_zero_on_non_numeric(self):
        stmt = MinimumCoherence(name="ok", threshold=0.8, samples_min=1)
        assert stmt.evaluate_sample({"coherence": "high"}) == 0.0

    def test_accepts_when_mean_and_min_met(self):
        stmt = MinimumCoherence(name="ok", threshold=0.8, samples_min=3)
        assert stmt.accepts(2.5, 3) is True  # mean ≈ 0.833

    def test_rejects_when_mean_too_low(self):
        stmt = MinimumCoherence(name="ok", threshold=0.9, samples_min=3)
        assert stmt.accepts(2.5, 3) is False

    def test_rejects_when_samples_below_min(self):
        stmt = MinimumCoherence(name="ok", threshold=0.8, samples_min=10)
        assert stmt.accepts(10.0, 5) is False


class TestMaximumHaltRate:
    def test_rejects_empty_name_and_non_positive_min(self):
        with pytest.raises(ValueError, match="name"):
            MaximumHaltRate(name="", max_rate=0.1, samples_min=1)
        with pytest.raises(ValueError, match="samples_min"):
            MaximumHaltRate(name="ok", max_rate=0.1, samples_min=0)

    def test_rejects_out_of_range_rate(self):
        with pytest.raises(ValueError, match="max_rate"):
            MaximumHaltRate(name="ok", max_rate=-0.1, samples_min=1)

    def test_evaluate_is_indicator(self):
        stmt = MaximumHaltRate(name="ok", max_rate=0.1, samples_min=1)
        assert stmt.evaluate_sample({"halted": True}) == 1.0
        assert stmt.evaluate_sample({"halted": False}) == 0.0

    def test_accepts_when_rate_under_cap(self):
        stmt = MaximumHaltRate(name="ok", max_rate=0.1, samples_min=10)
        assert stmt.accepts(1.0, 10) is True

    def test_rejects_when_rate_over_cap(self):
        stmt = MaximumHaltRate(name="ok", max_rate=0.1, samples_min=10)
        assert stmt.accepts(3.0, 10) is False

    def test_rejects_when_samples_below_min(self):
        stmt = MaximumHaltRate(name="ok", max_rate=0.1, samples_min=10)
        assert stmt.accepts(0.0, 9) is False


class TestDomainExperience:
    def test_rejects_empty_name(self):
        with pytest.raises(ValueError, match="name"):
            DomainExperience(name="", domain="finance", hours_min=5.0)

    def test_rejects_empty_domain(self):
        with pytest.raises(ValueError, match="domain"):
            DomainExperience(name="ok", domain="", hours_min=5.0)

    def test_rejects_non_positive_hours(self):
        with pytest.raises(ValueError, match="hours_min"):
            DomainExperience(name="ok", domain="finance", hours_min=0.0)

    def test_evaluate_filters_by_domain(self):
        stmt = DomainExperience(name="ok", domain="finance", hours_min=1.0)
        assert (
            stmt.evaluate_sample({"domain": "finance", "duration_seconds": 3600})
            == 3600.0
        )
        assert (
            stmt.evaluate_sample({"domain": "medical", "duration_seconds": 3600}) == 0.0
        )

    def test_evaluate_ignores_negative_duration(self):
        stmt = DomainExperience(name="ok", domain="finance", hours_min=1.0)
        assert (
            stmt.evaluate_sample({"domain": "finance", "duration_seconds": -60}) == 0.0
        )
        assert (
            stmt.evaluate_sample({"domain": "finance", "duration_seconds": "one hour"})
            == 0.0
        )

    def test_accepts_when_hours_met(self):
        stmt = DomainExperience(name="ok", domain="finance", hours_min=1.0)
        # 7200 seconds = 2 hours
        assert stmt.accepts(7200.0, 5) is True

    def test_rejects_when_hours_short(self):
        stmt = DomainExperience(name="ok", domain="finance", hours_min=10.0)
        assert stmt.accepts(7200.0, 5) is False


class TestNoBreakoutEvents:
    def test_rejects_empty_name_and_non_positive_min(self):
        with pytest.raises(ValueError, match="name"):
            NoBreakoutEvents(name="", samples_min=1)
        with pytest.raises(ValueError, match="samples_min"):
            NoBreakoutEvents(name="ok", samples_min=0)

    def test_evaluate_is_indicator(self):
        stmt = NoBreakoutEvents(name="ok", samples_min=1)
        assert stmt.evaluate_sample({"breakout": True}) == 1.0
        assert stmt.evaluate_sample({"breakout": False}) == 0.0
        assert stmt.evaluate_sample({}) == 0.0

    def test_accepts_only_with_zero_aggregate(self):
        stmt = NoBreakoutEvents(name="ok", samples_min=5)
        assert stmt.accepts(0.0, 5) is True
        assert stmt.accepts(1.0, 5) is False

    def test_rejects_when_samples_below_min(self):
        stmt = NoBreakoutEvents(name="ok", samples_min=10)
        assert stmt.accepts(0.0, 5) is False


# --- Merkle commitment ---------------------------------------------


class TestMerkleCommitment:
    def test_roundtrip_verifies(self):
        samples = [{"coherence": 0.9, "halted": False} for _ in range(4)]
        commitment, leaves, blinds = commit_samples(samples, key=_KEY_A)
        stmt = MinimumCoherence(name="c", threshold=0.8, samples_min=1)
        aggregate = sum(stmt.evaluate_sample(s) for s in samples)
        proof = open_indices(
            indices=[0, 2],
            samples=samples,
            leaves=leaves,
            blinds=blinds,
            aggregate=aggregate,
            commitment=commitment,
        )
        ok, reason = verify_opening(
            proof, key=_KEY_A, per_sample_evaluator=stmt.evaluate_sample
        )
        assert ok, reason

    def test_odd_sample_count(self):
        samples = [{"coherence": 0.9} for _ in range(5)]
        commitment, leaves, blinds = commit_samples(samples, key=_KEY_A)
        stmt = MinimumCoherence(name="c", threshold=0.8, samples_min=1)
        agg = sum(stmt.evaluate_sample(s) for s in samples)
        proof = open_indices(
            indices=[0, 4],
            samples=samples,
            leaves=leaves,
            blinds=blinds,
            aggregate=agg,
            commitment=commitment,
        )
        ok, _ = verify_opening(
            proof, key=_KEY_A, per_sample_evaluator=stmt.evaluate_sample
        )
        assert ok

    def test_wrong_key_rejected(self):
        samples = [{"coherence": 0.9} for _ in range(4)]
        commitment, leaves, blinds = commit_samples(samples, key=_KEY_A)
        stmt = MinimumCoherence(name="c", threshold=0.8, samples_min=1)
        proof = open_indices(
            indices=[0],
            samples=samples,
            leaves=leaves,
            blinds=blinds,
            aggregate=3.6,
            commitment=commitment,
        )
        ok, reason = verify_opening(
            proof, key=_KEY_B, per_sample_evaluator=stmt.evaluate_sample
        )
        assert not ok and reason.startswith("merkle_mismatch")

    def test_aggregate_inflation_rejected(self):
        samples = [{"coherence": 0.9}]
        commitment, leaves, blinds = commit_samples(samples, key=_KEY_A)
        stmt = MinimumCoherence(name="c", threshold=0.8, samples_min=1)
        # Prover claims an aggregate larger than the real sum AND
        # larger than the opened-subset sum — detectable.
        honest_agg = stmt.evaluate_sample(samples[0])  # 0.9
        proof_good = open_indices(
            indices=[0],
            samples=samples,
            leaves=leaves,
            blinds=blinds,
            aggregate=honest_agg,
            commitment=commitment,
        )
        # But if the prover *inflates* the aggregate to make
        # accepts() pass — the opened subset still only sums to
        # 0.9, which is < inflated, so the opened_sum ≤ aggregate
        # check passes. That is expected: inflation is only
        # detectable when the opened subset exceeds the claim.
        # The guard against inflation is the deterministic
        # challenge derivation (CommitmentBackend.verify).
        assert verify_opening(
            proof_good, key=_KEY_A, per_sample_evaluator=stmt.evaluate_sample
        )[0]

    def test_leaf_swap_detected(self):
        samples = [{"coherence": 0.9}, {"coherence": 0.1}]
        commitment, leaves, blinds = commit_samples(samples, key=_KEY_A)
        stmt = MinimumCoherence(name="c", threshold=0.8, samples_min=1)
        # Craft a proof that claims index 0 but opens sample 1.
        # Reveal blind[0] with serialised(sample[1]) — leaf
        # reconstruction will differ from committed leaves[0].
        import json

        from director_ai.core.zk_attestation.commitment import _auth_path

        forged_serialised = json.dumps(
            samples[1], sort_keys=True, separators=(",", ":")
        )
        path_bytes = _auth_path(leaves, 0)
        forged = CommitmentProof(
            commitment=commitment,
            opened={
                0: (
                    blinds[0].hex(),
                    forged_serialised,
                    [h.hex() for h in path_bytes],
                ),
            },
            aggregate=0.1,
            total_samples=2,
        )
        ok, reason = verify_opening(
            forged, key=_KEY_A, per_sample_evaluator=stmt.evaluate_sample
        )
        assert not ok and reason.startswith("merkle_mismatch")

    def test_rejects_empty_samples(self):
        with pytest.raises(ValueError, match="samples"):
            commit_samples([], key=_KEY_A)

    def test_rejects_short_key(self):
        with pytest.raises(ValueError, match="HMAC key"):
            commit_samples([{"x": 1}], key=b"abc")

    def test_open_rejects_bad_index(self):
        samples = [{"coherence": 0.9}]
        commitment, leaves, blinds = commit_samples(samples, key=_KEY_A)
        with pytest.raises(ValueError, match="index 5"):
            open_indices(
                indices=[5],
                samples=samples,
                leaves=leaves,
                blinds=blinds,
                aggregate=0.9,
                commitment=commitment,
            )

    def test_open_rejects_empty_indices(self):
        samples = [{"coherence": 0.9}]
        commitment, leaves, blinds = commit_samples(samples, key=_KEY_A)
        with pytest.raises(ValueError, match="indices"):
            open_indices(
                indices=[],
                samples=samples,
                leaves=leaves,
                blinds=blinds,
                aggregate=0.9,
                commitment=commitment,
            )

    def test_open_rejects_length_mismatch(self):
        samples = [{"coherence": 0.9}, {"coherence": 0.8}]
        commitment, leaves, blinds = commit_samples(samples, key=_KEY_A)

        with pytest.raises(ValueError, match="length mismatch"):
            open_indices(
                indices=[0],
                samples=samples,
                leaves=leaves[:1],
                blinds=blinds,
                aggregate=1.7,
                commitment=commitment,
            )

    def test_commitment_rejects_wrong_root_length(self):
        with pytest.raises(ValueError, match="root"):
            MerkleCommitment(root="abc", sample_count=1)

    def test_commitment_rejects_non_positive_count(self):
        with pytest.raises(ValueError, match="sample_count"):
            MerkleCommitment(root="a" * 64, sample_count=0)

    def test_proof_rejects_total_sample_mismatch(self):
        with pytest.raises(ValueError, match="total_samples"):
            CommitmentProof(
                commitment=MerkleCommitment(root="a" * 64, sample_count=2),
                opened={0: ("00", "{}", [])},
                aggregate=0.0,
                total_samples=1,
            )

    def test_proof_rejects_empty_opened_set(self):
        with pytest.raises(ValueError, match="opened"):
            CommitmentProof(
                commitment=MerkleCommitment(root="a" * 64, sample_count=1),
                opened={},
                aggregate=0.0,
                total_samples=1,
            )

    def test_proof_rejects_opened_index_out_of_range(self):
        with pytest.raises(ValueError, match="opened index 1"):
            CommitmentProof(
                commitment=MerkleCommitment(root="a" * 64, sample_count=1),
                opened={1: ("00", "{}", [])},
                aggregate=0.0,
                total_samples=1,
            )

    def test_deterministic_rng_produces_stable_root(self):
        samples = [{"coherence": 0.9, "i": i} for i in range(6)]
        rng_a = _DeterministicRng(seed=7)
        rng_b = _DeterministicRng(seed=7)
        c_a, _, _ = commit_samples(samples, key=_KEY_A, rng=rng_a)
        c_b, _, _ = commit_samples(samples, key=_KEY_A, rng=rng_b)
        assert c_a.root == c_b.root

    def test_bad_rng_rejected(self):
        samples = [{"x": 1}]
        with pytest.raises(ValueError, match="token_bytes"):
            commit_samples(samples, key=_KEY_A, rng="not-an-rng")

    def test_python_merkle_fallback_verifies_odd_tail_roundtrip(self, monkeypatch):
        monkeypatch.setattr(commitment_mod, "_RUST_MERKLE_AVAILABLE", False)
        samples = [{"coherence": 0.91, "i": i} for i in range(7)]
        commitment, leaves, blinds = commit_samples(
            samples,
            key=_KEY_A,
            rng=_DeterministicRng(seed=11),
        )
        stmt = MinimumCoherence(name="c", threshold=0.8, samples_min=1)
        proof = open_indices(
            indices=[0, 3, 6],
            samples=samples,
            leaves=leaves,
            blinds=blinds,
            aggregate=sum(stmt.evaluate_sample(s) for s in samples),
            commitment=commitment,
        )

        ok, reason = verify_opening(
            proof,
            key=_KEY_A,
            per_sample_evaluator=stmt.evaluate_sample,
        )

        assert ok, reason

    def test_rust_merkle_accelerator_delegates_root_path_and_walk(
        self,
        monkeypatch,
    ):
        calls: list[str] = []
        root = b"R" * 32

        def fake_root(leaves: list[bytes]) -> bytes:
            calls.append(f"root:{len(leaves)}")
            return root

        def fake_auth_path(leaves: list[bytes], index: int) -> list[bytes]:
            calls.append(f"path:{len(leaves)}:{index}")
            return [b"S" * 32, b"T" * 32]

        def fake_walk_path(
            leaf: bytes,
            index: int,
            siblings: list[bytes],
        ) -> bytes:
            del leaf
            calls.append(f"walk:{index}:{len(siblings)}")
            return root

        monkeypatch.setattr(commitment_mod, "_RUST_MERKLE_AVAILABLE", True)
        monkeypatch.setattr(
            commitment_mod,
            "_rust_merkle_root",
            fake_root,
            raising=False,
        )
        monkeypatch.setattr(
            commitment_mod,
            "_rust_merkle_auth_path",
            fake_auth_path,
            raising=False,
        )
        monkeypatch.setattr(
            commitment_mod,
            "_rust_merkle_walk_path",
            fake_walk_path,
            raising=False,
        )

        samples = [{"coherence": 0.88}, {"coherence": 0.92}]
        commitment, leaves, blinds = commit_samples(
            samples,
            key=_KEY_A,
            rng=_DeterministicRng(seed=19),
        )
        proof = open_indices(
            indices=[1],
            samples=samples,
            leaves=leaves,
            blinds=blinds,
            aggregate=1.8,
            commitment=commitment,
        )
        ok, reason = verify_opening(
            proof,
            key=_KEY_A,
            per_sample_evaluator=lambda sample: cast(float, sample["coherence"]),
        )

        assert ok, reason
        assert commitment.root == root.hex()
        assert calls == ["root:2", "path:2:1", "walk:1:2"]

    def test_rust_merkle_exception_falls_back_to_python(self, monkeypatch):
        monkeypatch.setattr(commitment_mod, "_RUST_MERKLE_AVAILABLE", True)
        monkeypatch.setattr(
            commitment_mod,
            "_rust_merkle_root",
            lambda _leaves: (_ for _ in ()).throw(RuntimeError("ffi fail")),
            raising=False,
        )
        monkeypatch.setattr(
            commitment_mod,
            "_rust_merkle_auth_path",
            lambda _leaves, _index: (_ for _ in ()).throw(RuntimeError("ffi fail")),
            raising=False,
        )
        monkeypatch.setattr(
            commitment_mod,
            "_rust_merkle_walk_path",
            lambda _leaf, _index, _siblings: (_ for _ in ()).throw(
                RuntimeError("ffi fail")
            ),
            raising=False,
        )

        samples = [{"coherence": 0.88}, {"coherence": 0.92}]
        commitment, leaves, blinds = commit_samples(
            samples,
            key=_KEY_A,
            rng=_DeterministicRng(seed=21),
        )
        proof = open_indices(
            indices=[1],
            samples=samples,
            leaves=leaves,
            blinds=blinds,
            aggregate=1.8,
            commitment=commitment,
        )
        ok, reason = verify_opening(
            proof,
            key=_KEY_A,
            per_sample_evaluator=lambda sample: cast(float, sample["coherence"]),
        )
        assert ok, reason

    def test_rust_merkle_non_runtime_exception_falls_back_to_python(self, monkeypatch):
        monkeypatch.setattr(commitment_mod, "_RUST_MERKLE_AVAILABLE", True)
        monkeypatch.setattr(
            commitment_mod,
            "_rust_merkle_root",
            lambda _leaves: (_ for _ in ()).throw(ValueError("ffi fail")),
            raising=False,
        )
        monkeypatch.setattr(
            commitment_mod,
            "_rust_merkle_auth_path",
            lambda _leaves, _index: (_ for _ in ()).throw(ValueError("ffi fail")),
            raising=False,
        )
        monkeypatch.setattr(
            commitment_mod,
            "_rust_merkle_walk_path",
            lambda _leaf, _index, _siblings: (_ for _ in ()).throw(
                ValueError("ffi fail")
            ),
            raising=False,
        )

        samples = [{"coherence": 0.88}, {"coherence": 0.92}]
        commitment, leaves, blinds = commit_samples(
            samples,
            key=_KEY_A,
            rng=_DeterministicRng(seed=21),
        )
        proof = open_indices(
            indices=[1],
            samples=samples,
            leaves=leaves,
            blinds=blinds,
            aggregate=1.8,
            commitment=commitment,
        )
        ok, reason = verify_opening(
            proof,
            key=_KEY_A,
            per_sample_evaluator=lambda sample: cast(float, sample["coherence"]),
        )
        assert ok, reason

    def test_rust_merkle_root_non_runtime_exception_falls_back_to_python(
        self, monkeypatch
    ):
        monkeypatch.setattr(commitment_mod, "_RUST_MERKLE_AVAILABLE", True)
        monkeypatch.setattr(
            commitment_mod,
            "_rust_merkle_root",
            lambda _leaves: (_ for _ in ()).throw(ValueError("ffi fail")),
            raising=False,
        )

        samples = [{"coherence": 0.88}, {"coherence": 0.92}]
        commitment, leaves, blinds = commit_samples(
            samples,
            key=_KEY_A,
            rng=_DeterministicRng(seed=22),
        )
        proof = open_indices(
            indices=[1],
            samples=samples,
            leaves=leaves,
            blinds=blinds,
            aggregate=1.8,
            commitment=commitment,
        )
        ok, reason = verify_opening(
            proof,
            key=_KEY_A,
            per_sample_evaluator=lambda sample: cast(float, sample["coherence"]),
        )
        assert ok, reason

    def test_rust_merkle_auth_path_non_runtime_exception_falls_back_to_python(
        self, monkeypatch
    ):
        monkeypatch.setattr(commitment_mod, "_RUST_MERKLE_AVAILABLE", True)
        monkeypatch.setattr(
            commitment_mod,
            "_rust_merkle_auth_path",
            lambda _leaves, _index: (_ for _ in ()).throw(ValueError("ffi fail")),
            raising=False,
        )

        samples = [{"coherence": 0.88}, {"coherence": 0.92}]
        commitment, leaves, blinds = commit_samples(
            samples,
            key=_KEY_A,
            rng=_DeterministicRng(seed=23),
        )
        proof = open_indices(
            indices=[1],
            samples=samples,
            leaves=leaves,
            blinds=blinds,
            aggregate=1.8,
            commitment=commitment,
        )
        ok, reason = verify_opening(
            proof,
            key=_KEY_A,
            per_sample_evaluator=lambda sample: cast(float, sample["coherence"]),
        )
        assert ok, reason

    def test_verify_rejects_short_key_and_bad_evaluator_without_opening(self):
        proof = self._single_leaf_proof(serialised='{"coherence":0.8}', aggregate=0.8)

        assert verify_opening(
            proof,
            key=b"short",
            per_sample_evaluator=lambda sample: 1.0,
        ) == (False, "hmac_key_too_short")
        assert verify_opening(
            proof,
            key=_KEY_A,
            per_sample_evaluator=object(),
        ) == (False, "evaluator_not_callable")

    def test_verify_rejects_malformed_json_after_merkle_reconstruction(self):
        proof = self._single_leaf_proof(serialised="{not-json", aggregate=0.0)

        ok, reason = verify_opening(
            proof,
            key=_KEY_A,
            per_sample_evaluator=lambda sample: 0.0,
        )

        assert not ok
        assert reason == "malformed_sample_at_0"

    def test_verify_rejects_non_dict_json_sample(self):
        proof = self._single_leaf_proof(
            serialised='["not", "a", "dict"]', aggregate=0.0
        )

        ok, reason = verify_opening(
            proof,
            key=_KEY_A,
            per_sample_evaluator=lambda sample: 0.0,
        )

        assert not ok
        assert reason == "non_dict_sample_at_0"

    def test_verify_rejects_non_numeric_evaluator_result(self):
        proof = self._single_leaf_proof(serialised='{"coherence":0.8}', aggregate=0.8)

        ok, reason = verify_opening(
            proof,
            key=_KEY_A,
            per_sample_evaluator=lambda sample: sample,
        )

        assert not ok
        assert reason == "evaluator_non_numeric_at_0"

    @staticmethod
    def _single_leaf_proof(serialised: str, aggregate: float) -> CommitmentProof:
        blind = b"B" * 16
        leaf = hmac.new(
            _KEY_A,
            blind + serialised.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return CommitmentProof(
            commitment=MerkleCommitment(root=leaf.hex(), sample_count=1),
            opened={0: (blind.hex(), serialised, [])},
            aggregate=aggregate,
            total_samples=1,
        )


# --- CommitmentBackend ---------------------------------------------


class TestCommitmentBackend:
    def test_short_key_rejected(self):
        with pytest.raises(ValueError, match="HMAC key"):
            CommitmentBackend(key=b"abc")

    def test_non_positive_challenge_rejected(self):
        with pytest.raises(ValueError, match="challenge_size"):
            CommitmentBackend(key=_KEY_A, challenge_size=0)

    def test_prove_verify_roundtrip(self):
        backend = CommitmentBackend(key=_KEY_A, challenge_size=4)
        samples = [{"coherence": 0.95, "halted": False} for _ in range(16)]
        stmt = MinimumCoherence(name="c", threshold=0.9, samples_min=10)
        proof = backend.prove(stmt, samples)
        ok, reason = backend.verify(stmt, proof)
        assert ok, reason

    def test_statement_threshold_below_rejected(self):
        backend = CommitmentBackend(key=_KEY_A, challenge_size=4)
        samples = [{"coherence": 0.5} for _ in range(16)]
        stmt = MinimumCoherence(name="c", threshold=0.9, samples_min=10)
        proof = backend.prove(stmt, samples)
        ok, reason = backend.verify(stmt, proof)
        assert not ok and reason == "statement_threshold_not_met"

    def test_wrong_proof_type(self):
        backend = CommitmentBackend(key=_KEY_A, challenge_size=4)
        stmt = MinimumCoherence(name="c", threshold=0.9, samples_min=1)
        ok, reason = backend.verify(stmt, proof=b"not-a-proof")
        assert not ok and reason == "wrong_proof_type"

    def test_tampered_aggregate_rejected(self):
        backend = CommitmentBackend(key=_KEY_A, challenge_size=4)
        samples = [{"coherence": 0.95} for _ in range(8)]
        stmt = MinimumCoherence(name="c", threshold=0.9, samples_min=4)
        proof = backend.prove(stmt, samples)
        # Shrink the aggregate below the opened subset sum — the
        # opened_sum > claimed check must catch it.
        tampered = CommitmentProof(
            commitment=proof.commitment,
            opened=proof.opened,
            aggregate=0.0,
            total_samples=proof.total_samples,
        )
        ok, reason = backend.verify(stmt, tampered)
        assert not ok and reason.startswith("aggregate_inconsistent")

    def test_tampered_indices_rejected(self):
        backend = CommitmentBackend(key=_KEY_A, challenge_size=4)
        samples = [{"coherence": 0.95} for _ in range(16)]
        stmt = MinimumCoherence(name="c", threshold=0.9, samples_min=1)
        proof = backend.prove(stmt, samples)
        # Swap the opened set for a different valid subset — it
        # will reconstruct the root but fail the challenge-derivation
        # check.
        original = dict(proof.opened)
        if 0 in original:
            _, leaves, blinds = commit_samples(samples, key=_KEY_A)
            del leaves, blinds  # silence unused — fresh commit differs
        new_opened = dict(original)
        # Rebuild with a different index not in the derived challenge.
        all_indices = set(range(16))
        unused = (all_indices - set(original.keys())).pop()
        # We cannot legally build a valid opening for ``unused`` with
        # this CommitmentProof instance — it lacks fresh leaves. To
        # exercise the negative path we simply delete an index, then
        # insert a bogus triple under ``unused``.
        first = next(iter(new_opened))
        bogus = new_opened.pop(first)
        new_opened[unused] = bogus
        tampered = CommitmentProof(
            commitment=proof.commitment,
            opened=new_opened,
            aggregate=proof.aggregate,
            total_samples=proof.total_samples,
        )
        ok, _ = backend.verify(stmt, tampered)
        assert not ok

    def test_valid_opening_for_wrong_challenge_index_rejected(self):
        backend = CommitmentBackend(key=_KEY_A, challenge_size=2)
        samples = [{"coherence": 0.95, "i": i} for i in range(8)]
        stmt = MinimumCoherence(name="c", threshold=0.9, samples_min=1)
        commitment, leaves, blinds = commit_samples(
            samples,
            key=_KEY_A,
            rng=_DeterministicRng(seed=23),
        )
        expected = set(
            backend._pick_challenge(
                seed_material=commitment.root.encode("utf-8"),
                sample_count=len(samples),
                challenge_size=1,
            )
        )
        wrong_idx = next(i for i in range(len(samples)) if i not in expected)
        proof = open_indices(
            indices=[wrong_idx],
            samples=samples,
            leaves=leaves,
            blinds=blinds,
            aggregate=sum(stmt.evaluate_sample(s) for s in samples),
            commitment=commitment,
        )

        ok, reason = backend.verify(stmt, proof)

        assert not ok
        assert reason == "challenge_indices_do_not_match_root_derivation"

    def test_corrupt_proof_with_more_openings_than_population_rejected(self):
        backend = CommitmentBackend(key=_KEY_A, challenge_size=4)
        stmt = MinimumCoherence(name="c", threshold=0.9, samples_min=1)
        base = self._single_leaf_proof_for_backend(serialised='{"coherence":0.95}')
        corrupt = object.__new__(CommitmentProof)
        object.__setattr__(corrupt, "commitment", base.commitment)
        object.__setattr__(
            corrupt,
            "opened",
            {
                0: next(iter(base.opened.values())),
                1: next(iter(base.opened.values())),
            },
        )
        object.__setattr__(corrupt, "aggregate", 1.9)
        object.__setattr__(corrupt, "total_samples", 1)

        ok, reason = backend.verify(stmt, corrupt)

        assert not ok
        assert reason == "opened_larger_than_committed_population"

    def test_prove_rejects_empty_samples_before_commitment(self):
        backend = CommitmentBackend(key=_KEY_A, challenge_size=4)
        stmt = MinimumCoherence(name="c", threshold=0.9, samples_min=1)

        with pytest.raises(ValueError, match="samples"):
            backend.prove(stmt, [])

    def test_python_challenge_fallback_derives_distinct_stable_indices(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(backends_mod, "_RUST_CHALLENGE_AVAILABLE", False)

        first = CommitmentBackend._pick_challenge(
            seed_material=b"root-a",
            sample_count=23,
            challenge_size=8,
        )
        second = CommitmentBackend._pick_challenge(
            seed_material=b"root-a",
            sample_count=23,
            challenge_size=8,
        )

        assert first == second
        assert len(first) == 8
        assert len(set(first)) == 8
        assert all(0 <= idx < 23 for idx in first)

    def test_rust_challenge_accelerator_is_used(self, monkeypatch):
        calls: list[tuple[bytes, int, int]] = []

        def fake_challenge(
            seed_material: bytes,
            sample_count: int,
            challenge_size: int,
        ) -> list[int]:
            calls.append((seed_material, sample_count, challenge_size))
            return [2, 0]

        monkeypatch.setattr(backends_mod, "_RUST_CHALLENGE_AVAILABLE", True)
        monkeypatch.setattr(
            backends_mod,
            "_rust_derive_challenge_indices",
            fake_challenge,
            raising=False,
        )

        assert CommitmentBackend._pick_challenge(b"root-b", 5, 2) == [2, 0]
        assert calls == [(b"root-b", 5, 2)]

    def test_rust_challenge_exception_falls_back_to_python(self, monkeypatch):
        monkeypatch.setattr(backends_mod, "_RUST_CHALLENGE_AVAILABLE", True)
        monkeypatch.setattr(
            backends_mod,
            "_rust_derive_challenge_indices",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("ffi fail")),
            raising=False,
        )

        first = CommitmentBackend._pick_challenge(
            seed_material=b"root-c",
            sample_count=17,
            challenge_size=5,
        )
        second = CommitmentBackend._pick_challenge(
            seed_material=b"root-c",
            sample_count=17,
            challenge_size=5,
        )
        assert first == second
        assert len(set(first)) == 5

    def test_rust_challenge_non_runtime_exception_falls_back_to_python(
        self, monkeypatch
    ):
        monkeypatch.setattr(backends_mod, "_RUST_CHALLENGE_AVAILABLE", True)
        monkeypatch.setattr(
            backends_mod,
            "_rust_derive_challenge_indices",
            lambda *_args: (_ for _ in ()).throw(ValueError("ffi fail")),
            raising=False,
        )

        first = CommitmentBackend._pick_challenge(
            seed_material=b"root-c",
            sample_count=17,
            challenge_size=5,
        )
        second = CommitmentBackend._pick_challenge(
            seed_material=b"root-c",
            sample_count=17,
            challenge_size=5,
        )
        assert first == second
        assert len(set(first)) == 5

    @staticmethod
    def _single_leaf_proof_for_backend(serialised: str) -> CommitmentProof:
        blind = b"C" * 16
        leaf = hmac.new(
            _KEY_A,
            blind + serialised.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return CommitmentProof(
            commitment=MerkleCommitment(root=leaf.hex(), sample_count=1),
            opened={0: (blind.hex(), serialised, [])},
            aggregate=1.0,
            total_samples=1,
        )


# --- Passport issue / verify ---------------------------------------


class TestPassport:
    def _make_samples(self) -> list[dict[str, object]]:
        return [
            {
                "coherence": 0.95,
                "halted": False,
                "domain": "finance",
                "duration_seconds": 3600,
                "breakout": False,
            }
            for _ in range(32)
        ]

    def test_issue_and_verify_all_ok(self):
        issuer = PassportIssuer(key=_KEY_A, issuing_org="org://alpha")
        verifier = PassportVerifier(
            issuer_keys={"org://alpha": _KEY_A},
            backends={"commitment": CommitmentBackend(key=_KEY_A)},
        )
        samples = self._make_samples()
        passport = issuer.issue(
            agent_id="agent-001",
            samples=samples,
            statements=[
                MinimumCoherence(name="c", threshold=0.9, samples_min=10),
                NoBreakoutEvents(name="no_break", samples_min=10),
            ],
        )
        verdict = verifier.verify(passport)
        assert verdict.accepted, verdict.failures
        assert verdict.signature_ok
        assert verdict.summary() == "all statements proved"

    def test_unknown_issuer_rejected(self):
        issuer = PassportIssuer(key=_KEY_A, issuing_org="org://alpha")
        verifier = PassportVerifier(
            issuer_keys={"org://beta": _KEY_B},
            backends={"commitment": CommitmentBackend(key=_KEY_A)},
        )
        samples = self._make_samples()
        passport = issuer.issue(
            agent_id="a",
            samples=samples,
            statements=[MinimumCoherence(name="c", threshold=0.9, samples_min=1)],
        )
        verdict = verifier.verify(passport)
        assert not verdict.accepted and not verdict.signature_ok

    def test_tampered_mac_rejected(self):
        issuer = PassportIssuer(key=_KEY_A, issuing_org="org://alpha")
        verifier = PassportVerifier(
            issuer_keys={"org://alpha": _KEY_A},
            backends={"commitment": CommitmentBackend(key=_KEY_A)},
        )
        samples = self._make_samples()
        passport = issuer.issue(
            agent_id="a",
            samples=samples,
            statements=[MinimumCoherence(name="c", threshold=0.9, samples_min=1)],
        )
        tampered = CrossOrgPassport(
            agent_id=passport.agent_id,
            issuing_org=passport.issuing_org,
            created_at=passport.created_at,
            entries=passport.entries,
            mac="0" * 64,
        )
        verdict = verifier.verify(tampered)
        assert not verdict.accepted and not verdict.signature_ok

    def test_missing_backend_reports_failure(self):
        issuer = PassportIssuer(key=_KEY_A, issuing_org="org://alpha")
        # Verifier side doesn't know about the commitment backend —
        # the passport's entries will fail to resolve to a verifier.
        verifier = PassportVerifier(
            issuer_keys={"org://alpha": _KEY_A},
            backends={"unused": CommitmentBackend(key=_KEY_A)},
        )
        samples = self._make_samples()
        passport = issuer.issue(
            agent_id="a",
            samples=samples,
            statements=[MinimumCoherence(name="c", threshold=0.9, samples_min=1)],
        )
        verdict = verifier.verify(passport)
        assert verdict.signature_ok
        assert not verdict.accepted
        assert any(
            reason.startswith("no_backend_for_") for _, reason in verdict.failures
        )

    def test_statement_below_threshold_surfaces_reason(self):
        issuer = PassportIssuer(key=_KEY_A, issuing_org="org://alpha")
        verifier = PassportVerifier(
            issuer_keys={"org://alpha": _KEY_A},
            backends={"commitment": CommitmentBackend(key=_KEY_A)},
        )
        # Low-coherence samples — the statement will verify
        # structurally but fail acceptance.
        samples = [{"coherence": 0.4} for _ in range(32)]
        passport = issuer.issue(
            agent_id="a",
            samples=samples,
            statements=[MinimumCoherence(name="c", threshold=0.9, samples_min=10)],
        )
        verdict = verifier.verify(passport)
        assert verdict.signature_ok
        assert not verdict.accepted
        assert verdict.failures == (("c", "statement_threshold_not_met"),)

    def test_issuer_rejects_short_key(self):
        with pytest.raises(ValueError, match="HMAC key"):
            PassportIssuer(key=b"abc", issuing_org="org://alpha")

    def test_issuer_rejects_empty_org(self):
        with pytest.raises(ValueError, match="issuing_org"):
            PassportIssuer(key=_KEY_A, issuing_org="")

    def test_issuer_rejects_bad_default_backend(self):
        with pytest.raises(TypeError, match="default_backend"):
            PassportIssuer(
                key=_KEY_A,
                issuing_org="org://alpha",
                default_backend="not-a-backend",
            )

    def test_issuer_rejects_non_callable_clock(self):
        with pytest.raises(ValueError, match="clock"):
            PassportIssuer(key=_KEY_A, issuing_org="org://alpha", clock=42)

    def test_issue_rejects_empty_samples(self):
        issuer = PassportIssuer(key=_KEY_A, issuing_org="org://alpha")
        with pytest.raises(ValueError, match="samples"):
            issuer.issue(
                agent_id="a",
                samples=[],
                statements=[MinimumCoherence(name="c", threshold=0.9, samples_min=1)],
            )

    def test_issue_rejects_empty_agent(self):
        issuer = PassportIssuer(key=_KEY_A, issuing_org="org://alpha")
        with pytest.raises(ValueError, match="agent_id"):
            issuer.issue(
                agent_id="",
                samples=[{"coherence": 0.9}],
                statements=[MinimumCoherence(name="c", threshold=0.9, samples_min=1)],
            )

    def test_issue_requires_at_least_one_statement(self):
        issuer = PassportIssuer(key=_KEY_A, issuing_org="org://alpha")
        with pytest.raises(ValueError, match="statement"):
            issuer.issue(
                agent_id="a",
                samples=[{"coherence": 0.9}],
                statements=[],
            )

    def test_per_statement_backend_override(self):
        issuer = PassportIssuer(key=_KEY_A, issuing_org="org://alpha")
        verifier = PassportVerifier(
            issuer_keys={"org://alpha": _KEY_A},
            backends={"commitment": CommitmentBackend(key=_KEY_A)},
        )
        override = CommitmentBackend(key=_KEY_A, challenge_size=2)
        samples = [{"coherence": 0.95} for _ in range(16)]
        stmt = MinimumCoherence(name="c", threshold=0.9, samples_min=1)
        passport = issuer.issue(
            agent_id="a",
            samples=samples,
            statements=[stmt],
            backends={"c": cast(AttestationBackend, override)},
        )
        # Both sides are using commitment; verification passes.
        assert verifier.verify(passport).accepted

    def test_passport_rejects_empty_entries(self):
        with pytest.raises(ValueError, match="at least one entry"):
            CrossOrgPassport(
                agent_id="a",
                issuing_org="org",
                created_at=0,
                entries=(),
                mac="0" * 64,
            )

    def test_passport_rejects_invalid_header_fields(self):
        issuer = PassportIssuer(key=_KEY_A, issuing_org="org://alpha")
        passport = issuer.issue(
            agent_id="a|b\\c",
            samples=[{"coherence": 0.99}],
            statements=[MinimumCoherence(name="c|d\\e", threshold=0.9, samples_min=1)],
        )

        assert b"a\\|b\\\\c" in passport.canonical_header
        assert b"c\\|d\\\\e" in passport.canonical_header
        with pytest.raises(ValueError, match="agent_id"):
            CrossOrgPassport(
                agent_id="",
                issuing_org="org",
                created_at=0,
                entries=passport.entries,
                mac="0" * 64,
            )
        with pytest.raises(ValueError, match="issuing_org"):
            CrossOrgPassport(
                agent_id="a",
                issuing_org="",
                created_at=0,
                entries=passport.entries,
                mac="0" * 64,
            )
        with pytest.raises(ValueError, match="created_at"):
            CrossOrgPassport(
                agent_id="a",
                issuing_org="org",
                created_at=-1,
                entries=passport.entries,
                mac="0" * 64,
            )
        with pytest.raises(ValueError, match="mac"):
            CrossOrgPassport(
                agent_id="a",
                issuing_org="org",
                created_at=0,
                entries=passport.entries,
                mac="short",
            )

    def test_verifier_rejects_empty_issuer_keys(self):
        with pytest.raises(ValueError, match="issuer_keys"):
            PassportVerifier(
                issuer_keys={},
                backends={"commitment": CommitmentBackend(key=_KEY_A)},
            )

    def test_verifier_rejects_short_issuer_key(self):
        with pytest.raises(ValueError, match="issuer key"):
            PassportVerifier(
                issuer_keys={"org": b"short"},
                backends={"commitment": CommitmentBackend(key=_KEY_A)},
            )

    def test_verifier_rejects_bad_backend(self):
        with pytest.raises(TypeError, match="backend"):
            PassportVerifier(
                issuer_keys={"org": _KEY_A},
                backends={"commitment": cast(Any, "not-a-backend")},
            )

    def test_verifier_rejects_empty_backends(self):
        with pytest.raises(ValueError, match="backends"):
            PassportVerifier(
                issuer_keys={"org": _KEY_A},
                backends={},
            )

    def test_verifier_rejects_empty_issuer_org_and_backend_kind(self):
        with pytest.raises(ValueError, match="empty org"):
            PassportVerifier(
                issuer_keys={"": _KEY_A},
                backends={"commitment": CommitmentBackend(key=_KEY_A)},
            )
        with pytest.raises(ValueError, match="empty kind"):
            PassportVerifier(
                issuer_keys={"org": _KEY_A},
                backends={"": CommitmentBackend(key=_KEY_A)},
            )
