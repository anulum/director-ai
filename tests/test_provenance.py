# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — provenance tests

"""Multi-angle coverage: CitationFact construction + hash derivation
+ tamper detection, MerkleTree determinism + odd-level duplication +
proof verification, ProvenanceChain HMAC + parent chaining + tamper
detection, SourceCredibility EMA with exponential decay + half-life
behaviour, ProvenanceVerifier end-to-end with integrity failures
and low-credibility flags."""

from __future__ import annotations

import dataclasses
import threading

import pytest

from director_ai.core.provenance import (
    CitationFact,
    FactVerificationError,
    HmacChainError,
    MerkleProof,
    MerkleTree,
    ProvenanceChain,
    ProvenanceVerifier,
    SourceCredibility,
    SourceScore,
)

# --- CitationFact --------------------------------------------------


class TestCitationFact:
    def test_hash_autoderived(self):
        fact = CitationFact(source_id="wiki", content="the sky is blue", timestamp=0.0)
        assert fact.content_hash
        assert len(fact.content_hash) == 64

    def test_hash_mismatch_rejected(self):
        with pytest.raises(FactVerificationError):
            CitationFact(
                source_id="wiki",
                content="x",
                timestamp=0.0,
                content_hash="0" * 64,
            )

    def test_source_id_collision_rejected(self):
        """Different (source_id, content) pairs produce different hashes."""
        a = CitationFact(source_id="ab", content="cd", timestamp=0.0)
        b = CitationFact(source_id="abc", content="d", timestamp=0.0)
        assert a.content_hash != b.content_hash

    def test_empty_source(self):
        with pytest.raises(ValueError, match="source_id"):
            CitationFact(source_id="", content="x", timestamp=0.0)

    def test_empty_content(self):
        with pytest.raises(ValueError, match="content"):
            CitationFact(source_id="s", content="", timestamp=0.0)

    def test_negative_timestamp(self):
        with pytest.raises(ValueError, match="timestamp"):
            CitationFact(source_id="s", content="x", timestamp=-1.0)

    def test_verify_integrity_detects_tamper(self):
        fact = CitationFact(source_id="s", content="x", timestamp=0.0)
        # Simulate at-rest tampering: mutate the frozen dataclass via
        # object.__setattr__, which is the route a disk-level
        # attacker would take (the constructor already rejects the
        # invalid pair).
        object.__setattr__(fact, "content", "y")
        with pytest.raises(FactVerificationError):
            fact.verify_integrity()


# --- MerkleTree ----------------------------------------------------


def _facts(n: int) -> list[CitationFact]:
    return [
        CitationFact(source_id=f"src{i}", content=f"content-{i}", timestamp=float(i))
        for i in range(n)
    ]


class TestMerkleTree:
    def test_root_is_deterministic(self):
        a = MerkleTree(_facts(4))
        b = MerkleTree(_facts(4))
        assert a.root == b.root

    def test_single_leaf(self):
        tree = MerkleTree(_facts(1))
        assert tree.leaf_count == 1
        assert tree.root == tree.facts()[0].content_hash

    def test_odd_level_duplicates_last(self):
        """Three leaves: last leaf duplicated at the first level."""
        tree = MerkleTree(_facts(3))
        # No exception; root is stable.
        proof = tree.proof_for(tree.facts()[2])
        assert proof.verify()

    def test_proof_verification_all_leaves(self):
        tree = MerkleTree(_facts(8))
        for fact in tree.facts():
            proof = tree.proof_for(fact)
            assert proof.verify()
            assert proof.root_hash == tree.root

    def test_tampered_proof_fails(self):
        tree = MerkleTree(_facts(4))
        proof = tree.proof_for(tree.facts()[0])
        tampered = dataclasses.replace(proof, leaf_hash="0" * 64)
        assert not tampered.verify()

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            MerkleTree([])

    def test_unknown_fact(self):
        tree = MerkleTree(_facts(2))
        outsider = CitationFact(source_id="ghost", content="x", timestamp=0.0)
        with pytest.raises(ValueError, match="not in tree"):
            tree.proof_for(outsider)

    def test_proof_is_frozen_dataclass(self):
        tree = MerkleTree(_facts(4))
        proof = tree.proof_for(tree.facts()[0])
        assert isinstance(proof, MerkleProof)


# --- ProvenanceChain ----------------------------------------------


class TestProvenanceChain:
    def test_append_and_verify(self):
        chain = ProvenanceChain(secret=b"s" * 32)
        chain.append(merkle_root="root-1")
        chain.append(merkle_root="root-2")
        ok, idx = chain.verify()
        assert ok and idx is None

    def test_short_secret(self):
        with pytest.raises(ValueError, match="secret"):
            ProvenanceChain(secret=b"short")

    def test_empty_root_rejected(self):
        chain = ProvenanceChain(secret=b"s" * 32)
        with pytest.raises(ValueError, match="merkle_root"):
            chain.append(merkle_root="")

    def test_tampered_tag_detected(self):
        chain = ProvenanceChain(secret=b"s" * 32)
        chain.append(merkle_root="root-1")
        chain._entries[0] = dataclasses.replace(chain._entries[0], tag="0" * 64)
        ok, idx = chain.verify()
        assert not ok and idx == 0

    def test_tampered_parent_hash_detected(self):
        chain = ProvenanceChain(secret=b"s" * 32)
        chain.append(merkle_root="root-1")
        chain.append(merkle_root="root-2")
        chain._entries[1] = dataclasses.replace(
            chain._entries[1], parent_hash="0" * 64
        )
        ok, idx = chain.verify()
        assert not ok and idx == 1

    def test_hmac_chain_error_class(self):
        assert issubclass(HmacChainError, ValueError)

    def test_concurrent_appends_are_atomic(self):
        chain = ProvenanceChain(secret=b"s" * 32)

        def writer(tag: str) -> None:
            for i in range(50):
                chain.append(merkle_root=f"root-{tag}-{i}")

        threads = [threading.Thread(target=writer, args=(f"t{i}",)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(chain) == 400
        ok, _ = chain.verify()
        assert ok


# --- SourceCredibility --------------------------------------------


class _FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class TestSourceCredibility:
    def test_prior_for_unseen_source(self):
        cred = SourceCredibility(prior=0.7, clock=_FakeClock())
        assert cred.score("new-source") == 0.7

    def test_observation_pulls_toward_signal(self):
        clock = _FakeClock()
        cred = SourceCredibility(prior=0.5, clock=clock)
        cred.observe("s", 1.0)
        # After a single high observation the score is (0.5 + 1.0) / 2 = 0.75.
        assert cred.score("s") == pytest.approx(0.75)

    def test_score_decays_with_age(self):
        clock = _FakeClock(start=0.0)
        cred = SourceCredibility(
            half_life_seconds=10.0, prior=0.5, clock=clock
        )
        cred.observe("s", 1.0)
        stored = cred.score("s")
        clock.now = 10.0  # one half-life later
        decayed = cred.score("s")
        # Decay pulls the score toward the prior.
        assert decayed < stored

    def test_half_life_returns_halfway_point(self):
        clock = _FakeClock(start=0.0)
        cred = SourceCredibility(
            half_life_seconds=5.0, prior=0.0, clock=clock
        )
        cred.observe("s", 1.0)
        before = cred.score("s")  # at t=0, full stored score
        clock.now = 5.0
        at_half_life = cred.score("s")
        # At one half-life, the stored weight is 0.5 so the score is
        # 0.5 * stored + 0.5 * prior — exactly halfway to the prior.
        assert at_half_life == pytest.approx((before + 0.0) / 2, abs=1e-6)

    def test_multiple_observations_accumulate_count(self):
        clock = _FakeClock()
        cred = SourceCredibility(clock=clock)
        for signal in (0.9, 0.8, 0.7):
            cred.observe("s", signal)
        snap = cred.snapshot()
        entry = next(s for s in snap if s.source_id == "s")
        assert entry.observation_count == 3

    def test_source_score_validation(self):
        with pytest.raises(ValueError, match="score"):
            SourceScore(source_id="s", score=1.5, last_updated=0.0, observation_count=0)
        with pytest.raises(ValueError, match="observation_count"):
            SourceScore(source_id="s", score=0.5, last_updated=0.0, observation_count=-1)

    def test_constructor_validation(self):
        with pytest.raises(ValueError, match="half_life_seconds"):
            SourceCredibility(half_life_seconds=0.0)
        with pytest.raises(ValueError, match="prior"):
            SourceCredibility(prior=1.5)

    def test_observe_validation(self):
        cred = SourceCredibility(clock=_FakeClock())
        with pytest.raises(ValueError, match="source_id"):
            cred.observe("", 0.5)
        with pytest.raises(ValueError, match="signal"):
            cred.observe("s", 2.0)

    def test_score_validation(self):
        cred = SourceCredibility()
        with pytest.raises(ValueError, match="source_id"):
            cred.score("")


# --- ProvenanceVerifier -------------------------------------------


class TestProvenanceVerifier:
    def _verifier(self, min_score: float = 0.2) -> ProvenanceVerifier:
        chain = ProvenanceChain(secret=b"s" * 32)
        cred = SourceCredibility(clock=_FakeClock())
        return ProvenanceVerifier(
            chain=chain,
            credibility=cred,
            min_source_score=min_score,
        )

    def test_healthy_response_all_ok(self):
        verifier = self._verifier()
        # Prime credibility so the source is above the minimum.
        for _ in range(5):
            verifier._credibility.observe("wiki", 1.0)
        facts = [
            CitationFact(source_id="wiki", content=f"c{i}", timestamp=float(i))
            for i in range(3)
        ]
        verdict = verifier.verify(facts)
        assert verdict.all_ok
        assert verdict.merkle_root
        assert verdict.trust_score > 0.5

    def test_integrity_failure_surfaced(self):
        verifier = self._verifier()
        fact = CitationFact(source_id="wiki", content="real", timestamp=0.0)
        # Same simulated at-rest tamper as the fact-level test.
        object.__setattr__(fact, "content", "fake")
        verdict = verifier.verify([fact])
        assert not verdict.all_ok
        failure = verdict.failures[0]
        assert not failure.integrity_ok

    def test_low_source_score_flagged(self):
        verifier = self._verifier(min_score=0.9)
        fact = CitationFact(source_id="unknown", content="x", timestamp=0.0)
        verdict = verifier.verify([fact])
        assert not verdict.all_ok
        assert verdict.failures[0].source_score < 0.9

    def test_chain_entry_included(self):
        verifier = self._verifier()
        fact = CitationFact(source_id="src", content="x", timestamp=0.0)
        verdict = verifier.verify([fact])
        assert verdict.chain_entry.merkle_root == verdict.merkle_root

    def test_empty_facts_rejected(self):
        verifier = self._verifier()
        with pytest.raises(ValueError, match="non-empty"):
            verifier.verify([])

    def test_bad_min_source_score(self):
        with pytest.raises(ValueError, match="min_source_score"):
            ProvenanceVerifier(
                chain=ProvenanceChain(secret=b"s" * 32),
                credibility=SourceCredibility(),
                min_source_score=1.5,
            )
