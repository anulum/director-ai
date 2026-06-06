# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — content_commitment tests

"""Tests for the byte-convention Merkle content commitment.

Covers root determinism and leaf-order sensitivity, odd-level last-node
duplication, Rust/Python bit-exact agreement on the root, authentication
path, and path walk, inclusion-proof verification, tampered-leaf and
tampered-sibling rejection, and leaf/index input validation."""

from __future__ import annotations

import hashlib

import pytest

import director_ai.core.provenance.content_commitment as cc
from director_ai.core.provenance import commit_root, prove_inclusion
from director_ai.core.provenance.content_commitment import InclusionProof


def _leaves(count: int) -> list[bytes]:
    return [hashlib.sha256(f"chunk-{i}".encode()).digest() for i in range(count)]


@pytest.fixture
def python_path(monkeypatch):
    """Force the pure-Python reference path for one test."""
    monkeypatch.setattr(cc, "_RUST_MERKLE", False)


# --- commit_root ---------------------------------------------------


class TestCommitRoot:
    def test_single_leaf_root_is_leaf(self):
        leaf = _leaves(1)
        assert commit_root(leaf) == leaf[0]

    def test_root_is_32_bytes(self):
        assert len(commit_root(_leaves(5))) == 32

    def test_deterministic(self):
        assert commit_root(_leaves(6)) == commit_root(_leaves(6))

    def test_order_sensitive(self):
        forward = _leaves(4)
        reversed_leaves = list(reversed(forward))
        assert commit_root(forward) != commit_root(reversed_leaves)

    def test_odd_level_duplicates_last(self):
        # Three leaves: level 1 = [H(a,b), H(c,c)]; verifying the proof for
        # the duplicated tail leaf exercises the odd-level reduction.
        leaves = _leaves(3)
        proof = prove_inclusion(leaves, 2)
        assert proof.verify()

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            commit_root([])

    def test_non_bytes_leaf_rejected(self):
        with pytest.raises(TypeError, match="not bytes"):
            commit_root(["deadbeef"])  # type: ignore[list-item]

    def test_empty_leaf_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            commit_root([b"\x00" * 32, b""])

    def test_bytearray_leaf_accepted(self):
        leaves = _leaves(3)
        as_bytearray = [bytearray(leaf) for leaf in leaves]
        assert commit_root(as_bytearray) == commit_root(leaves)


# --- Rust / Python parity ------------------------------------------


class TestParity:
    @pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 8, 9, 17])
    def test_root_parity(self, monkeypatch, count):
        leaves = _leaves(count)
        rust_root = commit_root(leaves)
        monkeypatch.setattr(cc, "_RUST_MERKLE", False)
        python_root = commit_root(leaves)
        assert rust_root == python_root

    @pytest.mark.parametrize("count", [3, 5, 8, 9])
    def test_proof_parity(self, monkeypatch, count):
        leaves = _leaves(count)
        rust_proofs = [prove_inclusion(leaves, i) for i in range(count)]
        monkeypatch.setattr(cc, "_RUST_MERKLE", False)
        python_proofs = [prove_inclusion(leaves, i) for i in range(count)]
        for rust_proof, python_proof in zip(rust_proofs, python_proofs, strict=True):
            assert rust_proof.siblings == python_proof.siblings
            assert rust_proof.root == python_proof.root

    def test_python_proof_verifies(self, python_path):
        leaves = _leaves(9)
        for index in range(9):
            assert prove_inclusion(leaves, index).verify()


# --- prove_inclusion + InclusionProof ------------------------------


class TestInclusionProof:
    def test_every_index_verifies(self):
        leaves = _leaves(7)
        for index in range(7):
            proof = prove_inclusion(leaves, index)
            assert proof.leaf == leaves[index]
            assert proof.verify()

    def test_tampered_leaf_fails(self):
        leaves = _leaves(5)
        proof = prove_inclusion(leaves, 2)
        forged = InclusionProof(
            leaf=hashlib.sha256(b"evil").digest(),
            index=proof.index,
            siblings=proof.siblings,
            root=proof.root,
        )
        assert not forged.verify()

    def test_tampered_sibling_fails(self):
        leaves = _leaves(6)
        proof = prove_inclusion(leaves, 1)
        broken_siblings = (hashlib.sha256(b"x").digest(),) + proof.siblings[1:]
        forged = InclusionProof(
            leaf=proof.leaf,
            index=proof.index,
            siblings=broken_siblings,
            root=proof.root,
        )
        assert not forged.verify()

    def test_index_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            prove_inclusion(_leaves(3), 3)

    def test_negative_index(self):
        with pytest.raises(ValueError, match="out of range"):
            prove_inclusion(_leaves(3), -1)

    def test_proof_validation_empty_leaf(self):
        with pytest.raises(ValueError, match="leaf must be non-empty"):
            InclusionProof(leaf=b"", index=0, siblings=(), root=b"\x00" * 32)

    def test_proof_validation_bad_root_length(self):
        with pytest.raises(ValueError, match="32-byte"):
            InclusionProof(leaf=b"x", index=0, siblings=(), root=b"short")

    def test_proof_validation_negative_index(self):
        with pytest.raises(ValueError, match="non-negative"):
            InclusionProof(leaf=b"x", index=-1, siblings=(), root=b"\x00" * 32)

    def test_single_leaf_proof_has_empty_path(self):
        leaves = _leaves(1)
        proof = prove_inclusion(leaves, 0)
        assert proof.siblings == ()
        assert proof.verify()
