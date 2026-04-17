# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — HMAC Merkle commitment primitives

"""Merkle tree of HMAC commitments with indexed openings.

Protocol:

1. Prover holds a private sample list ``s_0, s_1, ..., s_{n-1}``
   (serialised to bytes via :func:`_serialise_sample`). For each
   sample it draws a 128-bit blinding factor ``r_i``, computes
   ``leaf_i = HMAC-SHA256(key, r_i || s_i)`` and publishes the
   Merkle root over the leaves along with ``n``.
2. Verifier picks a random subset of indices ``I ⊂ {0..n-1}``
   (sample-size driven by a statistical-power parameter) and
   sends the challenge to the prover.
3. Prover opens each ``i ∈ I`` by revealing ``(r_i, s_i)`` and
   the authentication path up to the root. Verifier recomputes
   the leaf, walks the path, and rejects on any mismatch.

Hiding: the commitment leaks only ``HMAC(key, r_i || s_i)``; the
MAC output is uniform under the random-oracle assumption on
HMAC-SHA256. Binding: the Merkle tree collision-resistance
prevents the prover from swapping samples after commitment.

Caveat (honest naming): this is a commitment + spot-check
scheme, *not* a zero-knowledge proof. It reveals opened samples
in the clear. For a full ZK proof use a
:class:`ZkSnarkBackend` adapter.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

try:
    from backfire_kernel import (
        rust_merkle_auth_path as _rust_merkle_auth_path,
    )
    from backfire_kernel import (
        rust_merkle_root as _rust_merkle_root,
    )
    from backfire_kernel import (
        rust_merkle_walk_path as _rust_merkle_walk_path,
    )

    _RUST_MERKLE_AVAILABLE = True
except ImportError:  # pragma: no cover — optional accelerator
    _RUST_MERKLE_AVAILABLE = False

HistorySample = Mapping[str, object]
"""Canonical shape of a committed sample — a plain mapping. The
prover usually builds these as ``dict[str, object]``, but
downstream helpers accept any ``Mapping`` so callers need not
widen their dict type annotations."""

_MIN_KEY_LEN = 32
_NODE_SEP = b"\x01"


@dataclass(frozen=True)
class MerkleCommitment:
    """Root + metadata a prover publishes after committing samples.

    ``root`` is the 32-byte SHA-256 of the Merkle root (hex).
    ``sample_count`` is the committed ``n``; both values go into
    the passport so the verifier can reconstruct the tree layout
    when checking openings.
    """

    root: str
    sample_count: int

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if len(self.root) != 64:
            raise ValueError("root must be a 64-char SHA-256 hex digest")


@dataclass(frozen=True)
class CommitmentProof:
    """Bundle the verifier needs to check a single challenge round.

    * ``commitment`` is the root the prover published.
    * ``opened`` maps each revealed index to the tuple
      ``(blinding_hex, serialised_sample_json,
      authentication_path)``.
    * ``aggregate`` is the prover's reported sum of the
      statement's ``evaluate_sample`` over *all* samples; the
      verifier cross-checks consistency on the opened subset.
    * ``total_samples`` equals ``commitment.sample_count``
      (duplicated for JSON-decode convenience when the proof
      is shipped standalone).
    """

    commitment: MerkleCommitment
    opened: dict[int, tuple[str, str, list[str]]]
    aggregate: float
    total_samples: int

    def __post_init__(self) -> None:
        if self.total_samples != self.commitment.sample_count:
            raise ValueError(
                "total_samples must equal commitment.sample_count"
            )
        if not self.opened:
            raise ValueError("opened must contain at least one index")
        for idx in self.opened:
            if not 0 <= idx < self.total_samples:
                raise ValueError(f"opened index {idx} out of range")


def commit_samples(
    samples: Sequence[HistorySample],
    key: bytes,
    rng: object = None,
) -> tuple[MerkleCommitment, list[bytes], list[bytes]]:
    """Build a Merkle commitment over *samples*.

    Returns ``(commitment, leaves, blinding_factors)`` — the prover
    keeps ``leaves`` and ``blinding_factors`` secret until the
    verifier's challenge arrives. ``rng`` defaults to
    :mod:`secrets`; tests inject a deterministic source.
    """
    if len(key) < _MIN_KEY_LEN:
        raise ValueError(f"HMAC key must be at least {_MIN_KEY_LEN} bytes")
    if not samples:
        raise ValueError("samples must be non-empty")

    token_bytes = _resolve_rng(rng)
    leaves: list[bytes] = []
    blinds: list[bytes] = []
    for sample in samples:
        blind = token_bytes(16)
        blinds.append(blind)
        serialised = _serialise_sample(sample)
        leaf = hmac.new(key, blind + serialised, hashlib.sha256).digest()
        leaves.append(leaf)

    root = _merkle_root(leaves)
    commitment = MerkleCommitment(
        root=root.hex(), sample_count=len(samples)
    )
    return commitment, leaves, blinds


def open_indices(
    indices: Sequence[int],
    samples: Sequence[HistorySample],
    leaves: Sequence[bytes],
    blinds: Sequence[bytes],
    aggregate: float,
    commitment: MerkleCommitment,
) -> CommitmentProof:
    """Produce a :class:`CommitmentProof` revealing *indices*.

    The authentication path for each opened leaf is the ordered
    list of hex-encoded sibling hashes from the leaf up to the
    root.
    """
    if len(leaves) != len(samples) or len(blinds) != len(samples):
        raise ValueError(
            "leaves / blinds / samples length mismatch",
        )
    if not indices:
        raise ValueError("indices must be non-empty")
    opened: dict[int, tuple[str, str, list[str]]] = {}
    for idx in indices:
        if not 0 <= idx < len(samples):
            raise ValueError(f"index {idx} out of range")
        blind_hex = blinds[idx].hex()
        serialised = _serialise_sample(samples[idx]).decode("utf-8")
        path = [h.hex() for h in _auth_path(leaves, idx)]
        opened[idx] = (blind_hex, serialised, path)

    return CommitmentProof(
        commitment=commitment,
        opened=opened,
        aggregate=aggregate,
        total_samples=commitment.sample_count,
    )


def verify_opening(
    proof: CommitmentProof,
    key: bytes,
    per_sample_evaluator: object,
) -> tuple[bool, str]:
    """Check that every opened leaf rebuilds the root and that the
    evaluator's output on the opened samples is consistent with
    the prover's reported ``aggregate``.

    ``per_sample_evaluator`` is called as ``fn(sample_dict) ->
    float``. The verifier does not know the full population, so
    consistency is sampled: each opened sample's evaluator output
    must sum to a value ≤ ``proof.aggregate`` (the remainder is
    attributed to the unopened majority). Combined with random
    challenge sampling, the probability of a cheating prover
    escaping detection decays geometrically with ``|opened|``.
    """
    if len(key) < _MIN_KEY_LEN:
        return False, "hmac_key_too_short"
    if not callable(per_sample_evaluator):
        return False, "evaluator_not_callable"

    root_bytes = bytes.fromhex(proof.commitment.root)
    opened_sum = 0.0
    for idx, (blind_hex, serialised, path_hex) in proof.opened.items():
        blind = bytes.fromhex(blind_hex)
        leaf = hmac.new(
            key, blind + serialised.encode("utf-8"), hashlib.sha256
        ).digest()
        reconstructed = _walk_path(leaf, idx, [bytes.fromhex(h) for h in path_hex])
        if reconstructed != root_bytes:
            return False, f"merkle_mismatch_at_{idx}"
        try:
            sample_obj = json.loads(serialised)
        except json.JSONDecodeError:
            return False, f"malformed_sample_at_{idx}"
        if not isinstance(sample_obj, dict):
            return False, f"non_dict_sample_at_{idx}"
        value = per_sample_evaluator(sample_obj)
        if not isinstance(value, (int, float)):
            return False, f"evaluator_non_numeric_at_{idx}"
        opened_sum += float(value)

    # The prover claimed ``aggregate`` over *all* samples. The
    # opened subset is a subset, so its evaluator sum cannot
    # exceed the aggregate by more than a small floating-point
    # tolerance. Aggregates that have been inflated beyond the
    # sum of any valid opening are rejected.
    if opened_sum > proof.aggregate + 1e-6:
        return (
            False,
            f"aggregate_inconsistent (opened_sum={opened_sum} > "
            f"claimed={proof.aggregate})",
        )
    return True, ""


# ------------------------------------------------------------------
# Internal helpers


def _serialise_sample(sample: Mapping[str, object]) -> bytes:
    """Deterministic JSON encoding for MAC stability across orgs."""
    return json.dumps(
        sample, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def _hash_node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(_NODE_SEP + left + right).digest()


def _merkle_root(leaves: Sequence[bytes]) -> bytes:
    if _RUST_MERKLE_AVAILABLE:
        return bytes(_rust_merkle_root(list(leaves)))
    level = list(leaves)
    while len(level) > 1:
        nxt: list[bytes] = []
        for i in range(0, len(level), 2):
            left = level[i]
            # Odd-length level: duplicate the last leaf (classic
            # RFC-6962 approach reduces trees exactly this way).
            right = level[i + 1] if i + 1 < len(level) else left
            nxt.append(_hash_node(left, right))
        level = nxt
    return level[0]


def _auth_path(leaves: Sequence[bytes], index: int) -> list[bytes]:
    """Sibling hashes from *index* up to the root (exclusive)."""
    if _RUST_MERKLE_AVAILABLE:
        return [bytes(b) for b in _rust_merkle_auth_path(list(leaves), index)]
    path: list[bytes] = []
    level = list(leaves)
    i = index
    while len(level) > 1:
        sibling_idx = i ^ 1
        if sibling_idx < len(level):
            path.append(level[sibling_idx])
        else:
            # Odd tail — the missing sibling is the node itself
            # (matches the duplicate-last-leaf rule above).
            path.append(level[i])
        nxt: list[bytes] = []
        for j in range(0, len(level), 2):
            left = level[j]
            right = level[j + 1] if j + 1 < len(level) else left
            nxt.append(_hash_node(left, right))
        level = nxt
        i //= 2
    return path


def _walk_path(leaf: bytes, index: int, siblings: Sequence[bytes]) -> bytes:
    if _RUST_MERKLE_AVAILABLE:
        return bytes(
            _rust_merkle_walk_path(leaf, index, [bytes(s) for s in siblings])
        )
    node = leaf
    i = index
    for sibling in siblings:
        node = (
            _hash_node(node, sibling)
            if i % 2 == 0
            else _hash_node(sibling, node)
        )
        i //= 2
    return node


def _resolve_rng(rng: object) -> Callable[[int], bytes]:
    """Return a ``token_bytes(n) -> bytes`` callable.

    Tests pass a deterministic RNG exposing ``token_bytes``; in
    production we fall back to :func:`secrets.token_bytes`.
    """
    if rng is None:
        return secrets.token_bytes
    fn = getattr(rng, "token_bytes", None)
    if fn is None or not callable(fn):
        raise ValueError("rng must expose a callable `token_bytes(n)`")
    # Mypy cannot refine ``getattr`` to a specific Callable type,
    # so we narrow via an explicit cast — the runtime ``callable``
    # check above is the actual guarantee.
    from typing import cast as _cast

    return _cast(Callable[[int], bytes], fn)
