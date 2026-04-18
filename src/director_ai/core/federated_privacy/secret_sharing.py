# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — SecretShare + SecureAggregator

"""Additive ``n``-of-``n`` secret sharing over a prime modulus.

A party with a secret integer ``s`` draws ``n - 1`` random
shares uniformly over ``[0, p)`` and computes the last share as
``(s − Σ random) mod p`` so the shares sum back to ``s`` modulo
``p``. :class:`SecureAggregator` accumulates per-party share
vectors and reconstructs the multi-party total without ever
seeing any party's individual secret.

This is the classic building block behind secure aggregation
protocols (e.g. Bonawitz et al., 2017). It does not by itself
handle dropouts or a malicious aggregator — callers compose it
with DP noise and dropout recovery when those are required.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

# A 128-bit safe-prime-ish modulus large enough for real
# accumulator values without overflow; the exact bit-width is
# not security-critical for additive sharing but needs to exceed
# the expected aggregate range by several orders of magnitude.
DEFAULT_MODULUS = (1 << 127) - 1


class ShareError(ValueError):
    """Raised when a share structure is malformed (mismatched
    party count, negative shares, or aggregate inconsistency)."""


@dataclass(frozen=True)
class SecretShare:
    """One party's shares of a scalar secret.

    ``values`` holds exactly ``party_count`` entries, one share
    per intended recipient. The sender keeps one entry and
    routes the rest to the other parties over whatever transport
    the caller uses.
    """

    values: tuple[int, ...]
    modulus: int = field(default=DEFAULT_MODULUS)

    def __post_init__(self) -> None:
        if len(self.values) < 2:
            raise ShareError("SecretShare requires at least two parties")
        if self.modulus <= 0:
            raise ShareError("modulus must be positive")
        for v in self.values:
            if not 0 <= v < self.modulus:
                raise ShareError(f"share {v} is outside [0, {self.modulus})")

    @property
    def party_count(self) -> int:
        return len(self.values)


def split(
    secret: int,
    *,
    party_count: int,
    modulus: int = DEFAULT_MODULUS,
    seed: int | None = None,
) -> SecretShare:
    """Split ``secret`` into ``party_count`` additive shares
    modulo ``modulus``.

    ``seed`` makes the split reproducible (for tests); production
    code should let the system RNG fire for every call.
    """
    if party_count < 2:
        raise ShareError("party_count must be at least 2")
    if modulus <= 0:
        raise ShareError("modulus must be positive")
    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    mod_secret = secret % modulus
    shares: list[int] = []
    running = 0
    for _ in range(party_count - 1):
        draw = rng.randrange(modulus)
        shares.append(draw)
        running = (running + draw) % modulus
    last = (mod_secret - running) % modulus
    shares.append(last)
    return SecretShare(values=tuple(shares), modulus=modulus)


def reconstruct(share: SecretShare) -> int:
    """Recover the original secret modulo the share's modulus."""
    total = 0
    for v in share.values:
        total = (total + v) % share.modulus
    return total


class SecureAggregator:
    """Sum per-party share vectors componentwise.

    Every party submits a :class:`SecretShare` with the same
    ``party_count`` and ``modulus``. The aggregator sums the
    ``i``-th component across all submissions and reconstructs
    the multi-party total by summing the aggregated components —
    identical to reconstructing each share and then summing the
    secrets, but done without ever materialising any single
    party's secret.

    Parameters
    ----------
    party_count :
        Expected number of parties. Every submission must agree.
    modulus :
        Prime-ish modulus. Every submission must agree.
    """

    def __init__(
        self,
        *,
        party_count: int,
        modulus: int = DEFAULT_MODULUS,
    ) -> None:
        if party_count < 2:
            raise ShareError("party_count must be at least 2")
        if modulus <= 0:
            raise ShareError("modulus must be positive")
        self._party_count = party_count
        self._modulus = modulus
        self._accumulator: list[int] = [0] * party_count
        self._submissions = 0

    def submit(self, share: SecretShare) -> None:
        if share.party_count != self._party_count:
            raise ShareError(
                f"share has {share.party_count} parties; aggregator "
                f"expects {self._party_count}"
            )
        if share.modulus != self._modulus:
            raise ShareError(
                f"share modulus {share.modulus} != aggregator modulus {self._modulus}"
            )
        for i, value in enumerate(share.values):
            self._accumulator[i] = (self._accumulator[i] + value) % self._modulus
        self._submissions += 1

    @property
    def submissions(self) -> int:
        return self._submissions

    def reconstruct(self) -> int:
        """Return the sum of every submitted secret modulo
        ``modulus``. Raises :class:`ShareError` when no party
        has submitted."""
        if self._submissions == 0:
            raise ShareError("no submissions yet")
        total = 0
        for v in self._accumulator:
            total = (total + v) % self._modulus
        return total

    def reset(self) -> None:
        self._accumulator = [0] * self._party_count
        self._submissions = 0


def split_many(
    secrets: Sequence[int],
    *,
    party_count: int,
    modulus: int = DEFAULT_MODULUS,
    seed: int | None = None,
) -> tuple[SecretShare, ...]:
    """Convenience helper: split a whole list of secrets with
    independent seeds so the per-secret shares are uncorrelated.
    Returns one :class:`SecretShare` per secret."""
    if not secrets:
        raise ShareError("secrets must be non-empty")
    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    out: list[SecretShare] = []
    for secret in secrets:
        child_seed = rng.randrange(1 << 31)
        out.append(
            split(
                secret,
                party_count=party_count,
                modulus=modulus,
                seed=child_seed,
            )
        )
    return tuple(out)
