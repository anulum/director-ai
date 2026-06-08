# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Shamir threshold secret sharing

"""Shamir ``t``-of-``n`` threshold secret sharing over a prime field.

The additive sharing in :mod:`.secret_sharing` is ``n``-of-``n``: every party must
return its share, and any single dropout loses the secret. Shamir sharing splits a
secret across ``n`` parties so that *any* ``t`` of them reconstruct it while fewer
than ``t`` learn nothing — tolerating up to ``n - t`` dropouts and ``t - 1``
colluding parties. Because the shares are points on a polynomial, they are
additively homomorphic: summing each party's shares of several secrets and
reconstructing yields the sum, the secure-aggregation primitive for federated
scoring across confidential parties.

The field modulus is the Mersenne prime ``2**127 - 1`` (shared with the additive
module), so modular inverses for Lagrange interpolation always exist. This is the
information-theoretic secret-sharing layer; multiplicative MPC (SPDZ/MASCOT with
Beaver triples and an interactive online phase) is a separate networked protocol
and is out of scope here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .secret_sharing import DEFAULT_MODULUS, ShareError, _rng_from_seed

__all__ = ["ShamirShare", "shamir_reconstruct", "shamir_split", "shamir_sum_shares"]


@dataclass(frozen=True)
class ShamirShare:
    """One party's point ``(x, y)`` on the secret-sharing polynomial."""

    x: int
    y: int
    threshold: int
    modulus: int = field(default=DEFAULT_MODULUS)

    def __post_init__(self) -> None:
        if self.modulus <= 0:
            raise ShareError("modulus must be positive")
        if self.x <= 0:
            raise ShareError("share x must be positive (x=0 reveals the secret)")
        if not 0 <= self.y < self.modulus:
            raise ShareError(f"share y {self.y} is outside [0, {self.modulus})")
        if self.threshold < 1:
            raise ShareError("threshold must be at least 1")


def _poly_eval(coeffs: Sequence[int], x: int, modulus: int) -> int:
    """Evaluate the polynomial with ``coeffs`` (low-order first) at ``x``."""
    result = 0
    for coeff in reversed(coeffs):
        result = (result * x + coeff) % modulus
    return result


def shamir_split(
    secret: int,
    *,
    party_count: int,
    threshold: int,
    modulus: int = DEFAULT_MODULUS,
    seed: int | None = None,
    allow_insecure_seed: bool = False,
) -> tuple[ShamirShare, ...]:
    """Split ``secret`` into ``party_count`` shares, any ``threshold`` of which
    reconstruct it.

    ``seed`` makes the split reproducible only with ``allow_insecure_seed=True``;
    production code must leave it unset so the system CSPRNG fires.
    """
    if party_count < 2:
        raise ShareError("party_count must be at least 2")
    if not 1 <= threshold <= party_count:
        raise ShareError("threshold must be in [1, party_count]")
    if modulus <= 0:
        raise ShareError("modulus must be positive")
    rng = _rng_from_seed(seed, allow_insecure_seed=allow_insecure_seed)
    # Polynomial f(x) = secret + a_1 x + … + a_{t-1} x^{t-1}; f(0) is the secret.
    coeffs = [secret % modulus]
    coeffs.extend(rng.randrange(0, modulus) for _ in range(threshold - 1))
    return tuple(
        ShamirShare(
            x=x,
            y=_poly_eval(coeffs, x, modulus),
            threshold=threshold,
            modulus=modulus,
        )
        for x in range(1, party_count + 1)
    )


def shamir_reconstruct(shares: Sequence[ShamirShare]) -> int:
    """Recover the secret from at least ``threshold`` shares via Lagrange interpolation."""
    if not shares:
        raise ShareError("at least one share is required")
    modulus = shares[0].modulus
    threshold = shares[0].threshold
    if any(s.modulus != modulus for s in shares):
        raise ShareError("all shares must share a modulus")
    if any(s.threshold != threshold for s in shares):
        raise ShareError("all shares must share a threshold")
    xs = [s.x for s in shares]
    if len(set(xs)) != len(xs):
        raise ShareError("share x coordinates must be distinct")
    if len(shares) < threshold:
        raise ShareError(
            f"need at least {threshold} shares to reconstruct; got {len(shares)}"
        )
    # Lagrange interpolation evaluated at x = 0.
    secret = 0
    for j, share_j in enumerate(shares):
        numerator = 1
        denominator = 1
        for m, share_m in enumerate(shares):
            if m == j:
                continue
            numerator = (numerator * (-share_m.x)) % modulus
            denominator = (denominator * (share_j.x - share_m.x)) % modulus
        term = share_j.y * numerator * pow(denominator, -1, modulus)
        secret = (secret + term) % modulus
    return secret


def shamir_sum_shares(
    shares_per_secret: Sequence[Sequence[ShamirShare]],
) -> tuple[ShamirShare, ...]:
    """Add shares of several secrets component-wise (same ``x``) for secure summation.

    Each inner sequence is one secret's shares ordered by party. Reconstructing the
    returned shares yields the sum of the secrets without any party revealing its
    value — the federated secure-aggregation step. All secrets must have been split
    with the same party set, threshold, and modulus.
    """
    if not shares_per_secret:
        raise ShareError("at least one secret's shares are required")
    first = shares_per_secret[0]
    party_count = len(first)
    modulus = first[0].modulus
    threshold = first[0].threshold
    for group in shares_per_secret:
        if len(group) != party_count:
            raise ShareError("every secret must be shared across the same parties")
    summed: list[ShamirShare] = []
    for party in range(party_count):
        x = first[party].x
        total = 0
        for group in shares_per_secret:
            if group[party].x != x:
                raise ShareError("shares must be aligned by party (matching x)")
            total = (total + group[party].y) % modulus
        summed.append(ShamirShare(x=x, y=total, threshold=threshold, modulus=modulus))
    return tuple(summed)
