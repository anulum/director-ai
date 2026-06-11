# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Differentially Private token decoding

"""Differentially private next-token selection via the exponential mechanism.

A decoder that conditions on retrieved context leaks information about that
context through the tokens it emits. Selecting the next token with the
**exponential mechanism** bounds that leakage: token ``i`` with logit ``u_i`` is
released with probability proportional to ``exp(ε · u_i / (2 Δ))``, where ``Δ``
is the L∞ sensitivity of the logits to one record of the conditioning context.
That selection is ``ε``-differentially private in the conditioning data.

This is implemented with the **Gumbel-max trick**, which is exactly equivalent
to the exponential mechanism: adding i.i.d. ``Gumbel(0, β)`` noise to each logit
and taking the argmax samples token ``i`` with probability ``∝ exp(u_i / β)``.
Setting ``β = 2 Δ / ε`` recovers the exponential-mechanism distribution, so the
argmax is ``ε``-DP (McSherry & Talwar, *Mechanism Design via Differential
Privacy*, FOCS 2007; the Gumbel-max equivalence is the report-noisy-max view).

The decoder returns the selected index and the noisy logits it used, never the
raw context. Pure ``ε``-DP, so it composes on the same ``(ε)`` budget as the
Laplace retrieval ranking and Laplace score release in the DP-RAG pipeline.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DPTokenChoice:
    """The DP-selected token index, its noisy logit, and the ε spent."""

    index: int
    noisy_logit: float
    epsilon_spent: float

    def to_dict(self) -> dict[str, float | int]:
        """Tenant-safe view (no raw logits or context)."""
        return {
            "index": self.index,
            "noisy_logit": self.noisy_logit,
            "epsilon_spent": self.epsilon_spent,
        }


class DPTokenDecoder:
    """Select a next token under ``ε``-DP via the exponential mechanism.

    Parameters
    ----------
    sensitivity:
        L∞ sensitivity ``Δ`` of the logits to one record of the conditioning
        context (default ``1.0``). Must be non-negative.
    seed:
        Optional deterministic seed for tests and simulations only. Production
        leaves this unset so the mechanism reads system entropy each call. Each
        call advances the seed so successive selections draw independent noise.
    """

    def __init__(
        self,
        *,
        sensitivity: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if sensitivity < 0.0 or not math.isfinite(sensitivity):
            raise ValueError("sensitivity must be non-negative and finite")
        self._sensitivity = float(sensitivity)
        self._seed = seed
        self._calls = 0

    @property
    def sensitivity(self) -> float:
        """The declared L∞ logit sensitivity."""
        return self._sensitivity

    def select(self, logits: Sequence[float], *, epsilon: float) -> DPTokenChoice:
        """Return the ``ε``-DP selected token index for ``logits``.

        Adds ``Gumbel(0, 2Δ/ε)`` noise to each logit and takes the argmax — the
        exponential mechanism over the token vocabulary. ``Δ = 0`` (no
        sensitivity) means the logits carry no private signal, so the noise
        scale is zero and the plain argmax is returned.
        """
        if not logits:
            raise ValueError("logits must be non-empty")
        if epsilon <= 0.0 or not math.isfinite(epsilon):
            raise ValueError("epsilon must be positive and finite")
        if any(not math.isfinite(u) for u in logits):
            raise ValueError("logits must be finite")
        rng = self._next_rng()
        beta = 0.0 if self._sensitivity == 0.0 else 2.0 * self._sensitivity / epsilon
        best_index = 0
        best_value = -math.inf
        for i, logit in enumerate(logits):
            noisy = logit + beta * _gumbel(rng)
            if noisy > best_value:
                best_value = noisy
                best_index = i
        return DPTokenChoice(
            index=best_index,
            noisy_logit=best_value,
            epsilon_spent=epsilon,
        )

    def _next_rng(self) -> random.Random:
        per_call_seed = None if self._seed is None else self._seed + self._calls
        self._calls += 1
        return _rng_from_seed(per_call_seed)


def _gumbel(rng: random.Random) -> float:
    """Draw one standard ``Gumbel(0, 1)`` sample via inverse-CDF."""
    # u ∈ (0, 1); guard the open interval so the double log is finite.
    u = rng.random()
    while u <= 0.0 or u >= 1.0:
        u = rng.random()
    return -math.log(-math.log(u))


def _rng_from_seed(seed: int | None) -> random.Random:
    # Production privacy noise reads system entropy; a seed is for deterministic
    # tests/simulations only, mirroring the federated-privacy mechanisms.
    if seed is None:
        return random.SystemRandom()
    return random.Random(seed)  # nosec B311
