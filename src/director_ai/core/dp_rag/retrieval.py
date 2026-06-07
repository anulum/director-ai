# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Differentially Private RAG Retrieval
"""Differentially private retrieval ranking with a per-tenant privacy budget.

Retrieval similarity scores are a side channel: repeatedly ranking against a
tenant's knowledge base can leak which documents it contains. This adds calibrated
Laplace noise to the similarity scores before ranking (so the order is
differentially private in the scores) and meters every query against a per-tenant
``(ε, δ)`` budget with the shared :class:`PrivacyAccountant`. A query that would
push a tenant past its budget is refused **before** any noise is spent.

The privacy accounting is ``(ε, δ)`` with the accountant's basic/advanced
composition; full Rényi-DP (RDP) accounting is a future refinement (the shared
accountant documents RDP/zCDP as out of scope). The per-query ``ε`` is adaptive —
the caller chooses it per request and the cumulative cap is enforced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from director_ai.core.federated_privacy.accountant import (
    AccountantEntry,
    PrivacyAccountant,
)
from director_ai.core.federated_privacy.mechanisms import LaplaceMechanism

_SAFE_TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class DPBudgetExceededError(RuntimeError):
    """Raised when a query would exceed a tenant's privacy budget."""


def _validate_tenant_id(tenant_id: str) -> str:
    if tenant_id and not _SAFE_TENANT_RE.fullmatch(tenant_id):
        raise ValueError(f"invalid tenant_id: {tenant_id!r}")
    return tenant_id


@dataclass(frozen=True)
class ScoredItem:
    """A retrieval candidate and its (cleartext) similarity score."""

    item_id: str
    score: float


@dataclass(frozen=True)
class PrivateRanking:
    """The DP-ranked items plus the privacy budget consumed and remaining."""

    items: tuple[ScoredItem, ...]
    epsilon_spent: float
    epsilon_remaining: float
    tenant_id: str

    def to_dict(self) -> dict[str, object]:
        """Tenant-safe view (item ids + noised scores + budget, no raw text)."""
        return {
            "tenant_id": self.tenant_id,
            "epsilon_spent": self.epsilon_spent,
            "epsilon_remaining": self.epsilon_remaining,
            "items": [{"item_id": it.item_id, "score": it.score} for it in self.items],
        }


class DifferentiallyPrivateRetrieval:
    """Rank retrieval candidates under a per-tenant differential-privacy budget.

    Parameters
    ----------
    max_epsilon:
        Per-tenant cumulative privacy budget.
    sensitivity:
        L1 sensitivity of the similarity score (default 1.0 for cosine-like
        scores in ``[0, 1]``).
    seed:
        Optional base seed for reproducible noise in tests; production uses
        system entropy (``None``). Each query advances the seed so successive
        queries draw independent noise.
    """

    def __init__(
        self,
        max_epsilon: float,
        *,
        sensitivity: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if max_epsilon <= 0:
            raise ValueError("max_epsilon must be positive")
        if sensitivity < 0:
            raise ValueError("sensitivity must be non-negative")
        self._max_epsilon = float(max_epsilon)
        self._sensitivity = float(sensitivity)
        self._seed = seed
        self._calls = 0
        self._accountants: dict[str, PrivacyAccountant] = {}

    def _accountant(self, tenant_id: str) -> PrivacyAccountant:
        acc = self._accountants.get(tenant_id)
        if acc is None:
            acc = PrivacyAccountant(max_epsilon=self._max_epsilon)
            self._accountants[tenant_id] = acc
        return acc

    def remaining(self, tenant_id: str = "") -> float:
        """Privacy budget left for a tenant."""
        tid = _validate_tenant_id(tenant_id)
        spent = (
            self._accountants[tid].cumulative_epsilon()
            if tid in self._accountants
            else 0.0
        )
        return self._max_epsilon - spent

    def spent(self, tenant_id: str = "") -> float:
        """Privacy budget already consumed by a tenant."""
        tid = _validate_tenant_id(tenant_id)
        return (
            self._accountants[tid].cumulative_epsilon()
            if tid in self._accountants
            else 0.0
        )

    def rank(
        self,
        items: list[ScoredItem],
        *,
        tenant_id: str = "",
        epsilon: float,
        label: str = "dp_retrieval",
    ) -> PrivateRanking:
        """Return a DP-ranked copy of ``items`` and charge ``epsilon``.

        Raises :class:`DPBudgetExceededError` (before spending any noise) when the
        charge would push the tenant past ``max_epsilon``.
        """
        if not items:
            raise ValueError("items must be non-empty")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        tid = _validate_tenant_id(tenant_id)
        accountant = self._accountant(tid)
        if accountant.cumulative_epsilon() + epsilon > self._max_epsilon:
            raise DPBudgetExceededError(
                f"query epsilon {epsilon} would exceed tenant {tid!r} budget "
                f"(remaining {self.remaining(tid)})"
            )
        per_call_seed = None if self._seed is None else self._seed + self._calls
        self._calls += 1
        mechanism = LaplaceMechanism(
            epsilon=epsilon,
            sensitivity=self._sensitivity,
            seed=per_call_seed,
            allow_insecure_seed=per_call_seed is not None,
        )
        noised = [
            ScoredItem(item_id=it.item_id, score=mechanism.apply(it.score))
            for it in items
        ]
        noised.sort(key=lambda it: it.score, reverse=True)
        accountant.charge(AccountantEntry(label=label, epsilon=epsilon, delta=0.0))
        return PrivateRanking(
            items=tuple(noised),
            epsilon_spent=epsilon,
            epsilon_remaining=self._max_epsilon - accountant.cumulative_epsilon(),
            tenant_id=tid,
        )
