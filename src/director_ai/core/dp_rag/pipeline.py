# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Unified DP-RAG pipeline budget

"""A single per-tenant differential-privacy budget across the RAG pipeline.

:class:`~director_ai.core.dp_rag.retrieval.DifferentiallyPrivateRetrieval` meters
retrieval ranking against its own per-tenant budget. But a RAG answer leaks
through *three* stages, not one: retrieval ranking, next-token decoding, and any
released coherence score. Accounting each stage against a separate budget
under-counts the true privacy loss.

:class:`DPRagPipeline` charges all three stages against **one** per-tenant
``(ε)`` accountant, so the budget reflects the whole pipeline:

* :meth:`rank` — Laplace noise on retrieval similarity scores before ranking.
* :meth:`decode` — exponential-mechanism (Gumbel-max) next-token selection.
* :meth:`release_score` — Laplace noise on a released coherence score.

Every stage is pure ``ε``-DP, so the loss composes additively on the shared
accountant. A stage that would push a tenant past ``max_epsilon`` is refused with
:class:`~director_ai.core.dp_rag.retrieval.DPBudgetExceededError` **before** any
noise is drawn or budget charged. The per-stage charges are logged (stage label
+ ε + tenant), so a tenant can see where its budget went without any raw query,
logit, or score crossing the boundary.

For Gaussian-mechanism pipelines (DP-SGD-style logit noise rather than the
exponential mechanism), use
:class:`~director_ai.core.federated_privacy.rdp_accountant.RenyiAccountant` for
the tight RDP composition instead of this pure-``ε`` accountant.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from director_ai.core.federated_privacy.accountant import (
    AccountantEntry,
    PrivacyAccountant,
)
from director_ai.core.federated_privacy.mechanisms import LaplaceMechanism

from .decoding import DPTokenChoice, DPTokenDecoder
from .retrieval import (
    DPBudgetExceededError,
    ScoredItem,
    _validate_tenant_id,
)


@dataclass(frozen=True)
class StageCharge:
    """One recorded pipeline-stage charge against a tenant's budget."""

    stage: str
    epsilon: float
    tenant_id: str

    def to_dict(self) -> dict[str, object]:
        """Tenant-safe view of a single stage charge."""
        return {
            "stage": self.stage,
            "epsilon": self.epsilon,
            "tenant_id": self.tenant_id,
        }


@dataclass(frozen=True)
class PipelineRanking:
    """A DP-ranked candidate list plus the shared budget consumed/remaining."""

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


class DPRagPipeline:
    """Meter retrieval, decoding, and score release on one per-tenant budget.

    Parameters
    ----------
    max_epsilon:
        Per-tenant cumulative privacy budget shared across all three stages.
    retrieval_sensitivity:
        L1 sensitivity of the retrieval similarity score (default ``1.0``).
    decode_sensitivity:
        L∞ sensitivity of the decoder logits (default ``1.0``).
    score_sensitivity:
        L1 sensitivity of the released coherence score (default ``1.0``).
    seed:
        Optional base seed for reproducible noise in tests; production uses
        system entropy (``None``). Each noisy operation advances the seed so
        successive operations draw independent noise.
    """

    def __init__(
        self,
        max_epsilon: float,
        *,
        retrieval_sensitivity: float = 1.0,
        decode_sensitivity: float = 1.0,
        score_sensitivity: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if max_epsilon <= 0:
            raise ValueError("max_epsilon must be positive")
        for name, value in (
            ("retrieval_sensitivity", retrieval_sensitivity),
            ("decode_sensitivity", decode_sensitivity),
            ("score_sensitivity", score_sensitivity),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        self._max_epsilon = float(max_epsilon)
        self._retrieval_sensitivity = float(retrieval_sensitivity)
        self._decode_sensitivity = float(decode_sensitivity)
        self._score_sensitivity = float(score_sensitivity)
        self._seed = seed
        self._calls = 0
        self._accountants: dict[str, PrivacyAccountant] = {}
        self._stage_log: dict[str, list[StageCharge]] = {}

    def _accountant(self, tenant_id: str) -> PrivacyAccountant:
        acc = self._accountants.get(tenant_id)
        if acc is None:
            acc = PrivacyAccountant(max_epsilon=self._max_epsilon)
            self._accountants[tenant_id] = acc
        return acc

    def remaining(self, tenant_id: str = "") -> float:
        """Shared privacy budget left for a tenant across all stages."""
        tid = _validate_tenant_id(tenant_id)
        return self._max_epsilon - self.spent(tid)

    def spent(self, tenant_id: str = "") -> float:
        """Shared privacy budget already consumed by a tenant."""
        tid = _validate_tenant_id(tenant_id)
        acc = self._accountants.get(tid)
        return acc.cumulative_epsilon() if acc is not None else 0.0

    def stage_log(self, tenant_id: str = "") -> tuple[StageCharge, ...]:
        """Return the per-stage charges recorded for a tenant, in order."""
        tid = _validate_tenant_id(tenant_id)
        return tuple(self._stage_log.get(tid, ()))

    def _guard_budget(self, tid: str, epsilon: float, stage: str) -> PrivacyAccountant:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        accountant = self._accountant(tid)
        if accountant.cumulative_epsilon() + epsilon > self._max_epsilon:
            raise DPBudgetExceededError(
                f"{stage} epsilon {epsilon} would exceed tenant {tid!r} budget "
                f"(remaining {self.remaining(tid)})"
            )
        return accountant

    def _charge(
        self, accountant: PrivacyAccountant, tid: str, epsilon: float, stage: str
    ) -> None:
        accountant.charge(AccountantEntry(label=stage, epsilon=epsilon, delta=0.0))
        self._stage_log.setdefault(tid, []).append(
            StageCharge(stage=stage, epsilon=epsilon, tenant_id=tid)
        )

    def _next_seed(self) -> int | None:
        per_call_seed = None if self._seed is None else self._seed + self._calls
        self._calls += 1
        return per_call_seed

    def rank(
        self,
        items: list[ScoredItem],
        *,
        tenant_id: str = "",
        epsilon: float,
    ) -> PipelineRanking:
        """DP-rank ``items`` (Laplace) and charge ``epsilon`` to the shared budget."""
        if not items:
            raise ValueError("items must be non-empty")
        tid = _validate_tenant_id(tenant_id)
        accountant = self._guard_budget(tid, epsilon, "retrieve")
        per_call_seed = self._next_seed()
        mechanism = LaplaceMechanism(
            epsilon=epsilon,
            sensitivity=self._retrieval_sensitivity,
            seed=per_call_seed,
            allow_insecure_seed=per_call_seed is not None,
        )
        noised = [
            ScoredItem(item_id=it.item_id, score=mechanism.apply(it.score))
            for it in items
        ]
        noised.sort(key=lambda it: it.score, reverse=True)
        self._charge(accountant, tid, epsilon, "retrieve")
        return PipelineRanking(
            items=tuple(noised),
            epsilon_spent=epsilon,
            epsilon_remaining=self._max_epsilon - accountant.cumulative_epsilon(),
            tenant_id=tid,
        )

    def decode(
        self,
        logits: Sequence[float],
        *,
        tenant_id: str = "",
        epsilon: float,
    ) -> DPTokenChoice:
        """DP-select a next token (exponential mechanism) on the shared budget."""
        if not logits:
            raise ValueError("logits must be non-empty")
        tid = _validate_tenant_id(tenant_id)
        accountant = self._guard_budget(tid, epsilon, "decode")
        per_call_seed = self._next_seed()
        decoder = DPTokenDecoder(
            sensitivity=self._decode_sensitivity,
            seed=per_call_seed,
        )
        choice = decoder.select(logits, epsilon=epsilon)
        self._charge(accountant, tid, epsilon, "decode")
        return choice

    def release_score(
        self,
        score: float,
        *,
        tenant_id: str = "",
        epsilon: float,
    ) -> float:
        """DP-release a coherence ``score`` (Laplace) on the shared budget."""
        tid = _validate_tenant_id(tenant_id)
        accountant = self._guard_budget(tid, epsilon, "release")
        per_call_seed = self._next_seed()
        mechanism = LaplaceMechanism(
            epsilon=epsilon,
            sensitivity=self._score_sensitivity,
            seed=per_call_seed,
            allow_insecure_seed=per_call_seed is not None,
        )
        released = mechanism.apply(score)
        self._charge(accountant, tid, epsilon, "release")
        return released
