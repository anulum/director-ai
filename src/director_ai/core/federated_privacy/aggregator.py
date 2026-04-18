# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — FederatedCounter + FederatedHistogram

"""DP-noised cross-tenant aggregates.

:class:`FederatedCounter` sums per-tenant integer counts under
the Laplace mechanism. Each counter charges the accountant on
every release so the cumulative ε stays accurate across releases.

:class:`FederatedHistogram` extends the counter to a fixed
category set — one Laplace draw per category per release so the
``ε``-cost scales linearly with the number of categories under
basic composition.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .accountant import AccountantEntry, PrivacyAccountant
from .mechanisms import LaplaceMechanism


@dataclass(frozen=True)
class CounterRelease:
    """Output of one :meth:`FederatedCounter.release` call."""

    raw_sum: int
    noisy_sum: float
    epsilon_spent: float
    submissions: int


class FederatedCounter:
    """Laplace-noised cross-tenant count.

    Parameters
    ----------
    epsilon :
        Per-release ε. The underlying mechanism calibrates the
        Laplace scale from this value and the declared
        sensitivity.
    sensitivity :
        Upper bound on how much one tenant can shift the count.
        Typically ``1`` — each tenant contributes at most one
        event per release.
    accountant :
        Optional :class:`PrivacyAccountant`. When present, every
        release charges the accountant before the noisy value is
        returned; when the charge would blow the budget the
        accountant raises and the release is aborted.
    label :
        Caller-chosen tag recorded on the accountant entry. Default
        ``"counter"``.
    seed :
        Optional RNG seed for the Laplace mechanism.
    """

    def __init__(
        self,
        *,
        epsilon: float,
        sensitivity: float = 1.0,
        accountant: PrivacyAccountant | None = None,
        label: str = "counter",
        seed: int | None = None,
    ) -> None:
        if not label:
            raise ValueError("label must be non-empty")
        self._mechanism = LaplaceMechanism(
            epsilon=epsilon, sensitivity=sensitivity, seed=seed
        )
        self._accountant = accountant
        self._label = label
        self._lock = threading.Lock()
        self._contributions: dict[str, int] = {}

    def submit(self, *, tenant_id: str, count: int) -> None:
        if not tenant_id:
            raise ValueError("tenant_id must be non-empty")
        if count < 0:
            raise ValueError("count must be non-negative")
        with self._lock:
            self._contributions[tenant_id] = (
                self._contributions.get(tenant_id, 0) + count
            )

    def reset(self) -> None:
        with self._lock:
            self._contributions.clear()

    def release(self) -> CounterRelease:
        """Release the noisy sum. Charges the accountant, applies
        the Laplace mechanism, and clears the per-tenant
        contributions so the next release starts fresh."""
        with self._lock:
            raw_sum = sum(self._contributions.values())
            submissions = len(self._contributions)
            self._contributions.clear()
        if self._accountant is not None:
            self._accountant.charge(
                AccountantEntry(
                    label=self._label,
                    epsilon=self._mechanism.epsilon,
                    delta=0.0,
                    sensitivity=self._mechanism.sensitivity,
                )
            )
        noisy = self._mechanism.apply(float(raw_sum))
        return CounterRelease(
            raw_sum=raw_sum,
            noisy_sum=noisy,
            epsilon_spent=self._mechanism.epsilon,
            submissions=submissions,
        )


@dataclass(frozen=True)
class HistogramRelease:
    """Output of one :meth:`FederatedHistogram.release` call."""

    raw_counts: Mapping[str, int]
    noisy_counts: Mapping[str, float]
    epsilon_spent: float
    submissions: int
    categories: tuple[str, ...] = field(default_factory=tuple)


class FederatedHistogram:
    """Laplace-noised histogram over a fixed category set.

    Parameters
    ----------
    categories :
        Closed set of category labels. Tenant submissions
        outside this set raise :class:`KeyError`.
    epsilon :
        ε-budget charged per release. Divided uniformly across
        categories so the per-category ε is ``epsilon /
        len(categories)`` — under basic composition this gives
        the release a total of ``epsilon``.
    sensitivity :
        Per-tenant contribution bound. Default 1.
    accountant :
        Optional :class:`PrivacyAccountant`.
    label :
        Accountant tag. Default ``"histogram"``.
    seed :
        Optional RNG seed for the Laplace mechanism.
    """

    def __init__(
        self,
        *,
        categories: Iterable[str],
        epsilon: float,
        sensitivity: float = 1.0,
        accountant: PrivacyAccountant | None = None,
        label: str = "histogram",
        seed: int | None = None,
    ) -> None:
        cat_tuple = tuple(categories)
        if not cat_tuple:
            raise ValueError("categories must be non-empty")
        if len(set(cat_tuple)) != len(cat_tuple):
            raise ValueError("categories must be unique")
        for c in cat_tuple:
            if not c:
                raise ValueError("every category must be non-empty")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not label:
            raise ValueError("label must be non-empty")
        self._categories = cat_tuple
        self._total_epsilon = epsilon
        per_cat = epsilon / len(cat_tuple)
        self._mechanism = LaplaceMechanism(
            epsilon=per_cat, sensitivity=sensitivity, seed=seed
        )
        self._accountant = accountant
        self._label = label
        self._lock = threading.Lock()
        self._contributions: dict[str, int] = {c: 0 for c in cat_tuple}

    def submit(self, *, tenant_id: str, category: str, count: int = 1) -> None:
        if not tenant_id:
            raise ValueError("tenant_id must be non-empty")
        if category not in self._contributions:
            raise KeyError(f"category {category!r} not in the histogram's set")
        if count < 0:
            raise ValueError("count must be non-negative")
        with self._lock:
            self._contributions[category] += count

    def release(self) -> HistogramRelease:
        with self._lock:
            raw_counts = dict(self._contributions)
            submissions = sum(raw_counts.values())
            for c in self._categories:
                self._contributions[c] = 0
        if self._accountant is not None:
            self._accountant.charge(
                AccountantEntry(
                    label=self._label,
                    epsilon=self._total_epsilon,
                    delta=0.0,
                    sensitivity=self._mechanism.sensitivity,
                )
            )
        noisy_counts: dict[str, float] = {}
        for category in self._categories:
            noisy_counts[category] = self._mechanism.apply(float(raw_counts[category]))
        return HistogramRelease(
            raw_counts=raw_counts,
            noisy_counts=noisy_counts,
            epsilon_spent=self._total_epsilon,
            submissions=submissions,
            categories=self._categories,
        )
