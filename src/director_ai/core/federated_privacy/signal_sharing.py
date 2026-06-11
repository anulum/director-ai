# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# Director-Class AI -- federated safety signal sharing

"""Privacy-preserving aggregate sharing for Director safety signals."""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from director_ai.core.safety_protocol import (
    DirectorSafetySignal,
    validate_director_safety_signal,
)

from .accountant import PrivacyAccountant
from .aggregator import FederatedHistogram

DEFAULT_SIGNAL_CATEGORIES: tuple[str, ...] = (
    "decision:allow",
    "decision:warn",
    "decision:halt",
    "decision:block",
    "scope:streaming",
    "scope:containment",
    "scope:attestation",
    "scope:ontology",
    "scope:trajectory",
    "scope:cyber_physical",
    "scope:inference_server",
    "scope:swarm",
    "scope:agent",
)

_CATEGORY_RE = re.compile(r"^[a-z][a-z0-9_.:-]{0,96}$")


@dataclass(frozen=True)
class FederatedSafetySignalRelease:
    """One anonymous differentially private safety-signal aggregate."""

    noisy_counts: Mapping[str, float]
    epsilon_spent: float
    categories: tuple[str, ...]
    signal_count: int
    distinct_tenants: int
    privacy_unit: str = "tenant"
    mechanism: str = "laplace"
    raw_counts: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        """Return a transport-safe release payload.

        Raw aggregate counts are omitted by default. They are useful for local
        audits and tests, but cross-organisation sharing should use the noised
        values plus the attached privacy metadata.
        """
        payload: dict[str, Any] = {
            "noisy_counts": dict(self.noisy_counts),
            "epsilon_spent": self.epsilon_spent,
            "categories": list(self.categories),
            "signal_count": self.signal_count,
            "distinct_tenants": self.distinct_tenants,
            "privacy": {
                "payload_classification": "anonymous_dp_aggregate",
                "privacy_unit": self.privacy_unit,
                "mechanism": self.mechanism,
                "raw_payload_included": False,
            },
        }
        if include_raw:
            payload["raw_counts"] = dict(self.raw_counts)
            payload["privacy"]["raw_payload_included"] = True
        return payload


class FederatedSafetySignalAggregator:
    """Aggregate tenant-safe guard signals for federated sharing.

    The aggregator accepts only validated :class:`DirectorSafetySignal`
    envelopes. Each tenant can contribute at most one count to each category
    within a release window, bounding per-category tenant sensitivity at one.
    Releases are Laplace-noised through :class:`FederatedHistogram` and blocked
    until a minimum distinct-tenant cohort is present.
    """

    def __init__(
        self,
        *,
        epsilon: float,
        categories: Sequence[str] = DEFAULT_SIGNAL_CATEGORIES,
        accountant: PrivacyAccountant | None = None,
        min_tenants: int = 2,
        seed: int | None = None,
        allow_insecure_seed: bool = False,
    ) -> None:
        if min_tenants < 1:
            raise ValueError("min_tenants must be >= 1")
        category_tuple = tuple(categories)
        _validate_categories(category_tuple)
        self._histogram = FederatedHistogram(
            categories=category_tuple,
            epsilon=epsilon,
            accountant=accountant,
            label="federated_safety_signals",
            seed=seed,
            allow_insecure_seed=allow_insecure_seed,
        )
        self._categories = category_tuple
        self._category_set = frozenset(category_tuple)
        self._min_tenants = min_tenants
        self._lock = threading.Lock()
        self._seen_pairs: set[tuple[str, str]] = set()
        self._tenants: set[str] = set()
        self._accepted_signal_ids: set[str] = set()

    def submit_transport(self, payload: Mapping[str, Any]) -> tuple[str, ...]:
        """Validate and submit a transport payload."""
        return self.submit_signal(validate_director_safety_signal(payload))

    def submit_signal(self, signal: DirectorSafetySignal) -> tuple[str, ...]:
        """Submit one tenant-safe signal and return accepted categories."""
        tenant_id = signal.event.tenant_id.strip()
        if not tenant_id:
            raise ValueError("signal event tenant_id is required")
        candidate_categories = tuple(
            category
            for category in _signal_categories(signal)
            if category in self._category_set
        )
        accepted: list[str] = []
        with self._lock:
            if signal.signal_id in self._accepted_signal_ids:
                return ()
            for category in candidate_categories:
                pair = (tenant_id, category)
                if pair in self._seen_pairs:
                    continue
                self._seen_pairs.add(pair)
                self._histogram.submit(
                    tenant_id=tenant_id,
                    category=category,
                    count=1,
                )
                accepted.append(category)
            if accepted:
                self._tenants.add(tenant_id)
                self._accepted_signal_ids.add(signal.signal_id)
        return tuple(accepted)

    def release(self) -> FederatedSafetySignalRelease:
        """Release a DP-noised aggregate and reset the contribution window."""
        with self._lock:
            distinct_tenants = len(self._tenants)
            if distinct_tenants < self._min_tenants:
                raise ValueError(
                    f"min_tenants={self._min_tenants} required before release"
                )
            signal_count = len(self._accepted_signal_ids)
            self._seen_pairs.clear()
            self._tenants.clear()
            self._accepted_signal_ids.clear()
        release = self._histogram.release()
        return FederatedSafetySignalRelease(
            noisy_counts=release.noisy_counts,
            raw_counts=release.raw_counts,
            epsilon_spent=release.epsilon_spent,
            categories=release.categories,
            signal_count=signal_count,
            distinct_tenants=distinct_tenants,
        )

    def reset(self) -> None:
        """Drop pending contributions without releasing or charging budget."""
        with self._lock:
            self._seen_pairs.clear()
            self._tenants.clear()
            self._accepted_signal_ids.clear()
        self._histogram.reset()


def _signal_categories(signal: DirectorSafetySignal) -> tuple[str, ...]:
    event = signal.event
    return (
        f"decision:{event.policy_decision}",
        f"scope:{event.hook_scope}",
    )


def _validate_categories(categories: tuple[str, ...]) -> None:
    if not categories:
        raise ValueError("categories must be non-empty")
    if len(set(categories)) != len(categories):
        raise ValueError("categories must be unique")
    for category in categories:
        if not _CATEGORY_RE.fullmatch(category):
            raise ValueError(f"invalid category {category!r}")
