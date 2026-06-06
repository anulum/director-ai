# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Canary detector

"""Detect tripped canaries in a model's output and retrieved evidence.

Two signals, both of which mean something has gone wrong:

* **leakage** — a canary's sentinel token appears in the model's answer. No
  legitimate answer contains it, so its presence is exfiltration, injection, or
  the model regurgitating a poisoned chunk.
* **citation** — a canary chunk turns up in the evidence the answer was grounded
  in (recognised by its :data:`CANARY_FLAG` metadata). Retrieval surfaced a
  honeytoken it never should have.

Each detection yields a :class:`CanarySignal`, increments a counter, and fires an
optional alert callback so a deployment can page or block.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..metrics import metrics
from .registry import CANARY_FLAG, CANARY_ID_KEY, CanaryRegistry

__all__ = ["CanarySignal", "CanaryDetector"]

_CANARY_SIGNALS = "canary_signals_total"

_LEAKAGE = "leakage"
_CITATION = "citation"


@dataclass(frozen=True)
class CanarySignal:
    """A single tripped canary.

    Parameters
    ----------
    canary_id:
        The canary that tripped.
    tenant_id:
        The tenant the canary belongs to.
    signal:
        ``leakage`` (token in the answer) or ``citation`` (canary chunk in the
        evidence).
    """

    canary_id: str
    tenant_id: str
    signal: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict (no canary token text)."""
        return {
            "canary_id": self.canary_id,
            "tenant_id": self.tenant_id,
            "signal": self.signal,
        }


class CanaryDetector:
    """Scan answers and evidence for tripped canaries.

    Parameters
    ----------
    registry:
        The :class:`CanaryRegistry` holding the planted canaries.
    alert:
        Optional callback invoked once per :class:`CanarySignal`.
    """

    def __init__(
        self,
        registry: CanaryRegistry,
        *,
        alert: Callable[[CanarySignal], None] | None = None,
    ) -> None:
        self._registry = registry
        self._alert = alert

    def scan_answer(self, answer: str, tenant_id: str) -> list[CanarySignal]:
        """Return leakage signals for canary tokens present in ``answer``."""
        signals = [
            self._emit(CanarySignal(fact.canary_id, fact.tenant_id, _LEAKAGE))
            for fact in self._registry.facts_for(tenant_id)
            if fact.token in answer
        ]
        return signals

    def scan_evidence(
        self,
        chunks: Sequence[Mapping[str, Any]],
        tenant_id: str,
    ) -> list[CanarySignal]:
        """Return citation signals for canary chunks in the evidence set.

        Each chunk is a mapping carrying a ``metadata`` dict; a chunk flagged
        :data:`CANARY_FLAG` for a canary registered to ``tenant_id`` trips.
        """
        registered = {fact.canary_id for fact in self._registry.facts_for(tenant_id)}
        signals: list[CanarySignal] = []
        for chunk in chunks:
            metadata = chunk.get("metadata") or {}
            if not isinstance(metadata, Mapping) or not metadata.get(CANARY_FLAG):
                continue
            canary_id = str(metadata.get(CANARY_ID_KEY, ""))
            if canary_id in registered:
                signals.append(
                    self._emit(CanarySignal(canary_id, tenant_id, _CITATION))
                )
        return signals

    def scan(
        self,
        answer: str,
        tenant_id: str,
        *,
        evidence: Sequence[Mapping[str, Any]] = (),
    ) -> list[CanarySignal]:
        """Run both the answer and evidence scans and return all signals."""
        return self.scan_answer(answer, tenant_id) + self.scan_evidence(
            evidence, tenant_id
        )

    def _emit(self, signal: CanarySignal) -> CanarySignal:
        metrics.inc_labeled(_CANARY_SIGNALS, {"signal": signal.signal})
        if self._alert is not None:
            self._alert(signal)
        return signal
