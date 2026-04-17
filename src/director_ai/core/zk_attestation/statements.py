# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — typed attestation claims

"""Typed claims that compose a cross-org passport.

Each claim is a frozen dataclass carrying (a) the parameters of
the statement (threshold, window, etc.) and (b) a cheap
``evaluate_sample`` method that a prover uses when computing the
witness. The verifier never calls ``evaluate_sample`` — it simply
checks that the prover's reported aggregate is consistent with the
sample opening returned by the backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# Canonical shape of a historical sample the prover feeds to a
# statement. Exposed as ``Mapping`` (covariant) so callers whose
# instrumentation returns ``dict[str, float]`` (common) pass
# without needing a type-widening cast at every call site.
HistorySample = Mapping[str, object]


@runtime_checkable
class AttestationStatement(Protocol):
    """A claim that can be evaluated on a single historical sample.

    Every concrete statement is a frozen dataclass with a short
    identifier returned by :attr:`kind` and threshold parameters.
    The :meth:`evaluate_sample` method returns a single number
    (count / sum / mean contribution) that the backend aggregates.
    """

    @property
    def kind(self) -> str: ...

    @property
    def name(self) -> str: ...

    def evaluate_sample(self, sample: HistorySample) -> float: ...

    def accepts(self, aggregate: float, total_samples: int) -> bool:
        """Given the prover's aggregate and sample count, decide
        whether the claim holds."""


@dataclass(frozen=True)
class MinimumCoherence:
    """Claim: mean coherence ≥ ``threshold`` over ``samples_min``
    samples.

    ``evaluate_sample`` returns the sample's ``coherence`` field
    (float in [0, 1]) or 0 when the sample is missing the field;
    the backend averages across all samples and the verifier
    checks ``mean ≥ threshold and count ≥ samples_min``.
    """

    name: str
    threshold: float
    samples_min: int
    kind: str = field(default="minimum_coherence", init=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MinimumCoherence.name must be non-empty")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self.samples_min <= 0:
            raise ValueError("samples_min must be positive")

    def evaluate_sample(self, sample: HistorySample) -> float:
        coherence = sample.get("coherence", 0.0)
        if not isinstance(coherence, (int, float)):
            return 0.0
        return float(coherence)

    def accepts(self, aggregate: float, total_samples: int) -> bool:
        if total_samples < self.samples_min:
            return False
        mean = aggregate / total_samples
        return mean >= self.threshold


@dataclass(frozen=True)
class MaximumHaltRate:
    """Claim: halts / total ≤ ``max_rate`` over ≥ ``samples_min``
    samples. Useful for certifying an agent's continuity record
    before an operational hand-off."""

    name: str
    max_rate: float
    samples_min: int
    kind: str = field(default="maximum_halt_rate", init=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MaximumHaltRate.name must be non-empty")
        if not 0.0 <= self.max_rate <= 1.0:
            raise ValueError("max_rate must be in [0, 1]")
        if self.samples_min <= 0:
            raise ValueError("samples_min must be positive")

    def evaluate_sample(self, sample: HistorySample) -> float:
        halted = sample.get("halted", False)
        return 1.0 if bool(halted) else 0.0

    def accepts(self, aggregate: float, total_samples: int) -> bool:
        if total_samples < self.samples_min:
            return False
        rate = aggregate / total_samples
        return rate <= self.max_rate


@dataclass(frozen=True)
class DomainExperience:
    """Claim: ``hours_min`` worth of interaction time in
    ``domain``. The aggregate is the sum of the per-sample
    ``duration_seconds`` field filtered by domain match.
    """

    name: str
    domain: str
    hours_min: float
    kind: str = field(default="domain_experience", init=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("DomainExperience.name must be non-empty")
        if not self.domain:
            raise ValueError("domain must be non-empty")
        if self.hours_min <= 0:
            raise ValueError("hours_min must be positive")

    def evaluate_sample(self, sample: HistorySample) -> float:
        if sample.get("domain") != self.domain:
            return 0.0
        duration = sample.get("duration_seconds", 0)
        if not isinstance(duration, (int, float)):
            return 0.0
        return max(0.0, float(duration))

    def accepts(self, aggregate: float, total_samples: int) -> bool:
        del total_samples
        hours = aggregate / 3600.0
        return hours >= self.hours_min


@dataclass(frozen=True)
class NoBreakoutEvents:
    """Claim: zero samples tagged ``breakout=True``. Useful as a
    hard gate before granting an agent production credentials in
    a new org."""

    name: str
    samples_min: int
    kind: str = field(default="no_breakout_events", init=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("NoBreakoutEvents.name must be non-empty")
        if self.samples_min <= 0:
            raise ValueError("samples_min must be positive")

    def evaluate_sample(self, sample: HistorySample) -> float:
        return 1.0 if bool(sample.get("breakout", False)) else 0.0

    def accepts(self, aggregate: float, total_samples: int) -> bool:
        return total_samples >= self.samples_min and aggregate == 0.0
