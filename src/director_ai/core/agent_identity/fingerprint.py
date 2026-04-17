# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — BehavioralFingerprint + IdentityMonitor

"""Rolling behavioural statistics for identity-hijack detection.

:class:`BehavioralFingerprint` tracks per-feature mean and
variance via Welford's online algorithm — numerically stable for
long-running agents. :class:`IdentityMonitor` wraps a fingerprint,
converts a fresh observation into a z-score across each tracked
feature, and raises :class:`IdentityAnomaly` when the largest
z-score exceeds the configured threshold.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BehaviorObservation:
    """One point in behavioural feature space.

    Features are a mapping from feature name to value. Names are
    free-form strings — the fingerprint tracks whichever keys it
    has seen so operators can extend the feature set live without
    schema migration.
    """

    features: Mapping[str, float]
    source: str = ""

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("BehaviorObservation.features must be non-empty")
        for name, value in self.features.items():
            if not name:
                raise ValueError("feature names must be non-empty")
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"feature {name!r} is not finite: {value!r}"
                )


@dataclass
class _WelfordState:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.m2 / self.n if self.n > 1 else 0.0

    @property
    def stddev(self) -> float:
        return math.sqrt(self.variance)


@dataclass
class BehavioralFingerprint:
    """Rolling per-feature mean + variance.

    Parameters
    ----------
    min_samples :
        Minimum observations needed before z-scores are trusted.
        Until then :meth:`z_score` returns ``0.0`` for every
        feature. Default 16.
    """

    min_samples: int = 16
    _state: dict[str, _WelfordState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.min_samples <= 0:
            raise ValueError(f"min_samples must be positive; got {self.min_samples}")

    def update(self, observation: BehaviorObservation) -> None:
        for name, raw in observation.features.items():
            value = float(raw)
            state = self._state.setdefault(name, _WelfordState())
            state.update(value)

    def update_many(self, observations: list[BehaviorObservation]) -> None:
        for obs in observations:
            self.update(obs)

    def sample_count(self, feature: str | None = None) -> int:
        if feature is None:
            return max((s.n for s in self._state.values()), default=0)
        return self._state[feature].n if feature in self._state else 0

    def mean(self, feature: str) -> float:
        if feature not in self._state or self._state[feature].n == 0:
            raise KeyError(f"no samples for feature {feature!r}")
        return self._state[feature].mean

    def stddev(self, feature: str) -> float:
        if feature not in self._state:
            raise KeyError(f"no samples for feature {feature!r}")
        return self._state[feature].stddev

    def z_score(self, feature: str, value: float) -> float:
        """Return the z-score of ``value`` against the running
        distribution of ``feature``. ``0.0`` when the feature has
        fewer than :attr:`min_samples` samples or zero variance.
        """
        if feature not in self._state:
            return 0.0
        state = self._state[feature]
        if state.n < self.min_samples:
            return 0.0
        sigma = state.stddev
        if sigma == 0.0:
            return 0.0
        return (float(value) - state.mean) / sigma


@dataclass(frozen=True)
class IdentityAnomaly:
    """Result of an :meth:`IdentityMonitor.evaluate` call when the
    observation drifts past the z-score threshold."""

    feature: str
    z_score: float
    threshold: float
    source: str
    reason: str


@dataclass
class IdentityMonitor:
    """Z-score anomaly detector over a :class:`BehavioralFingerprint`.

    Parameters
    ----------
    fingerprint :
        Rolling statistics — typically one per (agent_id, tenant).
    z_threshold :
        Maximum absolute z-score before an observation is flagged.
        Default 3.0 — catches ~0.27 % of a Gaussian's tail.
    update_on_anomaly :
        When ``True`` (default ``False``) the monitor still folds
        the flagged observation into the fingerprint so the
        baseline drifts. Off by default — most deployments want
        the detector to remain suspicious until the anomaly is
        explicitly acknowledged.
    """

    fingerprint: BehavioralFingerprint
    z_threshold: float = 3.0
    update_on_anomaly: bool = False

    def __post_init__(self) -> None:
        if self.z_threshold <= 0:
            raise ValueError(f"z_threshold must be positive; got {self.z_threshold}")

    def evaluate(
        self, observation: BehaviorObservation
    ) -> IdentityAnomaly | None:
        worst: tuple[str, float] | None = None
        for name, raw in observation.features.items():
            z = self.fingerprint.z_score(name, float(raw))
            if abs(z) > self.z_threshold:
                if worst is None or abs(z) > abs(worst[1]):
                    worst = (name, z)
        if worst is None:
            self.fingerprint.update(observation)
            return None
        if self.update_on_anomaly:
            self.fingerprint.update(observation)
        feature, z = worst
        return IdentityAnomaly(
            feature=feature,
            z_score=z,
            threshold=self.z_threshold,
            source=observation.source,
            reason=(
                f"feature {feature!r} z={z:.3f} exceeds threshold "
                f"{self.z_threshold:.3f}"
            ),
        )
