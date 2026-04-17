# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ScaleScorer + ValueLatticeScorer

"""Per-scale alignment scoring.

An :class:`Action` carries a mapping from value name
(``"safety"``, ``"autonomy"``, ``"transparency"``, …) to an
action-specific weight in ``[-1, 1]`` — positive for values the
action advances, negative for values it violates.
:class:`ValueVector` is the scale's per-value weight; the
scorer's output is the dot product of the action vector against
the scale vector, mapped through a logistic so the result
stays in ``[0, 1]``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

AlignmentScale = Literal["agent", "swarm", "org", "planetary"]

_VALID_SCALES: tuple[AlignmentScale, ...] = (
    "agent",
    "swarm",
    "org",
    "planetary",
)


@dataclass(frozen=True)
class ValueVector:
    """Per-value weight vector at one scale.

    ``weights`` maps a value name to a weight in ``[0, 1]``. The
    weights do not need to sum to anything specific — they are
    dotted against an action's per-value signed weights to
    produce an affinity score.
    """

    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("ValueVector.weights must be non-empty")
        for name, weight in self.weights.items():
            if not name:
                raise ValueError("every value name must be non-empty")
            if not 0.0 <= weight <= 1.0:
                raise ValueError(
                    f"value weight for {name!r} must be in [0, 1]; got {weight!r}"
                )


@dataclass(frozen=True)
class Action:
    """An action the guardrail is scoring.

    ``label`` is the caller's opaque identifier. ``impacts``
    maps a value name to a signed number in ``[-1, 1]`` —
    positive for values the action advances, negative for values
    it violates.
    """

    label: str
    impacts: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Action.label must be non-empty")
        for name, impact in self.impacts.items():
            if not name:
                raise ValueError("every impact value name must be non-empty")
            if not -1.0 <= impact <= 1.0:
                raise ValueError(
                    f"impact for {name!r} must be in [-1, 1]; got {impact!r}"
                )


@runtime_checkable
class ScaleScorer(Protocol):
    """Protocol: score an :class:`Action` at one scale."""

    scale: AlignmentScale

    def score(self, action: Action) -> float: ...


class ValueLatticeScorer:
    """Affinity-based scale scorer.

    Output = logistic( sum_v weights[v] * impacts[v] ). Values
    present on the scale but not named by the action contribute
    zero; values on the action that the scale does not track are
    ignored.

    Parameters
    ----------
    scale :
        One of the four :data:`AlignmentScale` labels.
    values :
        The scale's :class:`ValueVector`.
    steepness :
        Logistic steepness. Default 4.0 — tuned so that a
        weighted sum of +0.5 produces a score around 0.88 and
        -0.5 produces around 0.12.
    """

    def __init__(
        self,
        *,
        scale: AlignmentScale,
        values: ValueVector,
        steepness: float = 4.0,
    ) -> None:
        if scale not in _VALID_SCALES:
            raise ValueError(
                f"scale must be one of {_VALID_SCALES}; got {scale!r}"
            )
        if steepness <= 0:
            raise ValueError("steepness must be positive")
        self.scale: AlignmentScale = scale
        self._values = values
        self._steepness = steepness

    def score(self, action: Action) -> float:
        affinity = 0.0
        for value, weight in self._values.weights.items():
            impact = action.impacts.get(value, 0.0)
            affinity += weight * impact
        return _logistic(affinity * self._steepness)


def _logistic(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
