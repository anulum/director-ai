# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ModuleBlueprint

"""Typed parameter-driven recipes for generated scorer modules.

The design deliberately constrains code generation to a finite
algorithm set with bounded hyperparameters — the autopoietic
engine searches inside this space rather than emitting free-form
Python. This keeps every materialised module auditable, safe,
and fast to build.

Four :data:`BlueprintKind` values ship:

* ``"length"`` — length-based heuristic scorer with a saturation
  threshold.
* ``"marker_count"`` — counts occurrences of caller-supplied
  marker phrases and normalises by expected density.
* ``"ngram_overlap"`` — Jaccard overlap between a prompt's
  token n-grams and a reference vocabulary.
* ``"ensemble"`` — weighted sum of other blueprints. Weights
  must be non-negative and sum to 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

BlueprintKind = Literal["length", "marker_count", "ngram_overlap", "ensemble"]

_VALID_KINDS: frozenset[BlueprintKind] = frozenset(
    ("length", "marker_count", "ngram_overlap", "ensemble")
)


@dataclass(frozen=True)
class EnsembleComponent:
    """One entry inside an ensemble blueprint.

    ``weight`` is a non-negative number; ensemble construction
    validates that the weights sum to 1.0 across components.
    ``blueprint`` is the child blueprint — it may itself be an
    ensemble, so the structure nests naturally.
    """

    weight: float
    blueprint: ModuleBlueprint

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError(f"weight must be non-negative; got {self.weight!r}")


@dataclass(frozen=True)
class ModuleBlueprint:
    """Typed recipe for a generated scorer module.

    Parameters
    ----------
    kind :
        One of :data:`BlueprintKind`.
    length_saturation :
        (``length`` kind only) Prompt length at which the score
        saturates at 1.0. Must be positive. Default 200.
    markers :
        (``marker_count`` kind only) Tuple of phrases to match.
        Case-insensitive substring matching. Must be non-empty.
    expected_markers :
        (``marker_count`` kind only) Expected marker density
        used to normalise the final score. Default 3.
    ngram_size :
        (``ngram_overlap`` kind only) n-gram width. Default 2.
    reference_vocabulary :
        (``ngram_overlap`` kind only) Reference n-gram tuple.
        Must be non-empty.
    components :
        (``ensemble`` kind only) Child blueprints with weights
        that sum to 1.0.
    """

    kind: BlueprintKind
    length_saturation: int = 200
    markers: tuple[str, ...] = ()
    expected_markers: int = 3
    ngram_size: int = 2
    reference_vocabulary: tuple[str, ...] = ()
    components: tuple[EnsembleComponent, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(
                f"kind must be one of {sorted(_VALID_KINDS)}; got {self.kind!r}"
            )
        if self.kind == "length":
            if self.length_saturation <= 0:
                raise ValueError(
                    "length_saturation must be positive for kind 'length'"
                )
        elif self.kind == "marker_count":
            if not self.markers:
                raise ValueError("kind 'marker_count' requires a non-empty markers tuple")
            if self.expected_markers <= 0:
                raise ValueError(
                    "expected_markers must be positive for kind 'marker_count'"
                )
            for m in self.markers:
                if not m:
                    raise ValueError("every marker must be non-empty")
        elif self.kind == "ngram_overlap":
            if self.ngram_size <= 0:
                raise ValueError("ngram_size must be positive")
            if not self.reference_vocabulary:
                raise ValueError(
                    "kind 'ngram_overlap' requires a non-empty reference_vocabulary"
                )
            for gram in self.reference_vocabulary:
                if not gram:
                    raise ValueError("every reference n-gram must be non-empty")
        elif self.kind == "ensemble":
            if not self.components:
                raise ValueError("kind 'ensemble' requires at least one component")
            total = sum(c.weight for c in self.components)
            if not 0.999 <= total <= 1.001:
                raise ValueError(f"ensemble weights must sum to 1.0; got {total}")


@dataclass(frozen=True)
class ArchitectureMutation:
    """One bounded edit to an existing :class:`ModuleBlueprint`.

    Mutations are pure functions of the blueprint + mutation
    parameters — no RNG at apply time because the caller (the
    engine) owns sampling.

    ``kind`` picks the mutation family:

    * ``"bump_length"`` — ``length_saturation += amount`` (kind
      must be ``length``).
    * ``"rescale_markers"`` — ``expected_markers += amount``
      (kind must be ``marker_count``).
    * ``"change_ngram"`` — sets ``ngram_size`` to ``value``
      (kind must be ``ngram_overlap``).
    * ``"rebalance_ensemble"`` — applies a weight delta to
      component ``index``; the remaining components are
      re-normalised uniformly so the total stays 1.0 (kind
      must be ``ensemble``).
    """

    kind: Literal[
        "bump_length", "rescale_markers", "change_ngram", "rebalance_ensemble"
    ]
    amount: int = 0
    value: int = 0
    index: int = 0
    delta: float = 0.0

    def apply(self, blueprint: ModuleBlueprint) -> ModuleBlueprint:
        if self.kind == "bump_length":
            if blueprint.kind != "length":
                raise ValueError("bump_length requires a 'length' blueprint")
            new_saturation = max(1, blueprint.length_saturation + self.amount)
            return replace(blueprint, length_saturation=new_saturation)
        if self.kind == "rescale_markers":
            if blueprint.kind != "marker_count":
                raise ValueError(
                    "rescale_markers requires a 'marker_count' blueprint"
                )
            new_expected = max(1, blueprint.expected_markers + self.amount)
            return replace(blueprint, expected_markers=new_expected)
        if self.kind == "change_ngram":
            if blueprint.kind != "ngram_overlap":
                raise ValueError(
                    "change_ngram requires a 'ngram_overlap' blueprint"
                )
            if self.value <= 0:
                raise ValueError("change_ngram requires a positive value")
            return replace(blueprint, ngram_size=self.value)
        if self.kind == "rebalance_ensemble":
            if blueprint.kind != "ensemble":
                raise ValueError(
                    "rebalance_ensemble requires an 'ensemble' blueprint"
                )
            if not 0 <= self.index < len(blueprint.components):
                raise ValueError(
                    f"index {self.index} out of range "
                    f"[0, {len(blueprint.components)})"
                )
            return _rebalance(blueprint, self.index, self.delta)
        raise ValueError(f"unknown mutation kind {self.kind!r}")  # pragma: no cover


def _rebalance(
    blueprint: ModuleBlueprint, index: int, delta: float
) -> ModuleBlueprint:
    """Apply ``delta`` to the weight at ``index`` and
    redistribute the opposite sign evenly across the remaining
    components, clamping to ``[0, 1]`` everywhere."""
    components = blueprint.components
    target_weight = max(0.0, min(1.0, components[index].weight + delta))
    applied_delta = target_weight - components[index].weight
    other_count = len(components) - 1
    new_components: list[EnsembleComponent] = []
    if other_count == 0:
        new_components.append(
            EnsembleComponent(weight=1.0, blueprint=components[index].blueprint)
        )
    else:
        redistribution = applied_delta / other_count
        for i, component in enumerate(components):
            if i == index:
                new_components.append(
                    EnsembleComponent(
                        weight=target_weight, blueprint=component.blueprint
                    )
                )
            else:
                new_weight = max(0.0, component.weight - redistribution)
                new_components.append(
                    EnsembleComponent(weight=new_weight, blueprint=component.blueprint)
                )
    # Re-normalise to absorb floating-point drift.
    total = sum(c.weight for c in new_components)
    if total <= 0:
        # Pathological rebalance — fall back to uniform weights.
        uniform = 1.0 / len(new_components)
        new_components = [
            EnsembleComponent(weight=uniform, blueprint=c.blueprint)
            for c in new_components
        ]
    else:
        new_components = [
            EnsembleComponent(weight=c.weight / total, blueprint=c.blueprint)
            for c in new_components
        ]
    return ModuleBlueprint(
        kind="ensemble",
        components=tuple(new_components),
    )
