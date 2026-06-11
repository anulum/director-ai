# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Autopoietic Blueprint Tests
"""Module-specific tests for autopoietic blueprint validation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

import director_ai.core.autopoietic.blueprint as blueprint_mod
from director_ai.core.autopoietic import (
    ArchitectureMutation,
    EnsembleComponent,
    ModuleBlueprint,
)


def test_ensemble_component_rejects_negative_weight() -> None:
    child = ModuleBlueprint(kind="length", length_saturation=10)

    with pytest.raises(ValueError, match="non-negative"):
        EnsembleComponent(weight=-0.1, blueprint=child)


def test_marker_count_requires_positive_expected_marker_count() -> None:
    with pytest.raises(ValueError, match="expected_markers"):
        ModuleBlueprint(kind="marker_count", markers=("alert",), expected_markers=0)


def test_ngram_overlap_rejects_empty_reference_gram() -> None:
    with pytest.raises(ValueError, match="reference n-gram"):
        ModuleBlueprint(
            kind="ngram_overlap",
            ngram_size=2,
            reference_vocabulary=("valid gram", ""),
        )


@pytest.mark.parametrize(
    ("mutation", "blueprint", "message"),
    [
        (
            ArchitectureMutation(kind="rescale_markers", amount=1),
            ModuleBlueprint(kind="length", length_saturation=10),
            "rescale_markers",
        ),
        (
            ArchitectureMutation(kind="change_ngram", value=2),
            ModuleBlueprint(kind="marker_count", markers=("x",), expected_markers=1),
            "change_ngram",
        ),
        (
            ArchitectureMutation(kind="rebalance_ensemble", index=0, delta=0.1),
            ModuleBlueprint(kind="length", length_saturation=10),
            "rebalance_ensemble",
        ),
    ],
)
def test_mutations_reject_wrong_blueprint_kind(
    mutation: ArchitectureMutation,
    blueprint: ModuleBlueprint,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        mutation.apply(blueprint)


def test_rebalance_single_component_keeps_unit_weight() -> None:
    child = ModuleBlueprint(kind="length", length_saturation=10)
    blueprint = ModuleBlueprint(
        kind="ensemble",
        components=(EnsembleComponent(weight=1.0, blueprint=child),),
    )

    rebalanced = ArchitectureMutation(
        kind="rebalance_ensemble",
        index=0,
        delta=-0.5,
    ).apply(blueprint)

    assert rebalanced.components[0].weight == pytest.approx(1.0)
    assert rebalanced.components[0].blueprint is child


def test_rebalance_pathological_zero_total_falls_back_to_uniform_weights() -> None:
    first = ModuleBlueprint(kind="length", length_saturation=10)
    second = ModuleBlueprint(kind="marker_count", markers=("x",), expected_markers=1)
    fake_blueprint = SimpleNamespace(
        components=(
            EnsembleComponent(weight=0.0, blueprint=first),
            EnsembleComponent(weight=0.0, blueprint=second),
        )
    )

    rebalanced = blueprint_mod._rebalance(cast(Any, fake_blueprint), index=0, delta=0.0)

    assert [component.weight for component in rebalanced.components] == pytest.approx(
        [0.5, 0.5]
    )


def test_sum_float_uses_python_fallback_when_rust_flag_disabled(monkeypatch) -> None:
    monkeypatch.setattr(blueprint_mod, "_RUST_BLUEPRINT", False)

    assert blueprint_mod._sum_float([0.25, 0.5, 0.25]) == pytest.approx(1.0)
