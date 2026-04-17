# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — autopoietic architecture evolution

"""Meta-layer that auto-generates candidate scorer modules,
evaluates them against a reference corpus under bounded
execution, and hot-swaps the winners into service.

The package centres on typed, parameter-driven module recipes
rather than free-form code generation. A :class:`ModuleBlueprint`
names an algorithm family (length heuristic, n-gram overlap,
marker-count combiner, weighted ensemble of the above) plus the
bounded hyperparameters. :class:`ModuleBuilder` materialises
the blueprint into a callable ``scorer(prompt) -> float`` and
:class:`BoundedSandbox` runs the scorer with a wall-clock timeout
so a pathological generated module cannot stall the engine.

:class:`ArchitectureMutation` expresses an edit on an existing
blueprint: bump a weight, swap a kernel, extend the ensemble.
Mutations are closed under composition so an engine can apply a
trajectory of edits to a seed blueprint.

:class:`ModuleTestSuite` scores a materialised module on a
corpus of :class:`ScoredSample` pairs (prompt + ground truth).
Two metrics: mean absolute error and rank correlation against
the ground truth — both reported so the engine can select on
whichever the operator cares about.

:class:`AutopoieticEngine` drives the generation → evaluation →
hot-swap loop. It maintains a :class:`ModuleRegistry` with
monotonic versioning and a promotion guard that only accepts
candidates strictly better than the incumbent by a configurable
margin.
"""

from .blueprint import (
    ArchitectureMutation,
    BlueprintKind,
    EnsembleComponent,
    ModuleBlueprint,
)
from .builder import BoundedSandbox, BuildError, ModuleBuilder, SandboxTimeout
from .engine import AutopoieticEngine, EvolutionCycle, ModuleRegistry
from .testsuite import ModuleTestSuite, ScoredSample, SuiteResult

__all__ = [
    "ArchitectureMutation",
    "AutopoieticEngine",
    "BlueprintKind",
    "BoundedSandbox",
    "BuildError",
    "EnsembleComponent",
    "EvolutionCycle",
    "ModuleBlueprint",
    "ModuleBuilder",
    "ModuleRegistry",
    "ModuleTestSuite",
    "SandboxTimeout",
    "ScoredSample",
    "SuiteResult",
]
