# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — evolutionary defense genome

"""Co-evolutionary adversarial search with a hot-swap defence registry.

The registry holds the resulting defence mutations. Four pieces:

* :class:`AdversarialGenome` — a recipe expressed as a typed
  sequence of perturbation operators with bounded parameters.
  Renders into a concrete adversarial prompt via
  :meth:`AdversarialGenome.render`. Hashable + equality-comparable
  so the population de-duplicates identical genomes.
* :class:`GenomePopulation` — generation container with fitness
  tracking. Supports tournament selection, single-point
  crossover, and per-gene mutation with a seeded RNG so CI runs
  are reproducible.
* :class:`EvolutionEngine` — drives the inner and outer loops.
  Every generation selects parents by tournament, produces
  children via crossover + mutation, evaluates them against a
  caller-supplied :class:`Defense` (anything with
  ``.score(prompt) -> float``), and retains the top ``elite``
  genomes untouched. Tracks the best-fitness trajectory so
  callers can detect plateaus.
* :class:`DefenseRegistry` — thread-safe hot-swap for the active
  :class:`Defense`. Monotonic version guard, atomic swap, and
  the ability to roll back to a named snapshot when a newly
  promoted defence turns out to be worse on a holdout.
"""

from .engine import EvolutionEngine, EvolutionReport, GenomePopulation
from .genome import AdversarialGenome, Gene, GeneOperator
from .redteam_loop import (
    AutoRedteamCycleInput,
    AutoRedteamCycleReport,
    AutoRedteamDefenceLoop,
)
from .registry import Defense, DefenseRegistry, DefenseSnapshot
from .update_pipeline import DefenseUpdatePipeline, DefenseUpdateReport

__all__ = [
    "AdversarialGenome",
    "AutoRedteamCycleInput",
    "AutoRedteamCycleReport",
    "AutoRedteamDefenceLoop",
    "Defense",
    "DefenseRegistry",
    "DefenseSnapshot",
    "DefenseUpdatePipeline",
    "DefenseUpdateReport",
    "EvolutionEngine",
    "EvolutionReport",
    "Gene",
    "GeneOperator",
    "GenomePopulation",
]
