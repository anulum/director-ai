# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — continuous fuzzing engine

"""Continuously fuzz a guard predicate to find obfuscations that bypass it.

A static adversarial suite checks a fixed list of attacks; a fuzzer mutates a seed
corpus round after round and surfaces the variant the guard misses. Given a
``predicate`` (``True`` = the guard flags this as an attack) and a corpus of
strings it *should* flag, :class:`ContinuousFuzzer` reports every mutation that
slipped through (a bypass) and every seed the guard failed on its own (a baseline
gap). The RNG is seeded, so each finding is replayable as a regression case.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .corpus import DEFAULT_ATTACK_CORPUS
from .mutators import MUTATORS, Mutator

__all__ = ["Bypass", "ContinuousFuzzer", "FuzzReport"]

GuardPredicate = Callable[[str], bool]


@dataclass(frozen=True)
class Bypass:
    """A mutation that evaded a guard which flagged its seed."""

    operator: str
    seed: str
    mutation: str

    def to_dict(self) -> dict[str, str]:
        """Serialise the replayable bypass (attack corpus, not tenant data)."""
        return {"operator": self.operator, "seed": self.seed, "mutation": self.mutation}


@dataclass(frozen=True)
class FuzzReport:
    """The outcome of a fuzzing run against one guard predicate."""

    seeds_tested: int
    mutations_run: int
    bypasses: tuple[Bypass, ...] = ()
    seed_misses: tuple[str, ...] = ()
    operators_used: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """True when no bypass and no baseline gap were found."""
        return not self.bypasses and not self.seed_misses

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report."""
        return {
            "ok": self.ok,
            "seeds_tested": self.seeds_tested,
            "mutations_run": self.mutations_run,
            "bypasses": [b.to_dict() for b in self.bypasses],
            "seed_misses": list(self.seed_misses),
            "operators_used": list(self.operators_used),
        }


class ContinuousFuzzer:
    """Mutate a seed corpus against a guard predicate and report bypasses."""

    def __init__(
        self,
        *,
        seed: int = 0,
        mutators: dict[str, Mutator] | None = None,
    ) -> None:
        self._rng = random.Random(seed)
        self._mutators = dict(mutators) if mutators is not None else dict(MUTATORS)
        if not self._mutators:
            raise ValueError("at least one mutator is required")

    def run(
        self,
        predicate: GuardPredicate,
        *,
        corpus: Sequence[str] | None = None,
        rounds_per_seed: int = 25,
    ) -> FuzzReport:
        """Fuzz ``predicate`` over ``corpus`` and collect any bypasses.

        For each seed: if the guard does not already flag it, record a
        ``seed_miss`` (a baseline gap) and skip mutating it. Otherwise apply
        ``rounds_per_seed`` random mutations; a mutation the guard fails to flag
        is a bypass.
        """
        if rounds_per_seed < 1:
            raise ValueError("rounds_per_seed must be at least 1")
        seeds = tuple(corpus) if corpus is not None else DEFAULT_ATTACK_CORPUS
        operator_names = sorted(self._mutators)

        bypasses: list[Bypass] = []
        seed_misses: list[str] = []
        used: set[str] = set()
        mutations_run = 0

        for seed in seeds:
            if not predicate(seed):
                seed_misses.append(seed)
                continue
            for _ in range(rounds_per_seed):
                name = self._rng.choice(operator_names)
                used.add(name)
                mutation = self._mutators[name](seed, self._rng)
                mutations_run += 1
                if not predicate(mutation):
                    bypasses.append(Bypass(operator=name, seed=seed, mutation=mutation))

        return FuzzReport(
            seeds_tested=len(seeds),
            mutations_run=mutations_run,
            bypasses=tuple(bypasses),
            seed_misses=tuple(seed_misses),
            operators_used=tuple(sorted(used)),
        )
