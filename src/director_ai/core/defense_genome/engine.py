# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — GenomePopulation + EvolutionEngine

"""Genetic algorithm over :class:`AdversarialGenome`.

The engine runs tournament selection, single-point crossover,
per-gene mutation with a seeded RNG, and elitism. Each generation
is evaluated against a caller-supplied :class:`Defense` — anything
with a ``.score(prompt) -> float`` that returns a ``[0, 1]``
safety probability. Higher means "defence believes this prompt
is safe"; lower means "defence wants to block it". The adversary
wins when it makes the defence return HIGH for a prompt derived
from an attack seed, so fitness is defined as
``defense.score(prompt)`` directly.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from .genome import AdversarialGenome, Gene, GeneOperator
from .registry import Defense


@dataclass(frozen=True)
class _ScoredGenome:
    genome: AdversarialGenome
    fitness: float


@dataclass(frozen=True)
class EvolutionReport:
    """Summary of one :meth:`EvolutionEngine.run` call."""

    generations: int
    best_genome: AdversarialGenome
    best_fitness: float
    best_rendered: str
    fitness_trajectory: tuple[float, ...]
    final_population: tuple[AdversarialGenome, ...]


class GenomePopulation:
    """Scored population with tournament selection.

    Parameters
    ----------
    members :
        Initial genomes; must be non-empty.
    defense :
        The defence under attack. Its ``score(prompt)`` is called
        once per unique genome per generation.
    seed_prompt :
        Prompt each genome perturbs before scoring.
    """

    def __init__(
        self,
        *,
        members: Sequence[AdversarialGenome],
        defense: Defense,
        seed_prompt: str,
    ) -> None:
        if not members:
            raise ValueError("members must be non-empty")
        if not seed_prompt:
            raise ValueError("seed_prompt must be non-empty")
        self._defense = defense
        self._seed_prompt = seed_prompt
        self._scored: list[_ScoredGenome] = [
            _ScoredGenome(
                genome=g,
                fitness=self._fitness(g),
            )
            for g in members
        ]

    def _fitness(self, genome: AdversarialGenome) -> float:
        rendered = genome.render(self._seed_prompt)
        safety = float(self._defense.score(rendered))
        return max(0.0, min(1.0, safety))

    def best(self) -> _ScoredGenome:
        return max(self._scored, key=lambda s: s.fitness)

    def fitness_summary(self) -> tuple[float, float, float]:
        """``(min, mean, max)`` fitness across the current population."""
        fits = [s.fitness for s in self._scored]
        return min(fits), sum(fits) / len(fits), max(fits)

    def tournament(self, *, k: int, rng: random.Random) -> AdversarialGenome:
        """Pick the fittest genome from a random sample of ``k``."""
        if k <= 0:
            raise ValueError(f"k must be positive; got {k}")
        k = min(k, len(self._scored))
        sample = rng.sample(self._scored, k)
        return max(sample, key=lambda s: s.fitness).genome

    def replace(self, new_members: Sequence[AdversarialGenome]) -> None:
        """Swap the population for a new generation."""
        if not new_members:
            raise ValueError("new_members must be non-empty")
        self._scored = [
            _ScoredGenome(genome=g, fitness=self._fitness(g)) for g in new_members
        ]

    def sorted_elite(self, *, count: int) -> tuple[AdversarialGenome, ...]:
        if count <= 0:
            return ()
        sorted_members = sorted(self._scored, key=lambda s: s.fitness, reverse=True)
        return tuple(s.genome for s in sorted_members[:count])

    def members(self) -> tuple[AdversarialGenome, ...]:
        return tuple(s.genome for s in self._scored)


class EvolutionEngine:
    """Tournament + crossover + mutation + elitism.

    Parameters
    ----------
    population_size :
        Members per generation. Default 32.
    genome_length :
        Genes per random genome at initialisation. Default 4.
    tournament_size :
        Sample size for tournament selection. Default 3.
    crossover_rate :
        Probability two parents produce a crossover child rather
        than a clone of parent A. Default 0.8.
    mutation_rate :
        Per-gene mutation probability. Default 0.1.
    elite :
        Top genomes carried over without mutation. Default 2.
    """

    def __init__(
        self,
        *,
        population_size: int = 32,
        genome_length: int = 4,
        tournament_size: int = 3,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elite: int = 2,
    ) -> None:
        if population_size <= 0:
            raise ValueError("population_size must be positive")
        if genome_length <= 0:
            raise ValueError("genome_length must be positive")
        if tournament_size <= 0:
            raise ValueError("tournament_size must be positive")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")
        if elite < 0:
            raise ValueError("elite must be non-negative")
        if elite >= population_size:
            raise ValueError("elite must be less than population_size")
        self._population_size = population_size
        self._genome_length = genome_length
        self._tournament_size = tournament_size
        self._crossover_rate = crossover_rate
        self._mutation_rate = mutation_rate
        self._elite = elite

    def run(
        self,
        *,
        defense: Defense,
        seed_prompt: str,
        generations: int,
        seed: int = 0,
    ) -> EvolutionReport:
        if generations <= 0:
            raise ValueError("generations must be positive")
        rng = random.Random(seed)
        initial = [
            AdversarialGenome.random(length=self._genome_length, rng=rng)
            for _ in range(self._population_size)
        ]
        population = GenomePopulation(
            members=initial, defense=defense, seed_prompt=seed_prompt
        )
        trajectory: list[float] = [population.best().fitness]
        for _ in range(generations):
            next_gen = list(population.sorted_elite(count=self._elite))
            while len(next_gen) < self._population_size:
                parent_a = population.tournament(k=self._tournament_size, rng=rng)
                parent_b = population.tournament(k=self._tournament_size, rng=rng)
                if rng.random() < self._crossover_rate:
                    child = _crossover(parent_a, parent_b, rng=rng)
                else:
                    child = parent_a
                child = _mutate(child, rate=self._mutation_rate, rng=rng)
                next_gen.append(child)
            population.replace(next_gen)
            trajectory.append(population.best().fitness)
        best = population.best()
        return EvolutionReport(
            generations=generations,
            best_genome=best.genome,
            best_fitness=best.fitness,
            best_rendered=best.genome.render(seed_prompt),
            fitness_trajectory=tuple(trajectory),
            final_population=population.members(),
        )


def _crossover(
    a: AdversarialGenome,
    b: AdversarialGenome,
    *,
    rng: random.Random,
) -> AdversarialGenome:
    """Single-point crossover with length matching.

    When parents have different lengths, the shorter one's length
    bounds the splice index so the child stays inside the operator
    domain of both parents.
    """
    max_split = min(len(a.genes), len(b.genes))
    if max_split <= 1:
        return a
    split = rng.randint(1, max_split - 1)
    return AdversarialGenome(genes=a.genes[:split] + b.genes[split:])


_OPERATORS: tuple[GeneOperator, ...] = (
    "char_swap",
    "char_drop",
    "casing_flip",
    "leet",
    "zero_width_inject",
    "marker_prefix",
    "paraphrase_scaffold",
)


def _mutate(
    genome: AdversarialGenome,
    *,
    rate: float,
    rng: random.Random,
) -> AdversarialGenome:
    """Per-gene mutation.

    With probability ``rate`` per gene, replace the gene with a
    freshly drawn one. Low ``rate`` keeps promising lineages
    intact; high ``rate`` explores aggressively.
    """
    mutated = []
    for gene in genome.genes:
        if rng.random() < rate:
            mutated.append(
                Gene(
                    operator=rng.choice(_OPERATORS),
                    parameter=rng.randrange(256),
                )
            )
        else:
            mutated.append(gene)
    return AdversarialGenome(genes=tuple(mutated))
