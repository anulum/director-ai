# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — evolutionary defence genome tests

"""Multi-angle coverage: Gene / AdversarialGenome validation,
every operator's perturbation semantics, random-genome
reproducibility, GenomePopulation tournament selection and elite
preservation, EvolutionEngine convergence on a keyword-blocker
defence, constructor validation, DefenseRegistry promote /
rollback / history eviction under concurrent writers."""

from __future__ import annotations

import random
import threading
from typing import Any, cast

import pytest

from director_ai.core.defense_genome import (
    AdversarialGenome,
    Defense,
    DefenseRegistry,
    EvolutionEngine,
    EvolutionReport,
    Gene,
    GenomePopulation,
)

# --- Gene / AdversarialGenome -------------------------------------


class TestGene:
    def test_valid(self):
        g = Gene(operator="char_swap", parameter=3)
        assert g.operator == "char_swap"

    def test_unknown_operator(self):
        bad = cast(Any, "bogus")
        with pytest.raises(ValueError, match="operator"):
            Gene(operator=bad, parameter=0)

    def test_negative_parameter(self):
        with pytest.raises(ValueError, match="parameter"):
            Gene(operator="char_swap", parameter=-1)


class TestAdversarialGenome:
    def test_render_preserves_determinism(self):
        g = AdversarialGenome(genes=(Gene("char_swap", 0), Gene("casing_flip", 0)))
        a = g.render("hello world")
        b = g.render("hello world")
        assert a == b

    def test_render_empty_prompt(self):
        g = AdversarialGenome(genes=(Gene("char_swap", 0),))
        assert g.render("") == ""

    def test_char_swap_semantics(self):
        g = AdversarialGenome(genes=(Gene("char_swap", 0),))
        assert g.render("abcd") == "bacd"

    def test_char_drop_semantics(self):
        g = AdversarialGenome(genes=(Gene("char_drop", 0),))
        assert g.render("abcd") == "bcd"

    def test_casing_flip_semantics(self):
        g = AdversarialGenome(genes=(Gene("casing_flip", 0),))
        assert g.render("abcd").startswith("A")

    def test_leet_substitutes_known_character(self):
        g = AdversarialGenome(genes=(Gene("leet", 0),))
        out = g.render("apple")
        # The first leet-mapped char is 'a' -> '4'.
        assert "4" in out

    def test_leet_no_candidates(self):
        g = AdversarialGenome(genes=(Gene("leet", 0),))
        # No a/e/i/o/s in the input.
        assert g.render("xyz") == "xyz"

    def test_zero_width_inject(self):
        g = AdversarialGenome(genes=(Gene("zero_width_inject", 1),))
        assert "\u200b" in g.render("hello")

    def test_marker_prefix(self):
        g = AdversarialGenome(genes=(Gene("marker_prefix", 0),))
        assert g.render("x").lower().startswith("ignore previous")

    def test_paraphrase_scaffold(self):
        g = AdversarialGenome(genes=(Gene("paraphrase_scaffold", 0),))
        out = g.render("attack")
        assert "Rephrase" in out or "Translate" in out or "question" in out

    def test_multi_gene_composition(self):
        g = AdversarialGenome(genes=(Gene("marker_prefix", 0), Gene("casing_flip", 0)))
        out = g.render("payload")
        assert out != "payload"

    def test_empty_genes_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            AdversarialGenome(genes=())

    def test_random_reproducible(self):
        rng = random.Random(42)
        a = AdversarialGenome.random(length=5, rng=rng)
        rng = random.Random(42)
        b = AdversarialGenome.random(length=5, rng=rng)
        assert a == b

    def test_random_bad_length(self):
        rng = random.Random(0)
        with pytest.raises(ValueError, match="length"):
            AdversarialGenome.random(length=0, rng=rng)


# --- GenomePopulation + EvolutionEngine ---------------------------


class _KeywordBlocker:
    """Defence that returns a safety score shaped by how many
    blocked keywords remain in the prompt. Provides a fitness
    gradient for the GA (partial bypasses are partially
    rewarded), which is the realistic shape a learnt scorer
    produces."""

    def __init__(self, blocked: tuple[str, ...]) -> None:
        self._blocked = tuple(p.lower() for p in blocked)

    def score(self, prompt: str) -> float:
        if not self._blocked:
            return 0.95
        lowered = prompt.lower()
        hits = sum(1 for b in self._blocked if b in lowered)
        # Full hit set -> 0.05 (unsafe); zero hits -> 0.95 (safe).
        return 0.95 - 0.9 * (hits / len(self._blocked))


class _AlwaysSafe:
    def score(self, prompt: str) -> float:
        return 1.0


class TestPopulation:
    def _population(self) -> GenomePopulation:
        rng = random.Random(0)
        members = [AdversarialGenome.random(length=3, rng=rng) for _ in range(8)]
        return GenomePopulation(
            members=members,
            defense=_KeywordBlocker(blocked=("attack",)),
            seed_prompt="launch an attack",
        )

    def test_best_has_highest_fitness(self):
        pop = self._population()
        fits = pop.fitness_summary()
        assert fits[0] <= fits[1] <= fits[2]
        assert pop.best().fitness == fits[2]

    def test_sorted_elite(self):
        pop = self._population()
        elite = pop.sorted_elite(count=3)
        assert len(elite) == 3

    def test_zero_elite(self):
        pop = self._population()
        assert pop.sorted_elite(count=0) == ()

    def test_tournament_determinism(self):
        pop = self._population()
        rng_a = random.Random(7)
        rng_b = random.Random(7)
        assert pop.tournament(k=3, rng=rng_a) == pop.tournament(k=3, rng=rng_b)

    def test_tournament_bad_k(self):
        pop = self._population()
        with pytest.raises(ValueError, match="k"):
            pop.tournament(k=0, rng=random.Random(0))

    def test_replace_refits(self):
        pop = self._population()
        new_members = [AdversarialGenome(genes=(Gene("char_swap", 0),))]
        pop.replace(new_members)
        assert len(pop.members()) == 1

    def test_empty_init_rejected(self):
        with pytest.raises(ValueError, match="members"):
            GenomePopulation(members=[], defense=_AlwaysSafe(), seed_prompt="x")

    def test_empty_seed_rejected(self):
        with pytest.raises(ValueError, match="seed_prompt"):
            GenomePopulation(
                members=[AdversarialGenome(genes=(Gene("char_swap", 0),))],
                defense=_AlwaysSafe(),
                seed_prompt="",
            )

    def test_protocol_runtime_check(self):
        assert isinstance(_AlwaysSafe(), Defense)
        assert isinstance(_KeywordBlocker(()), Defense)


class TestEngine:
    def test_convergence_against_prefix_defense(self):
        """The engine must find a genome that breaks a strict
        prefix-matching defence. Any character-level perturbation
        of position 0 or a prepended marker suffices, so the GA
        converges quickly — the signal we care about is that
        fitness actually reaches the ceiling under selection."""

        class _PrefixDefense:
            def score(self, prompt: str) -> float:
                return 0.05 if prompt.startswith("launch") else 0.95

        engine = EvolutionEngine(
            population_size=16,
            genome_length=2,
            tournament_size=3,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite=2,
        )
        report = engine.run(
            defense=_PrefixDefense(),
            seed_prompt="launch an attack",
            generations=10,
            seed=7,
        )
        assert isinstance(report, EvolutionReport)
        # Prefix-break perfection: the best genome produces a
        # rendered prompt that no longer starts with "launch".
        assert report.best_fitness == pytest.approx(0.95)
        assert not report.best_rendered.startswith("launch")

    def test_trajectory_monotone_in_elitism(self):
        """With elite >= 1 the best fitness never decreases."""
        engine = EvolutionEngine(
            population_size=16, genome_length=3, elite=2, mutation_rate=0.3
        )
        report = engine.run(
            defense=_KeywordBlocker(blocked=("banned",)),
            seed_prompt="say banned thing",
            generations=8,
            seed=123,
        )
        trajectory = report.fitness_trajectory
        for earlier, later in zip(trajectory[:-1], trajectory[1:], strict=True):
            assert later >= earlier - 1e-9

    def test_seed_reproducible(self):
        engine = EvolutionEngine(population_size=8, genome_length=2)
        a = engine.run(
            defense=_KeywordBlocker(blocked=("x",)),
            seed_prompt="the x thing",
            generations=4,
            seed=9,
        )
        b = engine.run(
            defense=_KeywordBlocker(blocked=("x",)),
            seed_prompt="the x thing",
            generations=4,
            seed=9,
        )
        assert a.best_genome == b.best_genome

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"population_size": 0}, "population_size"),
            ({"genome_length": 0}, "genome_length"),
            ({"tournament_size": 0}, "tournament_size"),
            ({"crossover_rate": 1.5}, "crossover_rate"),
            ({"mutation_rate": -0.1}, "mutation_rate"),
            ({"elite": -1}, "elite"),
            ({"elite": 100, "population_size": 10}, "elite"),
        ],
    )
    def test_constructor_validation(self, kwargs: dict, match: str):
        with pytest.raises(ValueError, match=match):
            EvolutionEngine(**kwargs)

    def test_bad_generations(self):
        engine = EvolutionEngine(population_size=4, genome_length=2)
        with pytest.raises(ValueError, match="generations"):
            engine.run(
                defense=_AlwaysSafe(),
                seed_prompt="x",
                generations=0,
            )


# --- DefenseRegistry ----------------------------------------------


class TestDefenseRegistry:
    def _defence(self, score: float = 0.9):
        class _D:
            def score(self, prompt: str) -> float:
                return score

        return _D()

    def test_promote_and_active(self):
        reg = DefenseRegistry()
        snap = reg.promote(defense=self._defence(), version=1, label="init")
        assert reg.active() is snap

    def test_monotonic_version_enforced(self):
        reg = DefenseRegistry()
        reg.promote(defense=self._defence(), version=5)
        with pytest.raises(ValueError, match="version"):
            reg.promote(defense=self._defence(), version=3)

    def test_non_strict_allows_regression(self):
        reg = DefenseRegistry(strict_versioning=False)
        reg.promote(defense=self._defence(), version=5)
        reg.promote(defense=self._defence(), version=3)
        active = reg.active()
        assert active is not None and active.version == 3

    def test_history_eviction(self):
        reg = DefenseRegistry(history_size=2)
        for v in range(1, 5):
            reg.promote(defense=self._defence(), version=v, label=f"v{v}")
        history = reg.history()
        assert len(history) == 2
        assert [s.label for s in history] == ["v2", "v3"]

    def test_rollback_to_label(self):
        reg = DefenseRegistry()
        reg.promote(defense=self._defence(0.1), version=1, label="bad")
        reg.promote(defense=self._defence(0.9), version=2, label="good")
        reg.promote(defense=self._defence(0.2), version=3, label="regression")
        snap = reg.rollback_to(label="good")
        assert snap.version == 2
        assert reg.active() is snap

    def test_rollback_unknown_label(self):
        reg = DefenseRegistry()
        reg.promote(defense=self._defence(), version=1, label="init")
        with pytest.raises(KeyError, match="nope"):
            reg.rollback_to(label="nope")

    def test_rollback_empty_label(self):
        reg = DefenseRegistry()
        with pytest.raises(ValueError, match="label"):
            reg.rollback_to(label="")

    def test_bad_history_size(self):
        with pytest.raises(ValueError, match="history_size"):
            DefenseRegistry(history_size=0)

    def test_clear(self):
        reg = DefenseRegistry()
        reg.promote(defense=self._defence(), version=1)
        reg.clear()
        assert reg.active() is None
        assert reg.history() == ()

    def test_concurrent_promotes(self):
        reg = DefenseRegistry(strict_versioning=False, history_size=1_000)
        def_ = self._defence()

        def writer(base: int) -> None:
            for i in range(50):
                reg.promote(defense=def_, version=base * 100 + i, label=f"w{base}-{i}")

        threads = [threading.Thread(target=writer, args=(b,)) for b in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # 400 promotions total; one stays active, the other 399
        # live in history (capped at 1_000 so none are evicted).
        assert len(reg.history()) == 399
