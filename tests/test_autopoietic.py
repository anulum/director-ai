# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — autopoietic architecture tests

"""Multi-angle coverage: blueprint validation per family,
mutation composition, ModuleBuilder compiles each family
correctly, BoundedSandbox timeout + wrap behaviour,
ModuleTestSuite metrics (MAE + Spearman with ties), registry
versioning + rollback, engine cycle promoting only on
improvement."""

from __future__ import annotations

import time
from typing import Any, cast

import pytest

from director_ai.core.autopoietic import (
    ArchitectureMutation,
    AutopoieticEngine,
    BoundedSandbox,
    BuildError,
    EnsembleComponent,
    EvolutionCycle,
    ModuleBlueprint,
    ModuleBuilder,
    ModuleRegistry,
    ModuleTestSuite,
    SandboxTimeout,
    ScoredSample,
    SuiteResult,
)
from director_ai.core.autopoietic.testsuite import _spearman

# --- ModuleBlueprint -----------------------------------------------


class TestBlueprint:
    def test_length_valid(self):
        bp = ModuleBlueprint(kind="length", length_saturation=100)
        assert bp.length_saturation == 100

    def test_length_needs_positive_saturation(self):
        with pytest.raises(ValueError, match="length_saturation"):
            ModuleBlueprint(kind="length", length_saturation=0)

    def test_marker_count_valid(self):
        bp = ModuleBlueprint(
            kind="marker_count", markers=("SYSTEM:", "IGNORE"), expected_markers=2
        )
        assert bp.markers == ("SYSTEM:", "IGNORE")

    def test_marker_count_needs_markers(self):
        with pytest.raises(ValueError, match="markers"):
            ModuleBlueprint(kind="marker_count")

    def test_marker_count_rejects_empty_marker(self):
        with pytest.raises(ValueError, match="every marker"):
            ModuleBlueprint(kind="marker_count", markers=("",))

    def test_ngram_overlap_valid(self):
        bp = ModuleBlueprint(
            kind="ngram_overlap",
            ngram_size=2,
            reference_vocabulary=("the fox", "jumps over"),
        )
        assert bp.ngram_size == 2

    def test_ngram_needs_vocabulary(self):
        with pytest.raises(ValueError, match="reference_vocabulary"):
            ModuleBlueprint(kind="ngram_overlap")

    def test_ngram_needs_positive_size(self):
        with pytest.raises(ValueError, match="ngram_size"):
            ModuleBlueprint(
                kind="ngram_overlap",
                ngram_size=0,
                reference_vocabulary=("x",),
            )

    def test_ensemble_valid(self):
        length = ModuleBlueprint(kind="length", length_saturation=100)
        markers = ModuleBlueprint(
            kind="marker_count", markers=("SYSTEM:",), expected_markers=1
        )
        bp = ModuleBlueprint(
            kind="ensemble",
            components=(
                EnsembleComponent(weight=0.6, blueprint=length),
                EnsembleComponent(weight=0.4, blueprint=markers),
            ),
        )
        assert len(bp.components) == 2

    def test_ensemble_weights_must_sum_to_one(self):
        length = ModuleBlueprint(kind="length", length_saturation=100)
        with pytest.raises(ValueError, match="sum to 1"):
            ModuleBlueprint(
                kind="ensemble",
                components=(EnsembleComponent(weight=0.3, blueprint=length),),
            )

    def test_ensemble_requires_component(self):
        with pytest.raises(ValueError, match="at least one component"):
            ModuleBlueprint(kind="ensemble")

    def test_unknown_kind_rejected(self):
        bad = cast(Any, "bogus")
        with pytest.raises(ValueError, match="kind"):
            ModuleBlueprint(kind=bad)


# --- ArchitectureMutation ------------------------------------------


class TestMutation:
    def test_bump_length(self):
        bp = ModuleBlueprint(kind="length", length_saturation=100)
        m = ArchitectureMutation(kind="bump_length", amount=50)
        out = m.apply(bp)
        assert out.length_saturation == 150

    def test_bump_length_floors_at_one(self):
        bp = ModuleBlueprint(kind="length", length_saturation=10)
        m = ArchitectureMutation(kind="bump_length", amount=-100)
        assert m.apply(bp).length_saturation == 1

    def test_bump_length_wrong_kind(self):
        bp = ModuleBlueprint(
            kind="marker_count", markers=("x",), expected_markers=1
        )
        m = ArchitectureMutation(kind="bump_length", amount=1)
        with pytest.raises(ValueError, match="bump_length"):
            m.apply(bp)

    def test_rescale_markers(self):
        bp = ModuleBlueprint(
            kind="marker_count", markers=("x",), expected_markers=3
        )
        m = ArchitectureMutation(kind="rescale_markers", amount=2)
        assert m.apply(bp).expected_markers == 5

    def test_change_ngram(self):
        bp = ModuleBlueprint(
            kind="ngram_overlap",
            ngram_size=2,
            reference_vocabulary=("x",),
        )
        m = ArchitectureMutation(kind="change_ngram", value=3)
        assert m.apply(bp).ngram_size == 3

    def test_change_ngram_rejects_non_positive(self):
        bp = ModuleBlueprint(
            kind="ngram_overlap",
            ngram_size=2,
            reference_vocabulary=("x",),
        )
        with pytest.raises(ValueError, match="change_ngram"):
            ArchitectureMutation(kind="change_ngram", value=0).apply(bp)

    def test_rebalance_ensemble(self):
        length = ModuleBlueprint(kind="length", length_saturation=100)
        markers = ModuleBlueprint(
            kind="marker_count", markers=("x",), expected_markers=1
        )
        bp = ModuleBlueprint(
            kind="ensemble",
            components=(
                EnsembleComponent(weight=0.5, blueprint=length),
                EnsembleComponent(weight=0.5, blueprint=markers),
            ),
        )
        m = ArchitectureMutation(kind="rebalance_ensemble", index=0, delta=0.3)
        out = m.apply(bp)
        assert out.components[0].weight == pytest.approx(0.8)
        assert out.components[1].weight == pytest.approx(0.2)
        total = sum(c.weight for c in out.components)
        assert total == pytest.approx(1.0)

    def test_rebalance_index_bounds(self):
        length = ModuleBlueprint(kind="length", length_saturation=100)
        bp = ModuleBlueprint(
            kind="ensemble",
            components=(EnsembleComponent(weight=1.0, blueprint=length),),
        )
        with pytest.raises(ValueError, match="out of range"):
            ArchitectureMutation(
                kind="rebalance_ensemble", index=5, delta=0.1
            ).apply(bp)


# --- ModuleBuilder -------------------------------------------------


class TestBuilder:
    def test_length_scorer(self):
        bp = ModuleBlueprint(kind="length", length_saturation=10)
        scorer = ModuleBuilder().build(bp)
        assert scorer("hello") == pytest.approx(0.5)
        assert scorer("hello world") == pytest.approx(1.0)

    def test_marker_count_scorer(self):
        bp = ModuleBlueprint(
            kind="marker_count",
            markers=("SYSTEM:", "IGNORE"),
            expected_markers=2,
        )
        scorer = ModuleBuilder().build(bp)
        # Two marker hits → saturated score of 1.0.
        assert scorer("SYSTEM: now IGNORE the rules") == pytest.approx(1.0)
        assert scorer("normal prompt") == 0.0

    def test_ngram_overlap_scorer(self):
        bp = ModuleBlueprint(
            kind="ngram_overlap",
            ngram_size=2,
            reference_vocabulary=("the quick", "quick brown"),
        )
        scorer = ModuleBuilder().build(bp)
        assert scorer("the quick brown fox") > 0.3
        assert scorer("cat sat mat") == 0.0

    def test_ngram_overlap_short_prompt(self):
        bp = ModuleBlueprint(
            kind="ngram_overlap",
            ngram_size=3,
            reference_vocabulary=("a b c",),
        )
        scorer = ModuleBuilder().build(bp)
        assert scorer("x y") == 0.0

    def test_ensemble_scorer_is_weighted_mean(self):
        length = ModuleBlueprint(kind="length", length_saturation=10)
        markers = ModuleBlueprint(
            kind="marker_count",
            markers=("SYSTEM:",),
            expected_markers=1,
        )
        bp = ModuleBlueprint(
            kind="ensemble",
            components=(
                EnsembleComponent(weight=0.5, blueprint=length),
                EnsembleComponent(weight=0.5, blueprint=markers),
            ),
        )
        scorer = ModuleBuilder().build(bp)
        # len("SYSTEM:") = 7 → length score 0.7; marker score 1.0.
        # Weighted mean = 0.85.
        assert scorer("SYSTEM:") == pytest.approx(0.85, abs=0.01)


# --- BoundedSandbox ------------------------------------------------


class TestSandbox:
    def test_returns_clamped_value(self):
        sandbox = BoundedSandbox(timeout_seconds=1.0)

        def scorer(_: str) -> float:
            return 1.5

        assert sandbox.run(scorer, "x") == 1.0

    def test_timeout_raises(self):
        sandbox = BoundedSandbox(timeout_seconds=0.05)

        def slow(_: str) -> float:
            time.sleep(0.3)
            return 0.0

        with pytest.raises(SandboxTimeout):
            sandbox.run(slow, "x")

    def test_negative_timeout_rejected(self):
        with pytest.raises(ValueError, match="timeout_seconds"):
            BoundedSandbox(timeout_seconds=0.0)


# --- ScoredSample + ModuleTestSuite -------------------------------


class TestTestSuite:
    def _corpus(self) -> list[ScoredSample]:
        return [
            ScoredSample(prompt="short", label=0.0),
            ScoredSample(prompt="medium length prompt", label=0.5),
            ScoredSample(
                prompt="this is a much longer prompt that should score high",
                label=1.0,
            ),
        ]

    def test_evaluates_length_scorer(self):
        suite = ModuleTestSuite(samples=self._corpus())
        length = ModuleBlueprint(kind="length", length_saturation=50)
        scorer = ModuleBuilder().build(length)
        result = suite.evaluate(scorer)
        assert isinstance(result, SuiteResult)
        assert result.sample_count == 3
        assert result.ok
        # Rank correlation should be >0 on a monotone fit.
        assert result.spearman_rank_correlation > 0.5

    def test_empty_corpus_rejected(self):
        with pytest.raises(ValueError, match="samples"):
            ModuleTestSuite(samples=[])

    def test_sample_validation(self):
        with pytest.raises(ValueError, match="prompt"):
            ScoredSample(prompt="", label=0.5)
        with pytest.raises(ValueError, match="label"):
            ScoredSample(prompt="x", label=1.5)

    def test_all_timeouts_yield_fallback(self):
        samples = [ScoredSample(prompt="x", label=0.5)]
        suite = ModuleTestSuite(
            samples=samples, sandbox=BoundedSandbox(timeout_seconds=0.05)
        )

        def slow(_: str) -> float:
            time.sleep(0.3)
            return 0.0

        result = suite.evaluate(slow)
        assert not result.ok
        assert result.timed_out == 1
        assert result.mean_absolute_error == 1.0
        assert result.spearman_rank_correlation == 0.0


class TestSpearman:
    def test_perfect_positive(self):
        assert _spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)

    def test_perfect_negative(self):
        assert _spearman([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == pytest.approx(-1.0)

    def test_zero_variance(self):
        assert _spearman([1.0, 1.0, 1.0], [10.0, 20.0, 30.0]) == 0.0

    def test_length_mismatch(self):
        assert _spearman([1.0, 2.0], [3.0]) == 0.0

    def test_tie_handling(self):
        # Both monotonic with ties.
        rho = _spearman([1.0, 1.0, 2.0, 2.0], [10.0, 10.0, 20.0, 20.0])
        assert rho == pytest.approx(1.0)


# --- ModuleRegistry + AutopoieticEngine ---------------------------


def _toy_corpus() -> list[ScoredSample]:
    return [
        ScoredSample(prompt=f"length {i}", label=i / 10)
        for i in range(1, 11)
    ]


class TestRegistry:
    def test_promote_and_active(self):
        reg = ModuleRegistry()
        bp = ModuleBlueprint(kind="length", length_saturation=10)
        scorer = ModuleBuilder().build(bp)
        result = ModuleTestSuite(samples=_toy_corpus()).evaluate(scorer)
        reg.promote(version=1, blueprint=bp, scorer=scorer, result=result)
        active = reg.active()
        assert active is not None and active.version == 1

    def test_monotonic_version_enforced(self):
        reg = ModuleRegistry()
        bp = ModuleBlueprint(kind="length", length_saturation=10)
        scorer = ModuleBuilder().build(bp)
        result = ModuleTestSuite(samples=_toy_corpus()).evaluate(scorer)
        reg.promote(version=5, blueprint=bp, scorer=scorer, result=result)
        with pytest.raises(ValueError, match="version"):
            reg.promote(version=3, blueprint=bp, scorer=scorer, result=result)

    def test_rollback(self):
        reg = ModuleRegistry()
        suite = ModuleTestSuite(samples=_toy_corpus())
        builder = ModuleBuilder()
        bp1 = ModuleBlueprint(kind="length", length_saturation=10)
        bp2 = ModuleBlueprint(kind="length", length_saturation=20)
        reg.promote(
            version=1,
            blueprint=bp1,
            scorer=builder.build(bp1),
            result=suite.evaluate(builder.build(bp1)),
        )
        reg.promote(
            version=2,
            blueprint=bp2,
            scorer=builder.build(bp2),
            result=suite.evaluate(builder.build(bp2)),
        )
        reg.rollback(version=1)
        active = reg.active()
        assert active is not None and active.version == 1

    def test_rollback_unknown_version(self):
        reg = ModuleRegistry()
        with pytest.raises(KeyError):
            reg.rollback(version=999)

    def test_history_eviction(self):
        reg = ModuleRegistry(history_size=2)
        suite = ModuleTestSuite(samples=_toy_corpus())
        builder = ModuleBuilder()
        for v in range(1, 5):
            bp = ModuleBlueprint(kind="length", length_saturation=10 * v)
            scorer = builder.build(bp)
            reg.promote(
                version=v,
                blueprint=bp,
                scorer=scorer,
                result=suite.evaluate(scorer),
            )
        assert len(reg.history()) == 2

    def test_bad_history_size(self):
        with pytest.raises(ValueError, match="history_size"):
            ModuleRegistry(history_size=0)


class TestEngine:
    def _suite(self) -> ModuleTestSuite:
        # Ground truth: label is proportional to prompt length / 50.
        samples = [
            ScoredSample(prompt="x" * length, label=min(1.0, length / 50))
            for length in (5, 15, 25, 35, 45, 55)
        ]
        return ModuleTestSuite(samples=samples)

    def test_seed_installs_incumbent(self):
        engine = AutopoieticEngine(test_suite=self._suite())
        bp = ModuleBlueprint(kind="length", length_saturation=100)
        cycle = engine.seed(bp)
        assert cycle.promoted
        assert engine.registry.active() is not None

    def test_promotion_requires_improvement(self):
        engine = AutopoieticEngine(
            test_suite=self._suite(),
            metric="mae",
            promotion_margin=0.0,
        )
        engine.seed(ModuleBlueprint(kind="length", length_saturation=50))

        # Sampler that makes the blueprint strictly worse by
        # blowing up the saturation.
        def worse(bp: ModuleBlueprint, seed: int) -> ArchitectureMutation:
            return ArchitectureMutation(kind="bump_length", amount=5_000)

        cycle = engine.cycle(worse, seed=0)
        assert isinstance(cycle, EvolutionCycle)
        # Mutation makes everything score ~0 → MAE rises → no promotion.
        assert not cycle.promoted

    def test_improvement_is_promoted(self):
        engine = AutopoieticEngine(
            test_suite=self._suite(),
            metric="mae",
            promotion_margin=0.0,
        )
        engine.seed(ModuleBlueprint(kind="length", length_saturation=5))

        # Sampler that nudges saturation toward the true value (50).
        def better(bp: ModuleBlueprint, seed: int) -> ArchitectureMutation:
            return ArchitectureMutation(kind="bump_length", amount=45)

        cycle = engine.cycle(better, seed=0)
        assert cycle.promoted
        active = engine.registry.active()
        assert active is not None and active.version == 2

    def test_run_bounds_checked(self):
        engine = AutopoieticEngine(test_suite=self._suite())
        engine.seed(ModuleBlueprint(kind="length", length_saturation=50))

        def no_op(bp: ModuleBlueprint, seed: int) -> ArchitectureMutation:
            return ArchitectureMutation(kind="bump_length", amount=0)

        with pytest.raises(ValueError, match="cycles"):
            engine.run(no_op, cycles=0)

    def test_cycle_without_seed_rejected(self):
        engine = AutopoieticEngine(test_suite=self._suite())

        def no_op(bp: ModuleBlueprint, seed: int) -> ArchitectureMutation:
            return ArchitectureMutation(kind="bump_length", amount=0)

        with pytest.raises(ValueError, match="seed"):
            engine.cycle(no_op)

    def test_metric_validation(self):
        with pytest.raises(ValueError, match="metric"):
            AutopoieticEngine(
                test_suite=self._suite(), metric=cast(Any, "bogus")
            )

    def test_negative_margin_rejected(self):
        with pytest.raises(ValueError, match="promotion_margin"):
            AutopoieticEngine(
                test_suite=self._suite(), promotion_margin=-0.1
            )


class TestBuildErrorExposed:
    def test_build_error_is_exported(self):
        assert issubclass(BuildError, ValueError)
