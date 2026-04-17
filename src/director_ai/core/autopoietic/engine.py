# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — AutopoieticEngine + ModuleRegistry

"""Orchestrate candidate generation, evaluation, and hot-swap.

The engine takes a seed :class:`ModuleBlueprint`, a mutation
sampler, and a :class:`ModuleTestSuite`. On every cycle it
applies a mutation, evaluates the resulting module against the
suite, and promotes the candidate only when it beats the
incumbent by ``promotion_margin`` on the chosen metric.

The :class:`ModuleRegistry` keeps the active blueprint + built
scorer with a monotonic version guard and a capped history of
previous promotions. Rolling back is easy: ``rollback(version)``
restores the named historical entry.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from .blueprint import ArchitectureMutation, ModuleBlueprint
from .builder import BoundedSandbox, ModuleBuilder, Scorer
from .testsuite import ModuleTestSuite, SuiteResult

MutationSampler = Callable[[ModuleBlueprint, int], ArchitectureMutation]
Metric = Literal["mae", "rank"]

_VALID_METRICS: frozenset[Metric] = frozenset(("mae", "rank"))


@dataclass(frozen=True)
class _RegistryEntry:
    version: int
    blueprint: ModuleBlueprint
    scorer: Scorer
    result: SuiteResult


class ModuleRegistry:
    """Thread-safe store for the active module and its history.

    Parameters
    ----------
    history_size :
        Maximum number of archived entries retained. Older
        entries fall off. Default 16.
    """

    def __init__(self, *, history_size: int = 16) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        self._lock = threading.Lock()
        self._history_limit = history_size
        self._active: _RegistryEntry | None = None
        self._history: list[_RegistryEntry] = []

    def promote(
        self,
        *,
        version: int,
        blueprint: ModuleBlueprint,
        scorer: Scorer,
        result: SuiteResult,
    ) -> None:
        entry = _RegistryEntry(
            version=version,
            blueprint=blueprint,
            scorer=scorer,
            result=result,
        )
        with self._lock:
            if self._active is not None and version <= self._active.version:
                raise ValueError(
                    f"new version {version} must exceed "
                    f"current {self._active.version}"
                )
            if self._active is not None:
                self._history.append(self._active)
                if len(self._history) > self._history_limit:
                    self._history = self._history[-self._history_limit :]
            self._active = entry

    def rollback(self, *, version: int) -> None:
        with self._lock:
            match = next(
                (e for e in reversed(self._history) if e.version == version),
                None,
            )
            if match is None:
                raise KeyError(f"no archived entry at version {version}")
            if self._active is not None:
                self._history.append(self._active)
                if len(self._history) > self._history_limit:
                    self._history = self._history[-self._history_limit :]
            self._active = match

    def active(self) -> _RegistryEntry | None:
        with self._lock:
            return self._active

    def history(self) -> tuple[_RegistryEntry, ...]:
        with self._lock:
            return tuple(self._history)


@dataclass(frozen=True)
class EvolutionCycle:
    """Report for one :meth:`AutopoieticEngine.cycle` call."""

    attempt_blueprint: ModuleBlueprint
    attempt_result: SuiteResult
    promoted: bool
    incumbent_version: int
    incumbent_result: SuiteResult | None = field(default=None)


class AutopoieticEngine:
    """Coordinate builder + sandbox + test suite + registry.

    Parameters
    ----------
    builder :
        Injected :class:`ModuleBuilder`. Defaults to a fresh one.
    sandbox :
        Injected :class:`BoundedSandbox`. Defaults to 0.5 s.
    test_suite :
        The :class:`ModuleTestSuite` every candidate faces.
    registry :
        The :class:`ModuleRegistry`. Defaults to a fresh one.
    metric :
        ``"mae"`` (minimise) or ``"rank"`` (maximise). Default
        ``"mae"``.
    promotion_margin :
        Candidate wins only when it beats the incumbent by at
        least this much on ``metric``. Default 0.01.
    """

    def __init__(
        self,
        *,
        test_suite: ModuleTestSuite,
        builder: ModuleBuilder | None = None,
        sandbox: BoundedSandbox | None = None,
        registry: ModuleRegistry | None = None,
        metric: Metric = "mae",
        promotion_margin: float = 0.01,
    ) -> None:
        if metric not in _VALID_METRICS:
            raise ValueError(f"metric must be one of {sorted(_VALID_METRICS)}")
        if promotion_margin < 0:
            raise ValueError("promotion_margin must be non-negative")
        self._builder = builder or ModuleBuilder()
        self._sandbox = sandbox or BoundedSandbox(timeout_seconds=0.5)
        self._test_suite = test_suite
        self._registry = registry or ModuleRegistry()
        self._metric = metric
        self._promotion_margin = promotion_margin
        self._next_version = 1

    def seed(self, blueprint: ModuleBlueprint) -> EvolutionCycle:
        """Install ``blueprint`` as the incumbent without running a
        mutation. Required before :meth:`cycle` can promote a
        successor."""
        scorer = self._builder.build(blueprint)
        result = self._test_suite.evaluate(scorer)
        version = self._next_version
        self._next_version += 1
        self._registry.promote(
            version=version,
            blueprint=blueprint,
            scorer=scorer,
            result=result,
        )
        return EvolutionCycle(
            attempt_blueprint=blueprint,
            attempt_result=result,
            promoted=True,
            incumbent_version=version,
            incumbent_result=result,
        )

    def cycle(
        self,
        sampler: MutationSampler,
        *,
        seed: int = 0,
    ) -> EvolutionCycle:
        """Apply one sampler-drawn mutation, evaluate the result,
        promote on success.

        Raises :class:`ValueError` when the registry has no
        incumbent — call :meth:`seed` first."""
        current = self._registry.active()
        if current is None:
            raise ValueError("no incumbent — call seed() first")
        mutation = sampler(current.blueprint, seed)
        candidate = mutation.apply(current.blueprint)
        scorer = self._builder.build(candidate)
        result = self._test_suite.evaluate(scorer)
        promoted = result.ok and self._is_improvement(result, current.result)
        if promoted:
            version = self._next_version
            self._next_version += 1
            self._registry.promote(
                version=version,
                blueprint=candidate,
                scorer=scorer,
                result=result,
            )
            incumbent_version = version
        else:
            incumbent_version = current.version
        active_after = self._registry.active()
        return EvolutionCycle(
            attempt_blueprint=candidate,
            attempt_result=result,
            promoted=promoted,
            incumbent_version=incumbent_version,
            incumbent_result=active_after.result if active_after else None,
        )

    def run(
        self,
        sampler: MutationSampler,
        *,
        cycles: int,
        seed: int = 0,
    ) -> tuple[EvolutionCycle, ...]:
        """Convenience loop. Raises :class:`ValueError` if
        ``cycles <= 0`` so the caller has an honest loop bound."""
        if cycles <= 0:
            raise ValueError("cycles must be positive")
        reports: list[EvolutionCycle] = []
        for i in range(cycles):
            reports.append(self.cycle(sampler, seed=seed + i))
        return tuple(reports)

    def _is_improvement(
        self, candidate: SuiteResult, incumbent: SuiteResult
    ) -> bool:
        if self._metric == "mae":
            return (
                incumbent.mean_absolute_error - candidate.mean_absolute_error
                >= self._promotion_margin
            )
        return (
            candidate.spearman_rank_correlation
            - incumbent.spearman_rank_correlation
            >= self._promotion_margin
        )

    @property
    def registry(self) -> ModuleRegistry:
        return self._registry
