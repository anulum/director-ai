# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ModuleBuilder + BoundedSandbox

"""Materialise a :class:`ModuleBlueprint` into a callable and
invoke it under a wall-clock timeout.

The builder compiles every blueprint family into a closed-form
pure Python function. No ``eval`` / ``exec`` — the generated
scorers are ordinary callables composed from the blueprint's
validated hyperparameters. :class:`BoundedSandbox` wraps any
callable with a threading-based timeout so a pathological
module cannot stall the autopoietic engine.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from .blueprint import ModuleBlueprint

Scorer = Callable[[str], float]


class BuildError(ValueError):
    """Raised when a blueprint cannot be materialised."""


class SandboxTimeout(TimeoutError):
    """Raised when a sandboxed call exceeds the wall-clock limit."""


class ModuleBuilder:
    """Compile a :class:`ModuleBlueprint` into a scorer."""

    def build(self, blueprint: ModuleBlueprint) -> Scorer:
        if blueprint.kind == "length":
            return self._build_length(blueprint)
        if blueprint.kind == "marker_count":
            return self._build_marker_count(blueprint)
        if blueprint.kind == "ngram_overlap":
            return self._build_ngram_overlap(blueprint)
        if blueprint.kind == "ensemble":
            return self._build_ensemble(blueprint)
        raise BuildError(f"unknown blueprint kind {blueprint.kind!r}")  # pragma: no cover

    def _build_length(self, blueprint: ModuleBlueprint) -> Scorer:
        saturation = float(blueprint.length_saturation)

        def scorer(prompt: str) -> float:
            length = float(len(prompt))
            return max(0.0, min(1.0, length / saturation))

        return scorer

    def _build_marker_count(self, blueprint: ModuleBlueprint) -> Scorer:
        markers = tuple(m.lower() for m in blueprint.markers)
        expected = float(blueprint.expected_markers)

        def scorer(prompt: str) -> float:
            lowered = prompt.lower()
            count = sum(lowered.count(m) for m in markers)
            return max(0.0, min(1.0, count / expected))

        return scorer

    def _build_ngram_overlap(self, blueprint: ModuleBlueprint) -> Scorer:
        reference = frozenset(blueprint.reference_vocabulary)
        n = blueprint.ngram_size

        def scorer(prompt: str) -> float:
            tokens = prompt.lower().split()
            if len(tokens) < n:
                return 0.0
            grams = frozenset(
                " ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
            )
            if not grams:
                return 0.0
            intersection = grams & reference
            union = grams | reference
            if not union:
                return 0.0
            return len(intersection) / len(union)

        return scorer

    def _build_ensemble(self, blueprint: ModuleBlueprint) -> Scorer:
        parts = tuple(
            (component.weight, self.build(component.blueprint))
            for component in blueprint.components
        )

        def scorer(prompt: str) -> float:
            total = 0.0
            for weight, child in parts:
                total += weight * child(prompt)
            return max(0.0, min(1.0, total))

        return scorer


@dataclass(frozen=True)
class _SandboxResult:
    value: float | None
    error: BaseException | None


class BoundedSandbox:
    """Run a scorer with a wall-clock timeout.

    Python's threading does not offer hard thread kill, but the
    sandbox runs the scorer on a daemon thread and returns
    :class:`SandboxTimeout` if the thread does not finish in
    ``timeout_seconds``. Generated scorers are pure Python
    composed from validated blueprints, so they always finish;
    the timeout guards against a blueprint family introduced
    later that needs expensive external work.

    Parameters
    ----------
    timeout_seconds :
        Wall-clock limit. Default 1.0 s — generous for a single
        prompt scorer.
    """

    def __init__(self, *, timeout_seconds: float = 1.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._timeout = timeout_seconds

    def run(self, scorer: Scorer, prompt: str) -> float:
        result = _SandboxResult(value=None, error=None)
        completed = threading.Event()

        def _target() -> None:
            nonlocal result
            try:
                value = float(scorer(prompt))
                result = _SandboxResult(value=value, error=None)
            except BaseException as exc:  # pragma: no cover — defensive
                result = _SandboxResult(value=None, error=exc)
            finally:
                completed.set()

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        if not completed.wait(self._timeout):
            raise SandboxTimeout(
                f"scorer did not finish within {self._timeout:.3f} s"
            )
        if result.error is not None:
            raise result.error  # pragma: no cover — defensive
        if result.value is None:
            raise BuildError("scorer returned None")
        return max(0.0, min(1.0, result.value))
