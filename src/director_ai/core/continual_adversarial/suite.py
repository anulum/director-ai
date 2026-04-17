# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — AdversarialSuite + SuiteVersion

"""Versioned adversarial test-case container.

Each :class:`SuiteVersion` snapshots the mined patterns + the
derived :class:`AdversarialCase` set at a point in time. The
:class:`AdversarialSuite` registry keeps the active version plus
a bounded history so operators can diff suites across time and
roll back on a bad promotion.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .miner import FailurePattern


@dataclass(frozen=True)
class AdversarialCase:
    """One test case derived from a mined pattern."""

    prompt: str
    expected_label: str
    source_pattern: str

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("AdversarialCase.prompt must be non-empty")
        if not self.expected_label:
            raise ValueError("AdversarialCase.expected_label must be non-empty")
        if not self.source_pattern:
            raise ValueError("AdversarialCase.source_pattern must be non-empty")


@dataclass(frozen=True)
class SuiteVersion:
    """Immutable snapshot of a suite at a point in time."""

    version: int
    cases: tuple[AdversarialCase, ...]
    patterns: tuple[FailurePattern, ...]
    promotion_reason: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.version <= 0:
            raise ValueError("SuiteVersion.version must be positive")
        if not self.cases:
            raise ValueError("SuiteVersion.cases must be non-empty")


class AdversarialSuite:
    """Thread-safe versioned suite registry.

    Parameters
    ----------
    history_size :
        Maximum archived versions retained. Older versions fall
        off. Default 16.
    """

    def __init__(self, *, history_size: int = 16) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        self._lock = threading.Lock()
        self._history_limit = history_size
        self._active: SuiteVersion | None = None
        self._history: list[SuiteVersion] = []

    def promote(self, version: SuiteVersion) -> None:
        with self._lock:
            if self._active is not None and version.version <= self._active.version:
                raise ValueError(
                    f"new version {version.version} must exceed "
                    f"current {self._active.version}"
                )
            if self._active is not None:
                self._history.append(self._active)
                if len(self._history) > self._history_limit:
                    self._history = self._history[-self._history_limit :]
            self._active = version

    def active(self) -> SuiteVersion | None:
        with self._lock:
            return self._active

    def history(self) -> tuple[SuiteVersion, ...]:
        with self._lock:
            return tuple(self._history)

    def rollback(self, *, version: int) -> SuiteVersion:
        with self._lock:
            match = next(
                (v for v in reversed(self._history) if v.version == version),
                None,
            )
            if match is None:
                raise KeyError(f"no archived version {version}")
            if self._active is not None:
                self._history.append(self._active)
                if len(self._history) > self._history_limit:
                    self._history = self._history[-self._history_limit :]
            self._active = match
            return match

    def diff(
        self, *, a: int, b: int
    ) -> tuple[tuple[AdversarialCase, ...], tuple[AdversarialCase, ...]]:
        """Return ``(only_in_a, only_in_b)`` as the symmetric diff
        of cases between two versions. Raises :class:`KeyError`
        when either version is not available."""
        with self._lock:
            archive = {self._active.version: self._active} if self._active else {}
            for v in self._history:
                archive[v.version] = v
        if a not in archive:
            raise KeyError(f"version {a} not in history")
        if b not in archive:
            raise KeyError(f"version {b} not in history")
        cases_a = set(archive[a].cases)
        cases_b = set(archive[b].cases)
        only_in_a = tuple(sorted(cases_a - cases_b, key=lambda c: c.prompt))
        only_in_b = tuple(sorted(cases_b - cases_a, key=lambda c: c.prompt))
        return only_in_a, only_in_b
