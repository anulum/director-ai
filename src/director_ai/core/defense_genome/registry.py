# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — DefenseRegistry

"""Thread-safe hot-swap registry for the active defence.

A :class:`Defense` is anything that returns a ``[0, 1]`` safety
probability for a rendered prompt (higher = more likely safe).
The :class:`EvolutionEngine` drives the defence, and callers
promote a new defence into service via :meth:`DefenseRegistry.promote`.
A :class:`DefenseSnapshot` ties each promoted defence to a
version + metadata tag so operators can roll back to a named
snapshot if a promoted defence turns out to be worse on a
holdout.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class Defense(Protocol):
    """Anything that scores a prompt's safety in ``[0, 1]``."""

    def score(self, prompt: str) -> float: ...


@dataclass(frozen=True)
class DefenseSnapshot:
    """One archived defence with its version + metadata.

    ``label`` is a human-readable tag ("pre-v3.14" / "fpr-fix")
    that :meth:`DefenseRegistry.rollback_to` matches on.
    ``metadata`` is free-form sidecar data (calibration fold
    size, training date, fitness vs. last genome population).
    """

    version: int
    defense: Defense
    label: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)


class DefenseRegistry:
    """Hot-swap store for the live defence.

    Parameters
    ----------
    strict_versioning :
        When ``True`` (default), :meth:`promote` rejects a new
        defence whose version is less than or equal to the
        currently active one. Callers use :meth:`rollback_to` to
        restore an archived snapshot without tripping the guard.
    history_size :
        Maximum number of archived snapshots retained. Older
        snapshots fall off the end. Default 16.
    """

    def __init__(
        self,
        *,
        strict_versioning: bool = True,
        history_size: int = 16,
    ) -> None:
        if history_size <= 0:
            raise ValueError("history_size must be positive")
        self._strict = strict_versioning
        self._lock = threading.Lock()
        self._active: DefenseSnapshot | None = None
        self._history: list[DefenseSnapshot] = []
        self._history_limit = history_size

    def promote(
        self,
        *,
        defense: Defense,
        version: int,
        label: str = "",
        metadata: Mapping[str, str] | None = None,
    ) -> DefenseSnapshot:
        """Install ``defense`` as the active snapshot atomically."""
        snapshot = DefenseSnapshot(
            version=version,
            defense=defense,
            label=label,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            if (
                self._strict
                and self._active is not None
                and snapshot.version <= self._active.version
            ):
                raise ValueError(
                    f"new defence version {snapshot.version} must be greater "
                    f"than current {self._active.version}"
                )
            if self._active is not None:
                self._history.append(self._active)
                if len(self._history) > self._history_limit:
                    self._history = self._history[-self._history_limit :]
            self._active = snapshot
        return snapshot

    def active(self) -> DefenseSnapshot | None:
        with self._lock:
            return self._active

    def history(self) -> tuple[DefenseSnapshot, ...]:
        with self._lock:
            return tuple(self._history)

    def rollback_to(self, *, label: str) -> DefenseSnapshot:
        """Promote the most recent archived snapshot with ``label``
        back into active service. The current active snapshot is
        pushed onto the history stack; rollback versioning bypasses
        the strict check so demotions are always possible."""
        if not label:
            raise ValueError("label must be non-empty")
        with self._lock:
            match = next(
                (s for s in reversed(self._history) if s.label == label),
                None,
            )
            if match is None:
                raise KeyError(f"no archived snapshot with label {label!r}")
            if self._active is not None:
                self._history.append(self._active)
                if len(self._history) > self._history_limit:
                    self._history = self._history[-self._history_limit :]
            self._active = match
            return match

    def clear(self) -> None:
        """Wipe the registry — used by tests and catastrophic
        recovery procedures only."""
        with self._lock:
            self._active = None
            self._history = []
