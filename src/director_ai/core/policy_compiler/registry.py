# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — PolicyRegistry

"""Thread-safe hot-swap registry for :class:`PolicyBundle`.

Readers call :meth:`PolicyRegistry.active` to fetch the current
bundle for a policy name; writers call :meth:`register` to swap in
a new one. The swap is atomic — readers never see a partially
constructed bundle because ``PolicyBundle`` is an immutable
dataclass and ``dict`` assignment under a ``threading.Lock`` is
the write boundary. A monotonically increasing version on the
bundle prevents an older bundle from clobbering a newer one when
two registrations race.
"""

from __future__ import annotations

import threading

from .compiler import PolicyBundle


class PolicyRegistry:
    """Named-bundle registry.

    Parameters
    ----------
    strict_versioning :
        When ``True`` (the default), :meth:`register` raises if a
        caller tries to install a bundle whose version is less
        than or equal to the currently active one. Turn off for
        tests that reuse versions deliberately.
    """

    def __init__(self, *, strict_versioning: bool = True) -> None:
        self._strict = strict_versioning
        self._lock = threading.Lock()
        self._bundles: dict[str, PolicyBundle] = {}

    def register(self, name: str, bundle: PolicyBundle) -> None:
        """Install ``bundle`` under ``name`` atomically.

        Raises :class:`ValueError` when ``strict_versioning`` is
        on and the new bundle's version is not strictly greater
        than the one it replaces.
        """
        if not name:
            raise ValueError("policy name must be non-empty")
        with self._lock:
            previous = self._bundles.get(name)
            if (
                self._strict
                and previous is not None
                and bundle.version <= previous.version
            ):
                raise ValueError(
                    f"new bundle version {bundle.version} must be greater "
                    f"than current version {previous.version} for policy {name!r}"
                )
            self._bundles[name] = bundle

    def active(self, name: str) -> PolicyBundle | None:
        """Return the currently active bundle for ``name``, or
        ``None`` when nothing has been registered yet. Cheap — a
        single dict lookup under the lock."""
        with self._lock:
            return self._bundles.get(name)

    def names(self) -> tuple[str, ...]:
        """Snapshot of currently registered policy names."""
        with self._lock:
            return tuple(self._bundles.keys())

    def unregister(self, name: str) -> bool:
        """Remove ``name`` from the registry. Returns ``True`` when
        a bundle was removed, ``False`` when the name was not
        registered."""
        with self._lock:
            return self._bundles.pop(name, None) is not None
