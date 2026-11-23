# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — tell an absent advanced-tier module from a missing dependency
"""Tell an absent advanced-tier module from a missing third-party dependency.

Both lazy facades, ``director_ai`` and ``director_ai.core``, import a symbol's
module on first access. When that import fails, the module is either absent
because it ships only in an advanced tier, or present but missing a
third-party dependency. Only the first case earns the tier diagnostic; the
second keeps its own error, so a user sees what is really missing.
"""

from __future__ import annotations

PACKAGE = "director_ai"
"""The package whose own modules can be absent from an installed tier."""


def absent_from_tier(missing: str | None, target: str) -> bool:
    """Return whether *target* failed to import because the tier lacks it.

    Parameters
    ----------
    missing : str | None
        The module name a ``ModuleNotFoundError`` reports in ``.name``.
    target : str
        The absolute name of the module that holds the requested symbol.

    Returns
    -------
    bool
        True when *missing* is *target* or one of its parent packages inside
        ``director_ai``: the module ships only in an advanced tier. False for
        any other module, such as a missing third-party dependency, and for
        an error that names no module.
    """
    if missing is None or not missing.startswith(f"{PACKAGE}."):
        return False
    return target == missing or target.startswith(f"{missing}.")
