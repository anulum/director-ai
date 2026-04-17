# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — execution-scope taxonomy

"""Execution-layer labels used by the containment primitives.

Four scopes, ordered from least to most consequential. Their
role in :mod:`director_ai.core.containment` is to decide which
outbound calls an agent may issue: only ``production`` touches
real users, real money, real physical systems.
"""

from __future__ import annotations

from typing import Literal, cast, get_args

ContainmentScope = Literal[
    "sandbox",
    "simulator",
    "shadow",
    "production",
]
"""
* ``sandbox`` — unit-test / hermetic, no outbound effects at all.
* ``simulator`` — end-to-end simulator with synthetic data; safe
  to drive actuators that talk to the simulator, not the real fleet.
* ``shadow`` — production traffic is mirrored to the agent but
  its outputs are discarded (A/B / canary).
* ``production`` — agent outputs reach real systems.
"""

_ALL_SCOPES: tuple[ContainmentScope, ...] = get_args(ContainmentScope)


def scope_allows_real_effects(scope: ContainmentScope) -> bool:
    """True when actions taken under *scope* may affect real users.

    Only ``production`` returns True. ``shadow`` sees real traffic
    but its outputs are suppressed, so from the agent's perspective
    it is still a rehearsal.
    """
    return scope == "production"


def validate_scope(scope: str) -> ContainmentScope:
    """Narrow a plain string into the :data:`ContainmentScope`
    literal. Raises :class:`ValueError` on unknown labels.
    """
    if scope not in _ALL_SCOPES:
        known = ", ".join(_ALL_SCOPES)
        raise ValueError(
            f"unknown containment scope {scope!r}; expected one of {known}"
        )
    # The runtime check above proves membership in the literal —
    # ``cast`` documents that narrowing to the static checker.
    return cast(ContainmentScope, scope)
