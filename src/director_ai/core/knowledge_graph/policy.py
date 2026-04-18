# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — traversal policy

"""Role + permission check for one edge in the knowledge graph.

A :class:`Principal` carries a role, a set of permissions, and an
optional tenant scope. The :class:`TraversalPolicy` on every edge
states which roles may traverse, which permissions are required,
whether the tenant must match, and the allowed
:class:`TraversalAction`. Policies are immutable dataclasses so
multiple edges can share the same policy instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TraversalAction = Literal["delegate", "compose", "escalate", "invoke"]

_VALID_ACTIONS: frozenset[TraversalAction] = frozenset(
    ("delegate", "compose", "escalate", "invoke")
)


@dataclass(frozen=True)
class Principal:
    """Who is trying to traverse.

    ``role`` is a string — "user", "auditor", "admin",
    per-tenant roles, whatever the deployment uses.
    ``permissions`` is an unordered set. ``tenant_id`` empty
    means "no tenant scoping" — policies that require a tenant
    match reject such principals.
    """

    role: str
    permissions: frozenset[str] = field(default_factory=frozenset)
    tenant_id: str = ""

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("Principal.role must be non-empty")

    def has(self, permission: str) -> bool:
        return permission in self.permissions


@dataclass(frozen=True)
class TraversalPolicy:
    """Policy attached to a :class:`SkillEdge`.

    ``allowed_roles`` — empty set means "any role"; any non-empty
    set restricts to those roles.
    ``required_permissions`` — principal must hold every one.
    ``require_same_tenant`` — when ``True`` the principal's
    ``tenant_id`` must equal the edge's ``tenant_id`` (supplied
    in :meth:`check` by the validator).
    ``allowed_actions`` — which :data:`TraversalAction` values
    the edge accepts. Empty set means "any action".
    """

    allowed_roles: frozenset[str] = field(default_factory=frozenset)
    required_permissions: frozenset[str] = field(default_factory=frozenset)
    require_same_tenant: bool = False
    allowed_actions: frozenset[TraversalAction] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        unknown = self.allowed_actions - _VALID_ACTIONS
        if unknown:
            raise ValueError(
                f"TraversalPolicy.allowed_actions must be a subset of "
                f"{sorted(_VALID_ACTIONS)}; got unknown {sorted(unknown)}"
            )

    @classmethod
    def allow_all(cls) -> TraversalPolicy:
        """Convenience: a policy that accepts every principal and
        every action. Useful as a default edge policy in open
        graphs where permission is enforced elsewhere."""
        return cls()

    def check(
        self,
        principal: Principal,
        *,
        action: TraversalAction,
        edge_tenant_id: str = "",
    ) -> tuple[bool, str]:
        """Return ``(allowed, reason)``. ``reason`` is human-readable."""
        if action not in _VALID_ACTIONS:
            return False, f"action {action!r} is not a valid TraversalAction"
        if self.allowed_actions and action not in self.allowed_actions:
            return False, (
                f"action {action!r} not in allowed actions "
                f"{sorted(self.allowed_actions)}"
            )
        if self.allowed_roles and principal.role not in self.allowed_roles:
            return False, (
                f"role {principal.role!r} not in allowed roles "
                f"{sorted(self.allowed_roles)}"
            )
        missing = self.required_permissions - principal.permissions
        if missing:
            return False, f"missing permissions: {sorted(missing)}"
        if self.require_same_tenant and (
            not principal.tenant_id or principal.tenant_id != edge_tenant_id
        ):
            return False, (
                f"tenant mismatch: principal {principal.tenant_id!r} "
                f"vs edge {edge_tenant_id!r}"
            )
        return True, "allowed"

    def merge(self, other: TraversalPolicy) -> TraversalPolicy:
        """Intersect two policies — the result is the strictest of
        the two. Used when composing an edge policy with a
        graph-level default.

        * Roles: intersection of both, or the non-empty one when
          the other is empty (empty = unrestricted).
        * Permissions: union.
        * ``require_same_tenant``: logical OR.
        * Actions: intersection, or the non-empty one.
        """
        return TraversalPolicy(
            allowed_roles=_intersect_allow_all(self.allowed_roles, other.allowed_roles),
            required_permissions=self.required_permissions | other.required_permissions,
            require_same_tenant=self.require_same_tenant or other.require_same_tenant,
            allowed_actions=_intersect_actions(
                self.allowed_actions, other.allowed_actions
            ),
        )


def _intersect_allow_all(a: frozenset[str], b: frozenset[str]) -> frozenset[str]:
    """Semantic intersection where empty means 'any'. Both empty →
    empty; one empty → the other; both non-empty → set intersection.
    """
    if not a:
        return b
    if not b:
        return a
    return a & b


def _intersect_actions(
    a: frozenset[TraversalAction], b: frozenset[TraversalAction]
) -> frozenset[TraversalAction]:
    """Same semantics as :func:`_intersect_allow_all` but with
    preserved Literal element type so the resulting
    ``allowed_actions`` still satisfies ``frozenset[TraversalAction]``.
    """
    if not a:
        return b
    if not b:
        return a
    return a & b
