# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — OntologyGraph

"""Small OWL-inspired graph with ``is_a``, ``disjoint_with``, and
``part_of`` edges.

The graph stores every relation as a set, so repeated additions
are idempotent. ``is_a`` cycles are rejected at edge time;
``disjoint_with`` is always treated as symmetric —
``add_disjoint("bird", "mammal")`` also forbids
``add_is_a("parrot", "mammal")`` once ``parrot is_a bird`` is
present (inherited disjointness).
"""

from __future__ import annotations

from dataclasses import dataclass, field


class OntologyCycleError(ValueError):
    """Raised when an ``is_a`` edge would close a cycle."""


@dataclass
class OntologyGraph:
    """Three-relation ontology.

    All methods accept plain string class names — no separate
    namespace object to keep the surface small. Call
    :meth:`is_subclass_of` / :meth:`disjoint_pairs` after
    construction to query.
    """

    _is_a: dict[str, set[str]] = field(default_factory=dict)
    _disjoint: dict[str, set[str]] = field(default_factory=dict)
    _part_of: dict[str, set[str]] = field(default_factory=dict)

    def add_class(self, name: str) -> None:
        """Register a class with no relations. Useful for classes
        that only appear as assertion targets."""
        if not name:
            raise ValueError("class name must be non-empty")
        self._is_a.setdefault(name, set())
        self._disjoint.setdefault(name, set())
        self._part_of.setdefault(name, set())

    def add_is_a(self, child: str, parent: str) -> None:
        """Declare ``child`` is a subclass of ``parent``.

        Raises :class:`OntologyCycleError` when adding this edge
        would close a cycle (parent already has ``child`` as an
        ancestor).
        """
        self.add_class(child)
        self.add_class(parent)
        if child == parent:
            raise OntologyCycleError(f"self is_a on {child!r}")
        if self.is_subclass_of(parent, child):
            raise OntologyCycleError(
                f"is_a {child!r} -> {parent!r} would cycle (parent is a descendant)"
            )
        self._is_a[child].add(parent)

    def add_disjoint(self, a: str, b: str) -> None:
        """Declare ``a`` and ``b`` mutually exclusive. Symmetric."""
        self.add_class(a)
        self.add_class(b)
        if a == b:
            raise ValueError(f"self disjoint on {a!r}")
        self._disjoint[a].add(b)
        self._disjoint[b].add(a)

    def add_part_of(self, part: str, whole: str) -> None:
        """Declare ``part`` is a component of ``whole``. Not used by
        the default checker yet — exposed for future mereological
        rules so callers can build complete graphs today."""
        self.add_class(part)
        self.add_class(whole)
        if part == whole:
            raise ValueError(f"self part_of on {part!r}")
        self._part_of[part].add(whole)

    def classes(self) -> tuple[str, ...]:
        return tuple(self._is_a.keys())

    def parents(self, name: str) -> frozenset[str]:
        return frozenset(self._is_a.get(name, set()))

    def ancestors(self, name: str) -> frozenset[str]:
        """Transitive ``is_a`` closure (not including ``name``)."""
        seen: set[str] = set()
        stack = list(self._is_a.get(name, set()))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self._is_a.get(current, set()))
        return frozenset(seen)

    def is_subclass_of(self, child: str, parent: str) -> bool:
        """True when ``child`` descends from ``parent``. Reflexive
        on identical names so the checker can short-circuit
        ``individual is_a declared-class`` assertions."""
        if child == parent:
            return True
        return parent in self.ancestors(child)

    def disjoint_pairs(self) -> frozenset[tuple[str, str]]:
        """Every directed disjoint pair, deduplicated. Inherited
        disjointness is computed on demand in :class:`OntologyChecker`
        — the graph only stores the declared pairs."""
        out: set[tuple[str, str]] = set()
        for a, partners in self._disjoint.items():
            for b in partners:
                out.add((a, b))
        return frozenset(out)

    def declared_disjoint(self, name: str) -> frozenset[str]:
        return frozenset(self._disjoint.get(name, set()))
