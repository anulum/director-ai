# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — OntologyChecker

"""Consistency checks over a set of ``individual is_a class``
assertions against an :class:`OntologyGraph`.

Two failure modes:

1. The asserted class is not registered in the ontology — the
   checker flags it as ``unknown_class`` so callers can decide
   whether to reject or fall through to an NLI check.
2. Two assertions about the same individual name two classes
   that are disjoint at the graph level (either directly or by
   inherited ``is_a`` chains).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from .graph import OntologyGraph

ViolationKind = Literal["unknown_class", "disjoint_conflict"]


@dataclass(frozen=True)
class OntologyViolation:
    """One check failure. ``subject`` is the individual, ``detail``
    names the classes involved. ``kind`` disambiguates the two
    failure modes so callers can branch cheaply."""

    kind: ViolationKind
    subject: str
    classes: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class Assertion:
    """One ``individual is_a class`` claim."""

    individual: str
    class_name: str

    def __post_init__(self) -> None:
        if not self.individual:
            raise ValueError("Assertion.individual must be non-empty")
        if not self.class_name:
            raise ValueError("Assertion.class_name must be non-empty")


@dataclass
class OntologyChecker:
    """Evaluate a set of :class:`Assertion` against an
    :class:`OntologyGraph` and return every violation.

    Parameters
    ----------
    graph :
        The ontology to check against.
    strict :
        When ``True`` (default), unknown classes are a violation.
        When ``False`` the checker skips unknown-class detection —
        useful for mixed ontologies where some classes are
        discovered at runtime.
    """

    graph: OntologyGraph
    strict: bool = True

    def check(self, assertions: Iterable[Assertion]) -> tuple[OntologyViolation, ...]:
        assertion_list = list(assertions)
        violations: list[OntologyViolation] = []
        known_classes = set(self.graph.classes())

        # Pass 1 — unknown classes.
        if self.strict:
            for a in assertion_list:
                if a.class_name not in known_classes:
                    violations.append(
                        OntologyViolation(
                            kind="unknown_class",
                            subject=a.individual,
                            classes=(a.class_name,),
                            detail=f"class {a.class_name!r} not in ontology",
                        )
                    )

        # Pass 2 — per-individual disjoint conflicts.
        by_individual: dict[str, list[str]] = {}
        for a in assertion_list:
            if a.class_name not in known_classes and self.strict:
                continue
            by_individual.setdefault(a.individual, []).append(a.class_name)

        for subject, classes in by_individual.items():
            reported_pairs: set[tuple[str, str]] = set()
            for i, cls_a in enumerate(classes):
                ancestors_a = self.graph.ancestors(cls_a) | {cls_a}
                for cls_b in classes[i + 1 :]:
                    ancestors_b = self.graph.ancestors(cls_b) | {cls_b}
                    conflict = _find_conflict(self.graph, ancestors_a, ancestors_b)
                    if conflict is None:
                        continue
                    pair = tuple(sorted((cls_a, cls_b)))
                    key = (pair[0], pair[1])
                    if key in reported_pairs:
                        continue
                    reported_pairs.add(key)
                    violations.append(
                        OntologyViolation(
                            kind="disjoint_conflict",
                            subject=subject,
                            classes=(cls_a, cls_b),
                            detail=(
                                f"{cls_a!r} is-a {conflict[0]!r} which is "
                                f"disjoint from {conflict[1]!r} (ancestor of {cls_b!r})"
                            ),
                        )
                    )
        return tuple(violations)


def _find_conflict(
    graph: OntologyGraph,
    ancestors_a: frozenset[str] | set[str],
    ancestors_b: frozenset[str] | set[str],
) -> tuple[str, str] | None:
    """Return the first ``(anc_a, anc_b)`` pair where ``anc_a`` is
    declared disjoint from ``anc_b``. ``None`` means no conflict.
    """
    for anc_a in ancestors_a:
        forbidden = graph.declared_disjoint(anc_a)
        for anc_b in ancestors_b:
            if anc_b in forbidden:
                return (anc_a, anc_b)
    return None
