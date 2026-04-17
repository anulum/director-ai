# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ontology oracle tests

"""Multi-angle coverage: OntologyGraph construction + cycle
rejection + transitive ancestry, OntologyChecker detects direct
and inherited disjointness, strict/non-strict unknown-class
handling, Assertion validation, and error paths."""

from __future__ import annotations

import pytest

from director_ai.core.ontology import (
    OntologyChecker,
    OntologyCycleError,
    OntologyGraph,
)
from director_ai.core.ontology.checker import Assertion

# --- OntologyGraph --------------------------------------------------


class TestOntologyGraph:
    def test_linear_is_a_chain(self):
        g = OntologyGraph()
        g.add_is_a("sparrow", "bird")
        g.add_is_a("bird", "animal")
        assert g.is_subclass_of("sparrow", "animal")
        assert not g.is_subclass_of("animal", "sparrow")

    def test_reflexive_subclass(self):
        g = OntologyGraph()
        g.add_class("thing")
        assert g.is_subclass_of("thing", "thing")

    def test_cycle_rejected(self):
        g = OntologyGraph()
        g.add_is_a("a", "b")
        g.add_is_a("b", "c")
        with pytest.raises(OntologyCycleError):
            g.add_is_a("c", "a")

    def test_self_is_a_rejected(self):
        g = OntologyGraph()
        with pytest.raises(OntologyCycleError):
            g.add_is_a("x", "x")

    def test_disjoint_is_symmetric(self):
        g = OntologyGraph()
        g.add_disjoint("bird", "mammal")
        assert "mammal" in g.declared_disjoint("bird")
        assert "bird" in g.declared_disjoint("mammal")

    def test_self_disjoint_rejected(self):
        g = OntologyGraph()
        with pytest.raises(ValueError, match="self disjoint"):
            g.add_disjoint("x", "x")

    def test_ancestors_transitive(self):
        g = OntologyGraph()
        g.add_is_a("a", "b")
        g.add_is_a("b", "c")
        g.add_is_a("c", "d")
        assert g.ancestors("a") == frozenset({"b", "c", "d"})

    def test_part_of_records(self):
        g = OntologyGraph()
        g.add_part_of("engine", "car")
        # Foundation scope: no checker consumes part_of yet, but the
        # relation must still be stored so future checkers can see it.
        assert "car" not in g.parents("engine")
        assert "engine" in g.classes()

    def test_empty_name_rejected(self):
        g = OntologyGraph()
        with pytest.raises(ValueError, match="non-empty"):
            g.add_class("")

    def test_idempotent_add_class(self):
        g = OntologyGraph()
        g.add_class("thing")
        g.add_class("thing")
        assert g.classes() == ("thing",)


# --- Assertion ------------------------------------------------------


class TestAssertion:
    def test_valid(self):
        a = Assertion(individual="x", class_name="bird")
        assert a.individual == "x"

    def test_empty_individual_rejected(self):
        with pytest.raises(ValueError, match="individual"):
            Assertion(individual="", class_name="bird")

    def test_empty_class_rejected(self):
        with pytest.raises(ValueError, match="class_name"):
            Assertion(individual="x", class_name="")


# --- OntologyChecker ------------------------------------------------


class TestOntologyChecker:
    def _animal_graph(self) -> OntologyGraph:
        g = OntologyGraph()
        g.add_is_a("sparrow", "bird")
        g.add_is_a("parrot", "bird")
        g.add_is_a("bird", "animal")
        g.add_is_a("dog", "mammal")
        g.add_is_a("mammal", "animal")
        g.add_disjoint("bird", "mammal")
        return g

    def test_consistent_assertions_pass(self):
        g = self._animal_graph()
        checker = OntologyChecker(graph=g)
        violations = checker.check(
            [Assertion("tweety", "sparrow"), Assertion("rex", "dog")]
        )
        assert violations == ()

    def test_direct_disjoint_conflict(self):
        g = self._animal_graph()
        checker = OntologyChecker(graph=g)
        violations = checker.check(
            [Assertion("chimera", "sparrow"), Assertion("chimera", "dog")]
        )
        assert len(violations) == 1
        v = violations[0]
        assert v.kind == "disjoint_conflict"
        assert v.subject == "chimera"
        assert set(v.classes) == {"sparrow", "dog"}

    def test_inherited_disjoint_conflict(self):
        """sparrow is_a bird, which is disjoint from mammal — any
        individual asserted as both ``sparrow`` and ``dog`` (a
        mammal) must raise."""
        g = self._animal_graph()
        checker = OntologyChecker(graph=g)
        violations = checker.check(
            [Assertion("x", "sparrow"), Assertion("x", "dog")]
        )
        assert any(v.kind == "disjoint_conflict" for v in violations)

    def test_unknown_class_flagged_in_strict_mode(self):
        g = self._animal_graph()
        checker = OntologyChecker(graph=g, strict=True)
        violations = checker.check([Assertion("x", "alien")])
        assert any(v.kind == "unknown_class" for v in violations)

    def test_unknown_class_tolerated_in_lenient_mode(self):
        g = self._animal_graph()
        checker = OntologyChecker(graph=g, strict=False)
        violations = checker.check([Assertion("x", "alien")])
        assert all(v.kind != "unknown_class" for v in violations)

    def test_duplicate_pair_reported_once(self):
        g = self._animal_graph()
        checker = OntologyChecker(graph=g)
        violations = checker.check(
            [
                Assertion("x", "sparrow"),
                Assertion("x", "sparrow"),
                Assertion("x", "dog"),
            ]
        )
        disjoint_violations = [v for v in violations if v.kind == "disjoint_conflict"]
        # Each unique unordered pair reported once.
        pairs = {tuple(sorted(v.classes)) for v in disjoint_violations}
        assert len(pairs) == 1

    def test_empty_assertion_set(self):
        g = self._animal_graph()
        checker = OntologyChecker(graph=g)
        assert checker.check([]) == ()
