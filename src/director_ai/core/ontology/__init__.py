# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ontology oracle package

"""Ontological consistency oracle.

A lightweight category / modal-logic enforcer that catches the
class-level contradictions a pure NLI layer misses: asserting
something is both a mammal and a reptile, or that a car is a
subclass of food when the ontology declares otherwise.

* :class:`OntologyGraph` — typed relations (``is_a``,
  ``disjoint_with``, ``part_of``) with cycle detection on ``is_a``
  edges and inherited disjointness across subclass chains.
* :class:`OntologyChecker` — evaluates type assertions (an
  individual is-a class) and flags disjointness violations.
  Integrates with :mod:`~director_ai.core.safety.policy` through
  a separate adapter when operators wire it into the Policy
  layer.

The :class:`OntologyGraph` API is the stable surface: modal
operators (``necessarily``, ``possibly``) and role relations
compose on top without breakage.
"""

from .checker import OntologyChecker, OntologyViolation
from .graph import OntologyCycleError, OntologyGraph

__all__ = [
    "OntologyChecker",
    "OntologyCycleError",
    "OntologyGraph",
    "OntologyViolation",
]
