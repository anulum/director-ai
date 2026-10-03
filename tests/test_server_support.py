# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — server_support route-flattening tests

"""Route-candidate flattening for metrics labels in ``server_support``."""

from __future__ import annotations

import director_ai.server_support as server_support


class _IncludedRouterStub:
    """Provide the candidate-list method used by included-router expansion."""

    def __init__(self, leaves: list[object]) -> None:
        """Store the supplied leaf objects for the included-router test double.

        Parameters
        ----------
        leaves : list[object]
            Objects returned unchanged when candidate expansion is requested.
        """
        self._leaves = leaves

    def effective_candidates(self) -> list[object]:
        """Return the original included-router leaf objects.

        Returns
        -------
        list[object]
            Same leaf list supplied when the test double was constructed.
        """
        return self._leaves


def test_flatten_route_candidates_expands_included_router_mounts() -> None:
    """Expand an included-router test double while retaining a plain leaf."""
    leaf_a, leaf_b, plain = object(), object(), object()
    routes = [_IncludedRouterStub([leaf_a, leaf_b]), plain]

    flattened = list(server_support._flatten_route_candidates(routes))

    assert flattened == [leaf_a, leaf_b, plain]


def test_flatten_route_candidates_keeps_non_callable_attribute_as_leaf() -> None:
    """Retain a route whose candidate attribute cannot be called."""

    class _WeirdRoute:
        """Represent a route with a deliberately non-callable candidate attribute."""

        effective_candidates = "not-callable"

    weird = _WeirdRoute()

    assert list(server_support._flatten_route_candidates([weird])) == [weird]
