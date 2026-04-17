# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — do-operator intervention

"""Pearl's ``do(X = x)`` operator.

An intervention fixes a variable to a specific value regardless of
its structural equation. Downstream descendants then re-evaluate
with the intervened value as their parent. This is the foundation
for counterfactual reasoning — "what would have happened if X had
been set to x?".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .graph import CausalGraph


@dataclass(frozen=True)
class Intervention:
    """One or more ``do(X = x)`` fixes applied simultaneously.

    Callers construct with a mapping — e.g.
    ``Intervention({"action": "abort", "flag": True})`` — and
    apply via :meth:`apply` to obtain the post-intervention values.
    """

    fixes: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.fixes:
            raise ValueError("Intervention.fixes must be non-empty")

    def apply(
        self,
        graph: CausalGraph,
        inputs: Mapping[str, object],
    ) -> dict[str, object]:
        """Return post-intervention values for every variable.

        Variables in :attr:`fixes` keep their assigned value.
        Variables with no path from any fixed variable keep their
        default structural-equation output. Descendants re-evaluate
        with the intervened parents.
        """
        for name in self.fixes:
            if name not in graph.variables():
                raise ValueError(f"intervention target {name!r} is not in graph")
        values: dict[str, object] = dict(inputs)
        for name in graph.topological_order():
            if name in self.fixes:
                values[name] = self.fixes[name]
                continue
            parents = graph.parents(name)
            if not parents:
                if name not in values:
                    values[name] = graph.equation(name)({})
                continue
            parent_values = {p: values[p] for p in parents}
            values[name] = graph.equation(name)(parent_values)
        return values
