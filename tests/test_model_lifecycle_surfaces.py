# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Re-export Surface Tests

"""The three re-export surfaces hold names only, so they may be long.

``__init__``, ``contracts`` and ``validators`` are exempt from the kernel's
300-line module rule on one condition: each holds a module docstring,
import statements and one ``__all__`` list of string literals, and nothing
else, so no logic can move into them.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import director_ai.model_lifecycle as kernel

PACKAGE = Path(kernel.__file__).parent
SURFACES = ("__init__.py", "contracts.py", "validators.py")


def surface_violations(source: str) -> list[str]:
    """Return every top-level statement of ``source`` a re-export surface may not hold."""
    tree = ast.parse(source)
    found: list[str] = []
    for index, node in enumerate(tree.body):
        if (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if isinstance(node, ast.Import | ast.ImportFrom):
            continue
        if (
            isinstance(node, ast.Assign)
            and [getattr(target, "id", None) for target in node.targets] == ["__all__"]
            and isinstance(node.value, ast.List)
            and all(
                isinstance(item, ast.Constant) and isinstance(item.value, str)
                for item in node.value.elts
            )
        ):
            continue
        found.append(f"line {node.lineno}: {type(node).__name__}")
    return found


@pytest.mark.parametrize("name", SURFACES)
def test_each_surface_holds_names_only(name: str) -> None:
    """Each surface is a docstring, imports and one literal ``__all__``."""
    source = (PACKAGE / name).read_text(encoding="utf-8")
    assert surface_violations(source) == []
    tree = ast.parse(source)
    assigned = [node for node in tree.body if isinstance(node, ast.Assign)]
    assert len(assigned) == 1


@pytest.mark.parametrize(
    "source",
    [
        "def helper() -> None:\n    return None\n",
        "class Rule:\n    pass\n",
        "VALUE = 1\n",
        '__all__ = ["a"] + ["b"]\n',
        '__all__ = ["a", name]\n',
        "if True:\n    import os\n",
        'print("side effect")\n',
        '"""Doc."""\n"""Second string."""\n',
    ],
    ids=[
        "function",
        "class",
        "constant",
        "computed-all",
        "name-in-all",
        "if",
        "call",
        "second-string",
    ],
)
def test_logic_in_a_surface_is_reported(source: str) -> None:
    """Every kind of statement other than names is reported."""
    assert surface_violations(source)


def test_a_names_only_surface_passes() -> None:
    """A docstring, imports and a literal ``__all__`` pass."""
    source = '"""Doc."""\n\nfrom __future__ import annotations\n\nimport os\nfrom .a import b\n\n__all__ = ["b", "os"]\n'
    assert surface_violations(source) == []
