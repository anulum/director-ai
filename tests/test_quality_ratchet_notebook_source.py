# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — notebook projection on every supported interpreter
"""Project the canary notebooks in process; every lane expects the same result.

The pytest lanes run Python 3.11, 3.12 and 3.13. Python 3.12 changed how
``tokenize`` reports a leading ``!``, so these expectations, identical on
every lane, show that the projection does not depend on that change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.quality_ratchet_canaries import CANARY_ROOT
from tools.quality_ratchet_notebook_source import NotebookSource, project_notebook

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / CANARY_ROOT / "notebook-types"


def notebook(*cells: str) -> bytes:
    """Return a Python notebook holding the given code cells.

    Parameters
    ----------
    *cells : str
        The code cells' sources.

    Returns
    -------
    bytes
        The notebook's JSON.
    """
    return json.dumps(
        {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {"language": "python", "name": "python3"},
                "language_info": {"name": "python"},
            },
            "cells": [
                {"cell_type": "code", "metadata": {}, "outputs": [], "source": cell}
                for cell in cells
            ],
        }
    ).encode()


def expected(
    cells: list[str], escapes: set[tuple[int, int]]
) -> tuple[str, list[tuple[int, int]]]:
    """Build the projection the canary expects from the original cells.

    Parameters
    ----------
    cells : list[str]
        The code cells' sources, in order.
    escapes : set[tuple[int, int]]
        The cell and line of each line the projection must comment out.

    Returns
    -------
    tuple[str, list[tuple[int, int]]]
        The generated text and the cell and line of each generated line.
    """
    lines: list[str] = []
    coordinates: list[tuple[int, int]] = []
    for cell, text in enumerate(cells, 1):
        for line, original in enumerate(text.split("\n"), 1):
            lines.append(f"# {original}" if (cell, line) in escapes else original)
            coordinates.append((cell, line))
    return "\n".join(lines) + "\n", coordinates


def test_the_canary_notebook_projects_the_same_on_every_interpreter() -> None:
    """Comment out only the static escape; keep every f-string and ``!=`` line."""
    path = f"{CANARY_ROOT}/notebook-types/canary.ipynb.json"
    data = (ROOT / path).read_bytes()
    cells = ["".join(cell["source"]) for cell in json.loads(data)["cells"]]
    projected = project_notebook(data, path)
    text, coordinates = expected(cells, {(2, 1)})
    assert projected == NotebookSource(
        text, tuple(coordinates), frozenset({15}), ((2, 1),)
    )
    generated = projected.text.split("\n")
    assert generated[coordinates.index((2, 1))] == "# !pip install director-ai"
    assert [generated[coordinates.index((16, line))] for line in (2, 4, 5)] == [
        'print(f"{shown!r}")',
        'padded = f"{shown!s:>10}"',
        "assert shown != 2",
    ]


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("refused-escape", "Unsupported notebook shell escape in {} at cell 1 line 2"),
        (
            "refused-double-escape",
            "Unsupported notebook shell escape in {} at cell 1 line 2",
        ),
        ("refused-assigned-escape", "Unsupported notebook Python syntax in {}"),
    ],
)
def test_the_refused_canaries_are_refused_for_the_same_reason_everywhere(
    name: str, message: str
) -> None:
    """Refuse ``!ls $HOME`` and ``!!ls`` as escapes and ``x = !ls`` as syntax."""
    path = f"{CANARY_ROOT}/notebook-types/{name}.ipynb.json"
    with pytest.raises(ValueError) as refused:
        project_notebook((ROOT / path).read_bytes(), path)
    assert str(refused.value) == message.format(path)


@pytest.mark.parametrize(
    "line",
    [
        'print(f"{shown!r}")',
        'padded = f"{shown!s:>10}"',
        "assert shown != 2",
        'same = f"{shown!a}" != "1"',
    ],
)
def test_a_bang_inside_a_statement_is_never_an_escape(line: str) -> None:
    """Keep every ``!`` that does not start a statement in the compiled text."""
    cell = f"shown = 1\n{line}\n"
    projected = project_notebook(notebook(cell), "notebooks/edge.ipynb")
    assert projected.text == cell + "\n"
    assert projected.escapes == ()


def test_each_projected_escape_names_its_cell_and_line() -> None:
    """Report every static escape at its own cell and line, in order."""
    projected = project_notebook(
        notebook(
            "value = 1\n!pip install director-ai\n",
            "!director-ai --version\nvalue = 2\n!pip install pytest\n",
        ),
        "notebooks/escapes.ipynb",
    )
    assert projected.escapes == ((1, 2), (2, 1), (2, 3))
    assert "# !director-ai --version" in projected.text.split("\n")


@pytest.mark.parametrize(
    ("cell", "message"),
    [
        (
            "!pip install it's",
            "Unsupported notebook shell escape in {} at cell 1 line 1",
        ),
        ("!ls 'x", "Unsupported notebook shell escape in {} at cell 1 line 1"),
        (
            "value = (\n!pip install director-ai\n)\n",
            "Unsupported notebook Python syntax in {}",
        ),
        ("%%bash\necho example\n", "Unsupported notebook Python syntax in {}"),
        (
            "if True:\n    value = 1\n    !pip install director-ai\n",
            "Unsupported notebook Python syntax in {}",
        ),
    ],
)
def test_an_unbalanced_or_embedded_escape_is_refused_the_same_everywhere(
    cell: str, message: str
) -> None:
    """Refuse an escape before its unbalanced quote, and an escape inside a statement."""
    path = "notebooks/refused.ipynb"
    with pytest.raises(ValueError) as refused:
        project_notebook(notebook(cell), path)
    assert str(refused.value) == message.format(path)


def test_the_canary_fixture_files_are_the_four_notebooks() -> None:
    """Keep the fixture to the canary notebook and its three refused notebooks."""
    assert sorted(path.name for path in FIXTURES.iterdir()) == [
        "canary.ipynb.json",
        "refused-assigned-escape.ipynb.json",
        "refused-double-escape.ipynb.json",
        "refused-escape.ipynb.json",
    ]
