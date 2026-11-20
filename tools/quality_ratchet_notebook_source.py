# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — typed notebook source projection
"""Project Python notebook cells without executing their code or shell commands."""

from __future__ import annotations

import ast
import inspect
import io
import re
import tokenize
from dataclasses import dataclass

from tools.ci_workflow_inventory import object_mapping
from tools.quality_ratchet_contracts import decode_quality_json

_STATIC_COMMAND = re.compile(r"!(?:pip|director-ai)[ \t]+[A-Za-z0-9_.+\[\]= \t-]+")


@dataclass(frozen=True)
class NotebookSource:
    """Retain compiler input with original notebook coordinates.

    Attributes
    ----------
    text : str
        One module containing the ordered notebook's Python cells.
    coordinates : tuple of tuple of int
        Original one-based cell and line for each generated line.
    asynchronous_cells : frozenset of int
        Cells which the native Python compiler admitted as top-level coroutines.
    """

    text: str
    coordinates: tuple[tuple[int, int], ...]
    asynchronous_cells: frozenset[int]


def _cell_text(source: object) -> str:
    """Read either of the two nbformat JSON source encodings."""
    if isinstance(source, list) and all(isinstance(part, str) for part in source):
        source = "".join(source)
    if not isinstance(source, str) or "\x00" in source:
        raise ValueError("Invalid notebook cell source")
    return source.replace("\r\n", "\n").replace("\r", "\n")


def _python_source(cell: str, path: str) -> list[str]:
    """Translate only independent external commands identified outside literals."""
    lines = cell.split("\n")
    start_of_statement = True
    for token in tokenize.generate_tokens(io.StringIO(cell).readline):
        line, column = token.start
        if token.type == tokenize.COMMENT:
            if re.match(r"#\s*(?:type\s*:\s*ignore\b|mypy\s*:)", token.string):
                raise ValueError(f"Notebook typing suppression in {path}")
            continue
        if token.type == tokenize.NEWLINE:
            start_of_statement = True
            continue
        if token.type == tokenize.NL:
            continue
        if (
            start_of_statement
            and column == 0
            and token.type == tokenize.ERRORTOKEN
            and token.string == "!"
        ):
            original = lines[line - 1]
            if _STATIC_COMMAND.fullmatch(original) is None:
                raise ValueError(f"Unsupported notebook shell escape in {path}")
            # Literal external commands produce no Python bindings or reads.
            lines[line - 1] = "# " + original
        start_of_statement = False
    return lines


def project_notebook(data: bytes, path: str) -> NotebookSource:
    """Project all code cells in JSON order and validate native Python syntax.

    Parameters
    ----------
    data : bytes
        Exact JSON source of a maintained nbformat v4 notebook.
    path : str
        Repository-relative notebook path used for compiler diagnostics.

    Returns
    -------
    NotebookSource
        Generated text, source coordinates and qualified asynchronous cell indices.

    Raises
    ------
    ValueError
        If the notebook format, kernel, source or interactive syntax is unsupported.
    """
    notebook = object_mapping(decode_quality_json(data))
    metadata = object_mapping(notebook.get("metadata"))
    kernel = object_mapping(metadata.get("kernelspec"))
    language = object_mapping(metadata.get("language_info"))
    cells = notebook.get("cells")
    if (
        type(notebook.get("nbformat")) is not int
        or notebook["nbformat"] != 4
        or kernel.get("language") != "python"
        or language.get("name") != "python"
        or not isinstance(cells, list)
    ):
        raise ValueError(f"Unsupported notebook format or kernel in {path}")
    lines: list[str] = []
    coordinates: list[tuple[int, int]] = []
    asynchronous: set[int] = set()
    flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
    try:
        for cell_index, raw in enumerate(cells, 1):
            cell = object_mapping(raw)
            kind = cell.get("cell_type")
            source = _cell_text(cell.get("source"))
            if kind in ("markdown", "raw"):
                continue
            if kind != "code":
                raise ValueError(f"Unsupported notebook cell type in {path}")
            projected = _python_source(source, path)
            text = "\n".join(projected)
            ast.parse(text, filename=path, feature_version=(3, 11))
            code = compile(text, path, "exec", flags=flags, dont_inherit=True)
            if code.co_flags & inspect.CO_COROUTINE:
                asynchronous.add(cell_index)
            coordinates.extend(
                (cell_index, line) for line in range(1, len(projected) + 1)
            )
            lines.extend(projected)
        if not lines:
            raise ValueError(f"No notebook code cells in {path}")
        combined = "\n".join(lines) + "\n"
        compile(combined, path, "exec", flags=flags, dont_inherit=True)
    except (SyntaxError, tokenize.TokenError) as exc:
        raise ValueError(f"Unsupported notebook Python syntax in {path}") from exc
    return NotebookSource(combined, tuple(coordinates), frozenset(asynchronous))
