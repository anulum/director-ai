# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Rust Compute Benchmark Documentation Tests
"""Regression tests for Rust compute benchmark metadata documentation."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_MODULE = ROOT / "benchmarks" / "rust_compute_bench.py"
DOC_ENTRY_RE = re.compile(r"^\s*\d+\.\s+(.+?)\s{2,}—", re.MULTILINE)


def _module_tree() -> ast.Module:
    """Parse the production benchmark runner into an AST."""
    return ast.parse(
        BENCHMARK_MODULE.read_text(encoding="utf-8"),
        filename=str(BENCHMARK_MODULE),
    )


def _module_docstring() -> str:
    """Return the benchmark runner's module docstring."""
    docstring = ast.get_docstring(_module_tree())
    assert docstring is not None
    return docstring


def _dict_string_value(node: ast.Dict, key_name: str) -> str | None:
    """Return a string literal value from a benchmark case dictionary."""
    for key, value in zip(node.keys, node.values, strict=True):
        if not isinstance(key, ast.Constant) or key.value != key_name:
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return None


def _benchmark_names_from_runner() -> list[str]:
    """Return benchmark case names from the production runner registry."""
    for node in ast.walk(_module_tree()):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "benchmarks"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.List):
            continue

        names: list[str] = []
        for item in node.value.elts:
            if not isinstance(item, ast.Dict):
                continue
            name = _dict_string_value(item, "name")
            if name is not None:
                names.append(name)
        if names:
            return names

    raise AssertionError("Could not find benchmark case names in rust_compute_bench.py")


def _benchmark_names_from_docstring() -> list[str]:
    """Return benchmark case names documented in the module docstring."""
    return [
        match.group(1).strip() for match in DOC_ENTRY_RE.finditer(_module_docstring())
    ]


def test_rust_compute_docstring_lists_every_benchmark_case() -> None:
    """Ensure the public benchmark docstring tracks every measured case."""
    assert _benchmark_names_from_docstring() == _benchmark_names_from_runner()
