#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Capability-coverage matrix generator and CI ratchet

"""Generate the public capability-matrix matrix and gate new gaps (WCC-1).

For every public export (``_LAZY_IMPORTS`` in ``src/director_ai/__init__.py``)
and every experimental hook (``EXPERIMENTAL_HOOKS`` in
``src/director_ai/experimental/__init__.py``) the matrix records four
static-inventory facts:

* ``wired`` — the lazy-import target module file exists (runtime importability
  is separately enforced by the production-assert and lazy-import test suites);
* ``tested`` — the symbol name appears in at least one file under ``tests/``;
* ``documented`` — the symbol name appears in at least one page under
  ``docs-site/`` (generated pages excluded);
* ``benchmarked`` — the symbol name appears under ``benchmarks/``
  (informational only, most public symbols have no compute hot path).

``--check`` is a ratchet, not a snapshot: it fails when the committed matrix
drifts, when any symbol is unwired, when a symbol outside the committed
baseline (``tools/capability_matrix_baseline.toml``) is untested or
undocumented — so every NEW public symbol must ship tested and documented —
and when a baseline entry goes stale (the gap was closed but not pruned, or
the symbol no longer exists). Shrinking the baseline is the tracked burndown.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

_JSON_OUTPUT = "docs/_generated/capability_matrix.json"
_MARKDOWN_OUTPUT = "docs/_generated/capability_matrix.md"
_BASELINE = "tools/capability_matrix_baseline.toml"
_PACKAGE_INIT = "src/director_ai/__init__.py"
_EXPERIMENTAL_INIT = "src/director_ai/experimental/__init__.py"
_PACKAGE_ROOT = "src/director_ai"
_TESTS_ROOT = "tests"
_DOCS_ROOT = "docs-site"
_BENCHMARKS_ROOT = "benchmarks"
_SCHEMA_VERSION = "capability-matrix.v1"


def _string_dict_of_tuples(module_path: Path, variable: str) -> dict[str, str]:
    """Parse ``variable = {"name": ("module", ...)}`` into name→module."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in tree.body:
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if variable in targets:
                value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == variable:
                value = node.value
        if not isinstance(value, ast.Dict):
            continue
        result: dict[str, str] = {}
        for key, item in zip(value.keys, value.values, strict=True):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            if isinstance(item, ast.Tuple) and item.elts:
                first = item.elts[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    result[key.value] = first.value
            elif isinstance(item, ast.Constant) and isinstance(item.value, str):
                result[key.value] = item.value
        return result
    raise ValueError(f"{variable} dict not found in {module_path}")


def _module_file_exists(package_root: Path, module: str) -> bool:
    """Whether a lazy-import target module resolves to a source file."""
    if module.startswith("."):
        base = package_root / module.lstrip(".").replace(".", "/")
    else:
        prefix = package_root.name + "."
        if not module.startswith(prefix) and module != package_root.name:
            return False
        remainder = module[len(prefix) :] if module.startswith(prefix) else ""
        base = package_root / remainder.replace(".", "/") if remainder else package_root
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


def _corpus(root: Path, suffix: str, *, exclude_part: str = "") -> list[str]:
    """Read every ``suffix`` file under ``root`` (optionally excluding a part)."""
    if not root.is_dir():
        return []
    texts: list[str] = []
    for path in sorted(root.rglob(f"*{suffix}")):
        if exclude_part and exclude_part in path.parts:
            continue
        texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return texts


def _mentioned(corpus: list[str], name: str) -> bool:
    """Whether ``name`` appears as a whole word anywhere in the corpus."""
    pattern = re.compile(r"\b" + re.escape(name) + r"\b")
    return any(pattern.search(text) for text in corpus)


def build_matrix(repo: Path) -> dict[str, Any]:
    """Build the deterministic coverage matrix for one repository checkout."""
    package_root = repo / _PACKAGE_ROOT
    exports = _string_dict_of_tuples(repo / _PACKAGE_INIT, "_LAZY_IMPORTS")
    hooks = _string_dict_of_tuples(repo / _EXPERIMENTAL_INIT, "EXPERIMENTAL_HOOKS")

    tests = _corpus(repo / _TESTS_ROOT, ".py")
    docs = _corpus(repo / _DOCS_ROOT, ".md", exclude_part="_generated")
    benchmarks = _corpus(repo / _BENCHMARKS_ROOT, ".py")

    rows: list[dict[str, Any]] = []
    for name, module in sorted(exports.items()):
        rows.append(
            {
                "name": name,
                "kind": "public_export",
                "module": module,
                "wired": _module_file_exists(package_root, module),
                "tested": _mentioned(tests, name),
                "documented": _mentioned(docs, name),
                "benchmarked": _mentioned(benchmarks, name),
            }
        )
    for name, module in sorted(hooks.items()):
        rows.append(
            {
                "name": name,
                "kind": "experimental_hook",
                "module": module,
                "wired": _module_file_exists(package_root, module),
                "tested": _mentioned(tests, name),
                "documented": _mentioned(docs, name),
                "benchmarked": _mentioned(benchmarks, name),
            }
        )

    def _gaps(field: str) -> list[str]:
        return [row["name"] for row in rows if not row[field]]

    return {
        "SPDX-License-Identifier": "Apache-2.0 AND BUSL-1.1",
        "schema_version": _SCHEMA_VERSION,
        "counts": {
            "public_exports": len(exports),
            "experimental_hooks": len(hooks),
            "unwired": len(_gaps("wired")),
            "untested": len(_gaps("tested")),
            "undocumented": len(_gaps("documented")),
            "benchmarked": sum(1 for row in rows if row["benchmarked"]),
        },
        "gaps": {
            "unwired": _gaps("wired"),
            "untested": _gaps("tested"),
            "undocumented": _gaps("documented"),
        },
        "rows": rows,
    }


def render_markdown(matrix: dict[str, Any]) -> str:
    """Render the coverage summary and gap tables as Markdown."""
    counts = matrix["counts"]
    gaps = matrix["gaps"]
    lines = [
        "# Capability coverage matrix",
        "",
        "Generated by `tools/capability_matrix.py` — do not edit.",
        "Static inventory: name-mention coverage of every public export and",
        "experimental hook across `tests/`, `docs-site/`, and `benchmarks/`.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Public exports | {counts['public_exports']} |",
        f"| Experimental hooks | {counts['experimental_hooks']} |",
        f"| Unwired | {counts['unwired']} |",
        f"| Untested (by name mention) | {counts['untested']} |",
        f"| Undocumented (by name mention) | {counts['undocumented']} |",
        f"| Benchmarked (informational) | {counts['benchmarked']} |",
        "",
    ]
    for label, names in (
        ("Unwired", gaps["unwired"]),
        ("Untested", gaps["untested"]),
        ("Undocumented", gaps["undocumented"]),
    ):
        lines.append(f"## {label} ({len(names)})")
        lines.append("")
        if names:
            lines.extend(f"- `{name}`" for name in names)
        else:
            lines.append("(none)")
        lines.append("")
    return "\n".join(lines)


def _load_baseline(repo: Path) -> dict[str, list[str]]:
    """Load the committed gap baseline (missing file = empty baseline)."""
    path = repo / _BASELINE
    if not path.is_file():
        return {"untested": [], "undocumented": []}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return {
        "untested": list(data.get("untested", [])),
        "undocumented": list(data.get("undocumented", [])),
    }


def check(repo: Path) -> list[str]:
    """Return ratchet violations for the current checkout (empty = pass)."""
    matrix = build_matrix(repo)
    errors: list[str] = []

    json_path = repo / _JSON_OUTPUT
    md_path = repo / _MARKDOWN_OUTPUT
    expected_json = json.dumps(matrix, indent=2, sort_keys=True) + "\n"
    expected_md = render_markdown(matrix)
    if not json_path.is_file() or json_path.read_text(encoding="utf-8") != (
        expected_json
    ):
        errors.append(
            f"stale generated matrix: {_JSON_OUTPUT} "
            "(run: python tools/capability_matrix.py)"
        )
    if not md_path.is_file() or md_path.read_text(encoding="utf-8") != expected_md:
        errors.append(
            f"stale generated matrix: {_MARKDOWN_OUTPUT} "
            "(run: python tools/capability_matrix.py)"
        )

    if matrix["gaps"]["unwired"]:
        errors.append(
            "unwired public symbols (lazy-import target missing): "
            + ", ".join(matrix["gaps"]["unwired"])
        )

    baseline = _load_baseline(repo)
    known_names = {row["name"] for row in matrix["rows"]}
    for field in ("untested", "undocumented"):
        current = set(matrix["gaps"][field])
        allowed = set(baseline[field])
        new_gaps = sorted(current - allowed)
        if new_gaps:
            errors.append(
                f"new {field} public symbols (add tests/docs, do not extend "
                f"the baseline): {', '.join(new_gaps)}"
            )
        stale = sorted(allowed - current)
        if stale:
            errors.append(
                f"stale baseline entries under [{field}] — the gap is closed, "
                f"prune them from {_BASELINE}: {', '.join(stale)}"
            )
        unknown = sorted(allowed - known_names)
        if unknown:
            errors.append(
                f"unknown baseline entries under [{field}] (no such public "
                f"symbol): {', '.join(unknown)}"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: write the matrix, or verify it with ``--check``."""
    parser = argparse.ArgumentParser(
        description="Generate or check the capability-matrix matrix (WCC-1).",
    )
    parser.add_argument("--repo", default=".", help="Repository root.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify outputs, wiring, and the gap ratchet; write nothing.",
    )
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve()

    if args.check:
        errors = check(repo)
        for error in errors:
            print(f"capability-matrix: {error}", file=sys.stderr)
        return 1 if errors else 0

    matrix = build_matrix(repo)
    json_path = repo / _JSON_OUTPUT
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (repo / _MARKDOWN_OUTPUT).write_text(render_markdown(matrix), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {repo / _MARKDOWN_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
