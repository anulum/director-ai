# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Code output verification — syntax, imports, and hallucinated API detection.

Verifies generated code for structural correctness:
1. Syntax validation (Python via ast, JSON via json.loads)
2. Import existence checking against known module registry
3. Hallucinated API detection (function calls not in library manifests)

Usage::

    result = verify_code(
        code="import pandas as pd\\ndf = pd.read_csv('data.csv')",
        language="python",
    )
"""

from __future__ import annotations

import ast
import json
import sys

from .types import CodeCheckResult

__all__ = ["verify_code"]

# stdlib modules (Python 3.11+) — not exhaustive but covers common cases
_STDLIB_MODULES = (
    frozenset(sys.stdlib_module_names)
    if hasattr(sys, "stdlib_module_names")
    else frozenset()
)

# Well-known third-party packages
_KNOWN_PACKAGES = frozenset(
    {
        "numpy",
        "np",
        "pandas",
        "pd",
        "torch",
        "tensorflow",
        "tf",
        "sklearn",
        "scipy",
        "matplotlib",
        "plt",
        "seaborn",
        "sns",
        "requests",
        "flask",
        "fastapi",
        "django",
        "sqlalchemy",
        "pydantic",
        "transformers",
        "datasets",
        "tokenizers",
        "PIL",
        "cv2",
        "openai",
        "anthropic",
        "langchain",
        "pytest",
        "unittest",
        "logging",
        "typing",
        "director_ai",
        "director_class_ai",
    }
)


def _extract_imports(tree: ast.Module) -> list[str]:
    """Extract top-level module names from import statements."""
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.append(node.module.split(".")[0])
    return modules


def _extract_function_calls(tree: ast.Module) -> list[str]:
    """Extract dotted function call names (e.g., 'pd.read_csv')."""
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                parts = []
                obj: ast.expr = node.func
                while isinstance(obj, ast.Attribute):
                    parts.append(obj.attr)
                    obj = obj.value
                if isinstance(obj, ast.Name):
                    parts.append(obj.id)
                parts.reverse()
                calls.append(".".join(parts))
            elif isinstance(node.func, ast.Name):
                calls.append(node.func.id)
    return calls


def verify_code(
    code: str,
    language: str = "python",
    known_modules: set[str] | None = None,
    api_manifest: dict[str, set[str]] | None = None,
) -> CodeCheckResult:
    """Verify generated code for syntax, imports, and API correctness.

    Parameters
    ----------
    code : str
        The generated code string.
    language : str
        "python" or "json". Others return syntax-only check.
    known_modules : set[str] | None
        Additional known module names beyond stdlib + common packages.
    api_manifest : dict[str, set[str]] | None
        ``{module_alias: {known_function_names}}``.
        E.g., ``{"pd": {"read_csv", "DataFrame", "merge"}}``.
        Calls to functions not in this set are flagged as hallucinated.

    Returns
    -------
    CodeCheckResult
    """
    if language == "json":
        try:
            json.loads(code)
            return CodeCheckResult(
                syntax_valid=True,
                unknown_imports=[],
                hallucinated_apis=[],
                error_count=0,
            )
        except (json.JSONDecodeError, TypeError) as e:
            return CodeCheckResult(
                syntax_valid=False,
                unknown_imports=[],
                hallucinated_apis=[],
                error_count=1,
                parse_error=str(e),
            )

    if language != "python":
        return CodeCheckResult(
            syntax_valid=True,
            unknown_imports=[],
            hallucinated_apis=[],
            error_count=0,
            parse_error=f"Language '{language}' — syntax check not supported, skipped",
        )

    # Python verification
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return CodeCheckResult(
            syntax_valid=False,
            unknown_imports=[],
            hallucinated_apis=[],
            error_count=1,
            parse_error=f"SyntaxError: {e.msg} (line {e.lineno})",
        )

    all_known = _STDLIB_MODULES | _KNOWN_PACKAGES
    if known_modules:
        all_known = all_known | known_modules

    imports = _extract_imports(tree)
    unknown_imports = [m for m in imports if m not in all_known]

    hallucinated_apis: list[str] = []
    if api_manifest:
        calls = _extract_function_calls(tree)
        for call in calls:
            parts = call.split(".", 1)
            if len(parts) == 2:
                module_alias, remainder = parts
                func_name = remainder.split(".")[0]
                if (
                    module_alias in api_manifest
                    and func_name not in api_manifest[module_alias]
                ):
                    hallucinated_apis.append(call)

    error_count = len(unknown_imports) + len(hallucinated_apis)

    return CodeCheckResult(
        syntax_valid=True,
        unknown_imports=unknown_imports,
        hallucinated_apis=hallucinated_apis,
        error_count=error_count,
    )
