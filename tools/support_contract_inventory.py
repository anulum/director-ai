# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — the source inventory the support contract is checked against
"""Read the public surface of the source tree without importing it.

The support contract must classify every public export and CLI command, so
the check needs their complete lists. They are read statically, as the
capability matrix reads them, so the check runs with only the linters
installed: the exports from ``_LAZY_IMPORTS``, the commands from
``_command_specs``, the split command's subcommands from its dispatcher, the
HTTP routes from the route decorators, and the settings the production
profile and its generated quickstart configuration turn on.
"""

from __future__ import annotations

import ast
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

PACKAGE_INIT = "src/director_ai/__init__.py"
CLI_MODULE = "src/director_ai/cli.py"
SUBCOMMAND_DISPATCHERS = {
    "compliance": (
        "src/director_ai/cli_verify/compliance.py",
        "_cmd_compliance",
        "sub",
    ),
}
"""Per split command: the module, the dispatcher function and its variable."""
PROFILES_MODULE = "src/director_ai/core/config_profiles.py"
QUICKSTART_MODULE = "src/director_ai/cli_quickstart.py"
QUICKSTART_FUNCTION = "_write_quickstart_config"
PACKAGE_ROOT = "src/director_ai"
ROUTE_METHODS = frozenset({"get", "post", "put", "patch", "delete", "websocket"})
_ENABLED_LINE = re.compile(r"([a-z_][a-z0-9_]*): true\n")


@dataclass(frozen=True)
class Inventory:
    """The public surface the contract is checked against.

    Attributes
    ----------
    exports : tuple[str, ...]
        The package root's lazy exports, sorted.
    commands : tuple[str, ...]
        The CLI commands, sorted.
    subcommands : dict[str, tuple[str, ...]]
        Each split command's dispatched subcommands, sorted.
    extras : tuple[str, ...]
        The project's optional dependency groups, sorted.
    routes : frozenset[str]
        Every path a route decorator in the package declares.
    profile_enabled : tuple[str, ...]
        The settings the ``production`` profile sets to true, sorted.
    quickstart_enabled : tuple[str, ...]
        The settings the generated production quickstart sets to true, sorted.
    profile_values : dict[str, object]
        The ``production`` profile's literal settings.
    """

    exports: tuple[str, ...]
    commands: tuple[str, ...]
    subcommands: dict[str, tuple[str, ...]]
    extras: tuple[str, ...]
    routes: frozenset[str]
    profile_enabled: tuple[str, ...]
    quickstart_enabled: tuple[str, ...]
    profile_values: dict[str, object]


def _parse(root: Path, relative: str) -> ast.Module:
    """Parse one source file of the tree."""
    return ast.parse((root / relative).read_text(encoding="utf-8"), filename=relative)


def _literal_keys(node: ast.Dict, where: str) -> list[str]:
    """Return a dict literal's string keys in source order, refusing any other key."""
    keys = []
    for key in node.keys:
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise ValueError(f"A non-literal key in {where}")
        keys.append(key.value)
    return keys


def lazy_exports(root: Path) -> tuple[str, ...]:
    """Return the names of the package root's ``_LAZY_IMPORTS`` mapping.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    tuple[str, ...]
        The export names, sorted.

    Raises
    ------
    ValueError
        If the mapping is missing or not a literal with string keys.
    """
    for node in _parse(root, PACKAGE_INIT).body:
        target: ast.expr | None = None
        if isinstance(node, ast.AnnAssign):
            target = node.target
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        value = getattr(node, "value", None)
        if isinstance(target, ast.Name) and target.id == "_LAZY_IMPORTS":
            if not isinstance(value, ast.Dict):
                raise ValueError("_LAZY_IMPORTS is not a dict literal")
            return tuple(sorted(_literal_keys(value, "_LAZY_IMPORTS")))
    raise ValueError("The package root defines no _LAZY_IMPORTS")


def _function(tree: ast.Module, name: str, where: str) -> ast.FunctionDef:
    """Return a module-level function definition by name."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ValueError(f"{where} defines no {name}")


def cli_commands(root: Path) -> tuple[str, ...]:
    """Return the commands ``_command_specs`` registers.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    tuple[str, ...]
        The command names, sorted.

    Raises
    ------
    ValueError
        If the function does not return one dict literal with string keys.
    """
    function = _function(_parse(root, CLI_MODULE), "_command_specs", CLI_MODULE)
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.Dict):
        raise ValueError("_command_specs does not return one dict literal")
    return tuple(sorted(_literal_keys(returns[0].value, "_command_specs")))


def dispatched_subcommands(
    root: Path, relative: str, function: str, variable: str
) -> tuple[str, ...]:
    """Return the literals a dispatcher compares its subcommand variable with.

    Parameters
    ----------
    root : Path
        Repository root.
    relative : str
        The dispatcher's module.
    function : str
        The dispatcher function.
    variable : str
        The name holding the subcommand.

    Returns
    -------
    tuple[str, ...]
        Every string ``variable == "…"`` compares against, sorted.

    Raises
    ------
    ValueError
        If the dispatcher compares with no literal.
    """
    found = set()
    for node in ast.walk(_function(_parse(root, relative), function, relative)):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == variable
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq)
            and isinstance(node.comparators[0], ast.Constant)
            and isinstance(node.comparators[0].value, str)
        ):
            found.add(node.comparators[0].value)
    if not found:
        raise ValueError(f"{relative} {function} dispatches no subcommand")
    return tuple(sorted(found))


def declared_routes(root: Path) -> frozenset[str]:
    """Return every path a route decorator in the package declares.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    frozenset[str]
        The literal first arguments of ``@<x>.get(…)``, ``@<x>.post(…)`` and
        the other route methods.
    """
    routes = set()
    for path in sorted((root / PACKAGE_ROOT).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr in ROUTE_METHODS
                    and decorator.args
                    and isinstance(decorator.args[0], ast.Constant)
                    and isinstance(decorator.args[0].value, str)
                ):
                    routes.add(decorator.args[0].value)
    return frozenset(routes)


def production_profile(root: Path) -> dict[str, object]:
    """Return the ``production`` profile's literal settings.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    dict[str, object]
        The settings, as Python literals.

    Raises
    ------
    ValueError
        If no module-level dict literal has a ``production`` entry that is
        itself a dict literal.
    """
    for node in ast.walk(_parse(root, PROFILES_MODULE)):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and key.value == "production"
                and isinstance(value, ast.Dict)
            ):
                names = _literal_keys(value, "the production profile")
                return {
                    name: ast.literal_eval(item)
                    for name, item in zip(names, value.values, strict=True)
                }
    raise ValueError("The profiles module defines no production profile")


def quickstart_enabled(root: Path) -> tuple[str, ...]:
    """Return the settings the generated production quickstart turns on.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    tuple[str, ...]
        Every setting the quickstart writer emits as a ``<setting>: true``
        line, sorted.

    Raises
    ------
    ValueError
        If the writer emits no such line.
    """
    function = _function(
        _parse(root, QUICKSTART_MODULE), QUICKSTART_FUNCTION, QUICKSTART_MODULE
    )
    found = {
        match.group(1)
        for node in ast.walk(function)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        for match in [_ENABLED_LINE.fullmatch(node.value)]
        if match
    }
    if not found:
        raise ValueError("The quickstart writer turns on no setting")
    return tuple(sorted(found))


def read_inventory(root: Path) -> Inventory:
    """Read the whole inventory of a source tree.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    Inventory
        The public surface.

    Raises
    ------
    ValueError, OSError, SyntaxError
        If a source file is missing, unparsable or not in the expected form.
    """
    profile = production_profile(root)
    with (root / "pyproject.toml").open("rb") as handle:
        extras = tomllib.load(handle)["project"]["optional-dependencies"]
    return Inventory(
        exports=lazy_exports(root),
        commands=cli_commands(root),
        subcommands={
            command: dispatched_subcommands(root, *where)
            for command, where in SUBCOMMAND_DISPATCHERS.items()
        },
        extras=tuple(sorted(extras)),
        routes=declared_routes(root),
        profile_enabled=tuple(
            sorted(name for name, value in profile.items() if value is True)
        ),
        quickstart_enabled=quickstart_enabled(root),
        profile_values=profile,
    )
