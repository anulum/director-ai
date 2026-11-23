# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — tests of the static source inventory of the support contract
"""Read the real tree's surface, and refuse each malformed source form."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.support_contract_inventory import (
    CLI_MODULE,
    PACKAGE_INIT,
    PROFILES_MODULE,
    QUICKSTART_MODULE,
    cli_commands,
    declared_routes,
    dispatched_subcommands,
    lazy_exports,
    production_profile,
    quickstart_enabled,
    read_inventory,
)

ROOT = Path(__file__).resolve().parents[1]


def test_the_real_tree_inventory() -> None:
    """Read 227 exports, 43 commands, the compliance subcommands and the defaults."""
    inventory = read_inventory(ROOT)
    assert len(inventory.exports) == 227 and len(set(inventory.exports)) == 227
    assert len(inventory.commands) == 43
    assert inventory.subcommands == {
        "compliance": (
            "anchor",
            "drift",
            "evidence-kit",
            "governance",
            "report",
            "status",
            "verify-anchors",
        )
    }
    assert {"/v1/review", "/v1/health", "/health"} <= inventory.routes
    assert "redact_pii" in inventory.profile_enabled
    assert "knowledge_write_require_signature" in inventory.quickstart_enabled
    assert inventory.profile_values["reranker_enabled"] is False
    assert "server" in inventory.extras


def write(root: Path, relative: str, text: str) -> None:
    """Write one source file under a test tree.

    Parameters
    ----------
    root : Path
        The test tree.
    relative : str
        The file's path in the tree.
    text : str
        Its content.
    """
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_lazy_exports_accept_an_annotated_mapping(tmp_path: Path) -> None:
    """Read an annotated ``_LAZY_IMPORTS`` and refuse other forms."""
    write(tmp_path, PACKAGE_INIT, '_LAZY_IMPORTS: dict = {"B": 1, "A": 2}\n')
    assert lazy_exports(tmp_path) == ("A", "B")
    write(tmp_path, PACKAGE_INIT, "_LAZY_IMPORTS = dict(a=1)\n")
    with pytest.raises(ValueError, match="not a dict literal"):
        lazy_exports(tmp_path)
    write(tmp_path, PACKAGE_INIT, "_LAZY_IMPORTS = {NAME: 1}\n")
    with pytest.raises(ValueError, match="non-literal key"):
        lazy_exports(tmp_path)
    write(tmp_path, PACKAGE_INIT, "OTHER = {}\n")
    with pytest.raises(ValueError, match="defines no _LAZY_IMPORTS"):
        lazy_exports(tmp_path)


def test_cli_commands_need_one_literal_return(tmp_path: Path) -> None:
    """Read the registered names; refuse a missing function or two returns."""
    write(tmp_path, CLI_MODULE, 'def _command_specs():\n    return {"b": 1, "a": 2}\n')
    assert cli_commands(tmp_path) == ("a", "b")
    write(
        tmp_path,
        CLI_MODULE,
        "def _command_specs(x):\n    if x:\n        return {}\n    return {}\n",
    )
    with pytest.raises(ValueError, match="does not return one dict literal"):
        cli_commands(tmp_path)
    write(tmp_path, CLI_MODULE, "def other():\n    return {}\n")
    with pytest.raises(ValueError, match="defines no _command_specs"):
        cli_commands(tmp_path)


def test_a_dispatcher_must_compare_its_variable(tmp_path: Path) -> None:
    """Read ``sub == "…"`` comparisons; refuse a dispatcher with none."""
    write(
        tmp_path,
        "m.py",
        'def run(sub, x):\n    if sub == "b" or x == "c" or sub != "d":\n        return\n    if sub == "a":\n        return\n',
    )
    assert dispatched_subcommands(tmp_path, "m.py", "run", "sub") == ("a", "b")
    write(tmp_path, "m.py", "def run(sub):\n    return sub\n")
    with pytest.raises(ValueError, match="dispatches no subcommand"):
        dispatched_subcommands(tmp_path, "m.py", "run", "sub")


def test_routes_come_from_literal_route_decorators(tmp_path: Path) -> None:
    """Collect literal paths of route methods only."""
    write(
        tmp_path,
        "src/director_ai/app.py",
        "@router.get('/a')\ndef a(): ...\n"
        "@router.post(PATH)\nasync def b(): ...\n"
        "@router.include('/c')\ndef c(): ...\n"
        "@plain\ndef d(): ...\n",
    )
    assert declared_routes(tmp_path) == frozenset({"/a"})


def test_the_production_profile_and_quickstart_must_exist(tmp_path: Path) -> None:
    """Read the literal profile and quickstart lines; refuse their absence."""
    write(
        tmp_path,
        PROFILES_MODULE,
        'P = {"fast": {}, "production": {"a": True, "b": 0}}\n',
    )
    assert production_profile(tmp_path) == {"a": True, "b": 0}
    write(tmp_path, PROFILES_MODULE, 'P = {"production": dict(a=1)}\n')
    with pytest.raises(ValueError, match="no production profile"):
        production_profile(tmp_path)
    write(
        tmp_path,
        QUICKSTART_MODULE,
        'def _write_quickstart_config():\n    return ["a: true\\n", "b: false\\n", 3]\n',
    )
    assert quickstart_enabled(tmp_path) == ("a",)
    write(
        tmp_path,
        QUICKSTART_MODULE,
        'def _write_quickstart_config():\n    return ["b: false\\n"]\n',
    )
    with pytest.raises(ValueError, match="turns on no setting"):
        quickstart_enabled(tmp_path)
