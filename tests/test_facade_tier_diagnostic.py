# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — tests of the lazy facades' advanced-tier diagnostic
"""Give the tier diagnostic only for a module the installed tier lacks.

Both lazy facades, ``director_ai`` and ``director_ai.core``, must name the
advanced tier when a symbol's own module is absent, and must pass any other
missing module through unchanged. The real case runs in a subprocess on a
copy of the package sliced as the Core wheel ships it.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

import pytest

import director_ai
import director_ai.core as core

ROOT = Path(__file__).resolve().parents[1]

_PROBE = """
import json, sys
import director_ai, director_ai.core as core
results = {}
for facade, names in json.loads(sys.argv[1]).items():
    module = director_ai if facade == "root" else core
    for name in names:
        try:
            getattr(module, name)
        except ModuleNotFoundError as error:
            results[f"{facade}:{name}"] = {
                "message": str(error),
                "name": error.name,
                "cause": type(error.__cause__).__name__,
            }
        else:
            results[f"{facade}:{name}"] = None
print(json.dumps(results))
"""
"""Resolve the named lazy symbols of both facades and report each refusal."""


def _lazy_targets() -> dict[str, tuple[str, str]]:
    """Read both facades' lazy maps from their source, keyed by facade and name.

    A root symbol that the root facade takes from ``director_ai.core`` resolves
    through the core facade when it is one of that facade's lazy symbols, so
    its real target and wording are the core facade's.

    Returns
    -------
    dict[str, tuple[str, str]]
        ``"root:<name>"`` or ``"core:<name>"`` to the absolute module that
        holds the symbol and the facade (``root`` or ``core``) that refuses it.
    """
    targets: dict[str, tuple[str, str]] = {}
    attributes: dict[str, str] = {}
    mapping: dict[str, object] = {}
    for facade, relative, variable, prefix in (
        ("root", "__init__.py", "_LAZY_IMPORTS", "director_ai"),
        ("core", "core/__init__.py", "_LAZY_PAID", "director_ai.core"),
    ):
        source = (ROOT / "src" / "director_ai" / relative).read_text(encoding="utf-8")
        for node in ast.parse(source).body:
            names: list[ast.expr]
            if isinstance(node, ast.Assign):
                names, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                names, value = [node.target], node.value
            else:
                continue
            if any(isinstance(n, ast.Name) and n.id == variable for n in names):
                mapping = ast.literal_eval(value)
        for name, entry in mapping.items():
            module = entry[0] if isinstance(entry, tuple) else entry
            targets[f"{facade}:{name}"] = (f"{prefix}{module}", facade)
            if isinstance(entry, tuple):
                attributes[name] = entry[1]
    for key, (module, facade) in list(targets.items()):
        via = f"core:{attributes.get(key[5:], '')}"
        if facade == "root" and module == "director_ai.core" and via in targets:
            targets[key] = targets[via]
    return targets


def _core_slice(base: Path) -> Path:
    """Return a site directory holding the package as the Core wheel ships it.

    Parameters
    ----------
    base : Path
        An empty directory to build the site in.

    Returns
    -------
    Path
        The site directory to put on ``PYTHONPATH``.
    """
    site = base / "site"
    shutil.copytree(
        ROOT / "src" / "director_ai",
        site / "director_ai",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    manifest = json.loads(
        (ROOT / "packages" / "paid_tier_manifest.json").read_text(encoding="utf-8")
    )
    for module in manifest["paid_modules"]:
        (site / module).unlink(missing_ok=True)
    for pattern in project["tool"]["setuptools"]["packages"]["find"]["exclude"]:
        path = site / pattern.rstrip("*").replace(".", "/")
        if path.is_dir():
            shutil.rmtree(path)
    return site


def _absent(site: Path, dotted: str) -> bool:
    """Return whether a module of the package is missing from *site*.

    Parameters
    ----------
    site : Path
        The sliced site directory.
    dotted : str
        The module's absolute dotted name.

    Returns
    -------
    bool
        True when neither a module file nor a package directory exists.
    """
    path = site / dotted.replace(".", "/")
    return not (path.with_suffix(".py").is_file() or path.is_dir())


def test_a_core_install_names_the_tier_for_every_absent_module(
    tmp_path: Path,
) -> None:
    """Refuse each symbol whose module Core lacks, and resolve every other one."""
    site = _core_slice(tmp_path)
    environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    expected = _lazy_targets()
    names: dict[str, list[str]] = {"root": [], "core": []}
    for key in sorted(expected):
        facade, name = key.split(":")
        names[facade].append(name)
    assert names["root"] == sorted(director_ai.__all__)
    result = subprocess.run(
        [sys.executable, "-P", "-c", _PROBE, json.dumps(names)],
        cwd=tmp_path,
        env={**environment, "PYTHONPATH": str(site)},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    refusals = json.loads(result.stdout.strip().splitlines()[-1])
    assert set(refusals) == set(expected)
    refused = {key for key, value in refusals.items() if value is not None}
    absent = {key for key, (target, _) in expected.items() if _absent(site, target)}
    assert refused == absent
    assert sum(expected[key][1] == "root" for key in refused) == 49
    assert sum(key.startswith("root:") for key in refused) > 49
    for key in refused:
        (target, facade), refusal = expected[key], refusals[key]
        assert f"({target} is not installed)" in refusal["message"], key
        tier = "an advanced tier" if facade == "root" else "the advanced tier"
        assert f"requires {tier}" in refusal["message"], key
        missing = refusal["name"]
        assert target == missing or target.startswith(f"{missing}."), key
        assert _absent(site, missing), key
        assert refusal["cause"] == "ModuleNotFoundError", key


def _raising(name: str | None) -> Callable[..., NoReturn]:
    """Return an import function that fails, reporting *name* as missing.

    Parameters
    ----------
    name : str | None
        The module name the failure reports in ``.name``.

    Returns
    -------
    Callable[..., NoReturn]
        A stand-in for ``importlib.import_module``.
    """

    def fail(module: str, package: str | None = None) -> NoReturn:
        """Fail as an import whose named module cannot be found."""
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    return fail


@pytest.mark.parametrize(
    "missing", ["fastapi", None, "director_ai.voice", "director_ai"]
)
def test_the_root_facade_passes_any_other_missing_module_through(
    monkeypatch: pytest.MonkeyPatch, missing: str | None
) -> None:
    """Re-raise a third-party, an unnamed or an unrelated module's error as it is."""
    monkeypatch.delitem(vars(director_ai), "create_app", raising=False)
    monkeypatch.setattr(importlib, "import_module", _raising(missing))
    with pytest.raises(ModuleNotFoundError) as info:
        _ = director_ai.create_app
    assert info.value.name == missing
    assert "advanced tier" not in str(info.value)
    assert info.value.__cause__ is None


@pytest.mark.parametrize(
    ("symbol", "missing", "target"),
    [
        ("create_app", "director_ai.server", "director_ai.server"),
        ("AuditLog", "director_ai.compliance", "director_ai.compliance.audit_log"),
    ],
)
def test_the_root_facade_names_the_tier_for_its_target_or_parent(
    monkeypatch: pytest.MonkeyPatch, symbol: str, missing: str, target: str
) -> None:
    """Name the tiers, keep the missing module's name, and chain the original."""
    monkeypatch.delitem(vars(director_ai), symbol, raising=False)
    monkeypatch.setattr(importlib, "import_module", _raising(missing))
    with pytest.raises(ModuleNotFoundError) as info:
        getattr(director_ai, symbol)
    assert str(info.value) == (
        f"{symbol} requires an advanced tier ({target} is not installed). "
        "Install director-ai-pro, or director-ai-full for the labs packages."
    )
    assert info.value.name == missing
    assert isinstance(info.value, ImportError)
    assert isinstance(info.value.__cause__, ModuleNotFoundError)
    assert info.value.__cause__.name == missing


@pytest.mark.parametrize("missing", ["torch", None, "director_ai.server"])
def test_the_core_facade_passes_any_other_missing_module_through(
    monkeypatch: pytest.MonkeyPatch, missing: str | None
) -> None:
    """Keep a missing dependency's own error instead of blaming the tier."""
    monkeypatch.delitem(vars(core), "tune", raising=False)
    monkeypatch.setattr(core, "import_module", _raising(missing))
    with pytest.raises(ModuleNotFoundError) as info:
        _ = core.tune
    assert info.value.name == missing
    assert "advanced tier" not in str(info.value)


def test_the_core_facade_names_the_tier_for_an_absent_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Name the tier when the whole advanced package is absent."""
    monkeypatch.delitem(vars(core), "tune", raising=False)
    monkeypatch.setattr(core, "import_module", _raising("director_ai.core.calibration"))
    with pytest.raises(ModuleNotFoundError, match="requires the advanced tier") as info:
        _ = core.tune
    assert info.value.name == "director_ai.core.calibration"
    assert isinstance(info.value.__cause__, ModuleNotFoundError)
