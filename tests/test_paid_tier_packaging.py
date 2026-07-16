# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — paid-tier packaging parity tests (ladder P3)
"""Parity tests for the D1/D2 tier re-slice (decisions of 2026-07-16).

The free/paid wheel boundary is declared in four places that must agree:
the root ``pyproject.toml`` exclude list (whole paid packages), the pro and
full ``pyproject.toml`` include lists, ``MANIFEST.in`` (sdist), and
``packages/paid_tier_manifest.json`` (paid single modules + labs split,
consumed by the ``setup.py`` build hooks and the wheel check scripts).
These tests pin the parity so the boundary cannot drift silently.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MANIFEST = json.loads(
    (ROOT / "packages" / "paid_tier_manifest.json").read_text(encoding="utf-8")
)


def _find_config(pyproject: Path) -> dict[str, list[str]]:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    config: dict[str, list[str]] = data["tool"]["setuptools"]["packages"]["find"]
    return config


ROOT_EXCLUDE = set(_find_config(ROOT / "pyproject.toml")["exclude"])
PRO_INCLUDE = set(
    _find_config(ROOT / "packages" / "director-ai-pro" / "pyproject.toml")["include"]
)
FULL_INCLUDE = set(
    _find_config(ROOT / "packages" / "director-ai-full" / "pyproject.toml")["include"]
)
LABS = set(MANIFEST["labs_packages"])
PAID_MODULES = list(MANIFEST["paid_modules"])


def test_every_paid_module_exists_in_the_source_tree() -> None:
    missing = [m for m in PAID_MODULES if not (SRC / m).is_file()]
    assert not missing, f"manifest lists absent modules: {missing}"


def test_paid_modules_are_not_inside_excluded_packages() -> None:
    prefixes = [g.rstrip("*").replace(".", "/") + "/" for g in ROOT_EXCLUDE]
    redundant = [m for m in PAID_MODULES if any(m.startswith(p) for p in prefixes)]
    assert not redundant, f"modules already covered by a package exclude: {redundant}"


def test_paid_modules_never_name_a_parent_init() -> None:
    inits = [m for m in PAID_MODULES if m.endswith("__init__.py")]
    assert not inits, f"manifest must not slice package __init__ files: {inits}"


def test_full_wheel_includes_exactly_the_excluded_packages() -> None:
    assert FULL_INCLUDE == ROOT_EXCLUDE, (
        "full include must mirror the root exclude;"
        f" only-in-full={sorted(FULL_INCLUDE - ROOT_EXCLUDE)}"
        f" only-in-root={sorted(ROOT_EXCLUDE - FULL_INCLUDE)}"
    )


def test_pro_wheel_is_full_minus_labs() -> None:
    assert LABS <= FULL_INCLUDE, sorted(LABS - FULL_INCLUDE)
    assert PRO_INCLUDE == FULL_INCLUDE - LABS, (
        "pro include must be the full include minus the labs set;"
        f" unexpected={sorted(PRO_INCLUDE ^ (FULL_INCLUDE - LABS))}"
    )


def test_every_excluded_package_exists_in_the_source_tree() -> None:
    missing = [
        g for g in ROOT_EXCLUDE if not (SRC / g.rstrip("*").replace(".", "/")).is_dir()
    ]
    assert not missing, f"exclude globs without a source package: {missing}"


def test_sdist_manifest_mirrors_the_wheel_boundary() -> None:
    text = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    pruned = set(re.findall(r"^prune (\S+)$", text, re.M))
    expected_prunes = {"src/" + g.rstrip("*").replace(".", "/") for g in ROOT_EXCLUDE}
    assert pruned == expected_prunes, (
        f"missing prunes={sorted(expected_prunes - pruned)}"
        f" stale prunes={sorted(pruned - expected_prunes)}"
    )

    excluded = set(re.findall(r"^exclude (\S+)$", text, re.M))
    expected_excludes = {"src/" + m for m in PAID_MODULES}
    assert excluded == expected_excludes, (
        f"missing excludes={sorted(expected_excludes - excluded)}"
        f" stale excludes={sorted(excluded - expected_excludes)}"
    )

    for shim in ("packages/tier_build_hooks.py", "packages/paid_tier_manifest.json"):
        assert f"include {shim}" in text, f"sdist must carry {shim}"


def test_free_package_data_does_not_leak_proto_stubs() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = data["tool"]["setuptools"]["package-data"]["director_ai"]
    leaks = [g for g in package_data if "proto" in g or g.endswith(".pyi")]
    assert not leaks, f"root package-data would leak paid stub files: {leaks}"


def test_no_busl_headered_file_escapes_the_paid_boundary() -> None:
    """Every BUSL-1.1 file under ``src/`` must be paid-tier only.

    A BUSL-headered module that is neither inside an excluded package nor a
    paid single module would ship in the Apache-2.0 free wheel — the exact
    leak that shipped ``core/scoring/temporal_refresh.py`` in 3.18.0. This
    guard fails the build before such a file can reach the public wheel.
    """
    excluded_pkg_dirs = [
        (SRC / g.rstrip("*").replace(".", "/")).resolve() for g in ROOT_EXCLUDE
    ]
    paid_module_files = {(SRC / m).resolve() for m in PAID_MODULES}

    leaks: list[str] = []
    for py in (SRC / "director_ai").rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        first_line = py.read_text(encoding="utf-8", errors="replace").partition("\n")[0]
        if first_line.endswith("BUSL-1.1"):
            resolved = py.resolve()
            in_pkg = any(d in resolved.parents for d in excluded_pkg_dirs)
            if not in_pkg and resolved not in paid_module_files:
                leaks.append(str(py.relative_to(SRC)))
    assert not leaks, f"BUSL files outside the paid boundary (would ship free): {leaks}"


def test_build_hooks_agree_with_the_manifest() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "tier_build_hooks", ROOT / "packages" / "tier_build_hooks.py"
    )
    assert spec is not None and spec.loader is not None
    hooks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hooks)

    assert hooks.paid_module_paths() == frozenset(PAID_MODULES)
    assert "director_ai.server" in hooks.paid_module_names()
    assert hooks.FreeTierBuildPy._paid_names == hooks.paid_module_names()
