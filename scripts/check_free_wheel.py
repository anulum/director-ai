# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — free-wheel contents check and import simulation

"""Verify the built free ``director-ai`` wheel is a clean core-only artefact.

Two layers of proof:

1. **Contents** — the wheel must not carry any paid package (the
   ``packages.find.exclude`` globs from the root ``pyproject.toml``), any
   paid single module (``packages/paid_tier_manifest.json``) or the gRPC
   stub type files, and must still carry the free essentials.
2. **Import simulation** — the wheel is unpacked into a temporary directory
   and imported by a subprocess whose ``PYTHONPATH`` points at the unpacked
   tree (``-P`` keeps the current directory off ``sys.path``), proving the
   core imports standalone and the lazy facades raise the friendly
   "advanced tier" error instead of breaking.

The simulation needs the free runtime dependencies (numpy, requests)
importable by the invoking interpreter. Run it with an interpreter that has
no editable ``director-ai`` install — an editable install registers a
meta-path finder that would shadow the unpacked wheel.

Usage: ``python scripts/check_free_wheel.py <dist-dir>``
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "packages" / "paid_tier_manifest.json"

_FREE_ESSENTIALS = (
    "director_ai/__init__.py",
    "director_ai/core/__init__.py",
    "director_ai/py.typed",
    "director_ai/cli.py",
    "director_ai/_cli_verify.py",
    "director_ai/guard.py",
)

_SIMULATION = """
import importlib.util

import director_ai

assert director_ai.__version__, "free wheel lost its version"

import director_ai.cli
import director_ai.core as core
import director_ai.guard

for absent in (
    "director_ai.server",
    "director_ai.proxy",
    "director_ai.middleware",
    "director_ai.testing",
    "director_ai.experimental",
    "director_ai.core.calibration",
    "director_ai.core.streaming_repair",
    "director_ai.core.training",
    "director_ai.core.verified_scorer",
    "director_ai.core.sharded_nli",
):
    if importlib.util.find_spec(absent) is not None:
        raise SystemExit(f"paid module importable in the free wheel: {absent}")

try:
    core.LLMGenerator
except ImportError as exc:
    if "advanced tier" not in str(exc):
        raise SystemExit(f"boundary error lost the friendly message: {exc}")
else:
    raise SystemExit("core.LLMGenerator resolved in a core-only install")

print("import simulation OK")
"""


def _paid_package_prefixes() -> list[str]:
    """Return the excluded package globs as wheel path prefixes."""
    root = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    excludes = root["tool"]["setuptools"]["packages"]["find"]["exclude"]
    return [g.rstrip("*").replace(".", "/") + "/" for g in excludes]


def _check_contents(whl: str) -> list[str]:
    """Return content failures for the wheel at *whl*."""
    names = set(zipfile.ZipFile(whl).namelist())
    failures: list[str] = []

    for prefix in _paid_package_prefixes():
        hits = sorted(n for n in names if n.startswith(prefix))
        if hits:
            failures.append(f"ships paid package files under {prefix}: {hits[:3]} …")

    paid_modules = json.loads(MANIFEST.read_text(encoding="utf-8"))["paid_modules"]
    shipped = [m for m in paid_modules if m in names]
    if shipped:
        failures.append(f"ships {len(shipped)} paid single modules: {shipped[:5]} …")

    stubs = [n for n in names if n.startswith("director_ai/proto/")]
    if stubs:
        failures.append(f"ships gRPC stub files: {stubs[:3]} …")

    missing = [e for e in _FREE_ESSENTIALS if e not in names]
    if missing:
        failures.append(f"lost free essentials: {missing}")
    if not any(n.startswith("director_ai/core/models/") for n in names):
        failures.append("lost the core/models data files")

    return failures


def _run_simulation(whl: str) -> int:
    """Unpack *whl* and import it in a clean subprocess; return an exit code."""
    with tempfile.TemporaryDirectory(prefix="free-wheel-sim-") as tmp:
        unpack = Path(tmp) / "wheel"
        zipfile.ZipFile(whl).extractall(unpack)
        env = dict(os.environ)
        prior = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{unpack}{os.pathsep}{prior}" if prior else str(unpack)
        proc = subprocess.run(
            [sys.executable, "-P", "-c", _SIMULATION],
            cwd=tmp,
            env=env,
            capture_output=True,
            text=True,
        )
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode


def main(dist_dir: str) -> int:
    """Check the single free wheel in *dist_dir*; return an exit code."""
    matches = glob.glob(f"{dist_dir}/*.whl")
    if len(matches) != 1:
        print(f"FAIL: expected exactly one wheel in {dist_dir}, found {matches}")
        return 1
    whl = matches[0]

    failures = _check_contents(whl)
    if failures:
        for f in failures:
            print(f"FAIL: {whl}: {f}")
        return 1

    if _run_simulation(whl) != 0:
        print(f"FAIL: {whl}: import simulation failed")
        return 1

    count = len(zipfile.ZipFile(whl).namelist())
    print(f"OK: {whl.split('/')[-1]} (free) — {count} files, core-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
