#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Python-only contributor gate

"""Run contributor checks without optional runtime toolchains."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

BLOCKED_TOOLCHAINS = ("cargo", "go", "julia", "lake", "lean", "wasm-pack")
DEFAULT_TESTS = (
    "tests/test_consumer_api.py",
    "tests/test_config.py",
    "tests/test_backends.py",
    "tests/test_python_only_contributor_path.py",
)


@dataclass(frozen=True)
class Gate:
    """One command in the Python-only contributor path."""

    name: str
    command: tuple[str, ...]


def build_gates(
    test_paths: Sequence[str], *, no_tests: bool = False
) -> tuple[Gate, ...]:
    """Return the gate list for the current interpreter."""
    gates = [
        Gate("preflight-fast", (sys.executable, "tools/preflight.py", "--no-tests")),
    ]
    if not no_tests:
        gates.append(
            Gate(
                "pytest-python-smoke",
                (sys.executable, "-m", "pytest", *tuple(test_paths), "-q"),
            )
        )
    _reject_blocked_toolchains(gates)
    return tuple(gates)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Python-only contributor checks.",
    )
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Run only the fast Python lint/security/version gates.",
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print the planned gate commands as JSON and exit.",
    )
    parser.add_argument(
        "tests",
        nargs="*",
        help="Optional pytest paths. Defaults to the Python smoke suite.",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    tests = tuple(args.tests) or DEFAULT_TESTS
    gates = build_gates(tests, no_tests=args.no_tests)
    if args.print_plan:
        print(
            json.dumps(
                [{"name": gate.name, "command": list(gate.command)} for gate in gates],
                indent=2,
            )
        )
        return 0

    env = os.environ.copy()
    env.setdefault("DIRECTOR_AI_PYTHON_ONLY", "1")
    for gate in gates:
        print(f"\n== {gate.name} ==")
        result = subprocess.run(gate.command, cwd=root, env=env)
        if result.returncode != 0:
            print(f"{gate.name} failed with exit code {result.returncode}")
            return result.returncode
    return 0


def _reject_blocked_toolchains(gates: Sequence[Gate]) -> None:
    for gate in gates:
        command_name = Path(gate.command[0]).name
        if command_name in BLOCKED_TOOLCHAINS:
            raise ValueError(
                f"{gate.name} uses blocked optional toolchain {command_name!r}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
