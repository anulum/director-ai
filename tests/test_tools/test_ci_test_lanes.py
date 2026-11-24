# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — every test file runs in exactly one CI lane
"""Every tracked test file is collected by exactly one CI lane that has its tools.

The Python partitions collect ``tests/``; native toolchain tests live with the
lane that installs their toolchain. A file under no lane would never run, and a
file under two would run twice or with the wrong tools, so each lane is declared
here with the workflow command that collects it, and the two are checked against
the tracked files and the workflow text.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from tools.audit_ci_workflow_modularity import read_workflow
from tools.ci_workflow_inventory import object_mapping

ROOT = Path(__file__).resolve().parents[2]
LANES: dict[str, tuple[str, str]] = {
    "tests/": ("ci-python.yml", r"-m pytest tests/ -v --tb=short"),
    "tools/quality_ratchet_rust_tests/": (
        "ci-rust.yml",
        r"python -m pytest tools/quality_ratchet_rust_tests$",
    ),
    "formal/HaltMonitor/tests/": (
        "ci-lean.yml",
        r"python -m pytest formal/HaltMonitor/tests$",
    ),
    "packages/director-ai-lite/tests/": (
        "ci-lite.yml",
        r"cd packages/director-ai-lite\nPYTHONPATH=src pytest tests/ -q",
    ),
    "tools/quality_ratchet_julia_tests/": (
        "ci-quality.yml",
        r'"\$QUALITY_PYTHON" -m pytest tools/quality_ratchet_julia_tests ',
    ),
    "tools/quality_ratchet_required_tests/": (
        "ci-quality.yml",
        r"^\s*tools/quality_ratchet_required_tests$",
    ),
    "tools/run_training_standalone_tests/": (
        "ci-extras.yml",
        r"^\s*tools/run_training_standalone_tests/test_run_training_standalone\.py$",
    ),
}
"""Each lane root mapped to its workflow and the command that collects it."""
NATIVE_FIXTURES = re.compile(r"\b(?:rust_tools|lean_binary|julia_binary)\b")
THIS_FILE = Path(__file__).resolve().relative_to(ROOT).as_posix()
"""This file names the native fixtures to search for them, so it is not scanned."""


def _test_files() -> list[str]:
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout.decode()
    files: list[str] = []
    for path in filter(None, tracked.split("\0")):
        parts = path.split("/")
        name = parts[-1]
        in_test_directory = parts[0] == "tests" or any(
            part == "tests" or part.endswith("_tests") for part in parts[:-1]
        )
        if in_test_directory and name.startswith("test_") and name.endswith(".py"):
            files.append(path)
    return files


def _run_scripts(workflow: str) -> list[str]:
    jobs = object_mapping(read_workflow(ROOT / ".github/workflows" / workflow)["jobs"])
    scripts: list[str] = []
    for job in jobs.values():
        steps = object_mapping(job).get("steps", [])
        assert isinstance(steps, list)
        scripts.extend(str(object_mapping(step).get("run", "")) for step in steps)
    return scripts


def test_every_test_file_belongs_to_exactly_one_lane() -> None:
    """Collect each tracked test file in one lane, never in none or two."""
    offenders = {
        path: [root for root in LANES if path.startswith(root)]
        for path in _test_files()
    }
    assert {path: roots for path, roots in offenders.items() if len(roots) != 1} == {}


def test_every_lane_is_collected_by_its_workflow() -> None:
    """Find each lane's collecting command in a step of its declared workflow."""
    for root, (workflow, command) in LANES.items():
        pattern = re.compile(command, re.M)
        assert any(pattern.search(script) for script in _run_scripts(workflow)), root


def test_every_lane_holds_a_test_file() -> None:
    """Declare no lane that collects nothing."""
    files = _test_files()
    assert all(any(path.startswith(root) for path in files) for root in LANES)


def test_no_partition_test_needs_a_native_toolchain() -> None:
    """Keep Rust, Lean and Julia toolchain tests out of the Python partitions."""
    needing = [
        path
        for path in _test_files()
        if path.startswith("tests/")
        and path != THIS_FILE
        and NATIVE_FIXTURES.search((ROOT / path).read_text(encoding="utf-8"))
    ]
    assert needing == []
