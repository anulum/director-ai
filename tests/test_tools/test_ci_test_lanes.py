# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — every test file runs in exactly one CI lane
"""Every tracked test file is collected by exactly one CI lane that has its tools.

The Python partitions collect ``tests/``; native toolchain tests live with the
lane that installs their toolchain. Each completeness lane is declared here with
its workflow, the command that collects it and the paths that command names.
A file under no completeness lane would never run, and a file under two would
run twice, or with the wrong tools. The extras lanes are deliberately not
completeness lanes: they re-run named ``tests/`` files with optional
dependencies installed, so those files are collected by the partitions and
again by an extras lane by design. The one extras entry declared here, the
standalone training runner test, is collected by no other lane.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import pytest

from tools.audit_ci_workflow_modularity import read_workflow
from tools.ci_workflow_inventory import object_mapping

ROOT = Path(__file__).resolve().parents[2]


class Lane(NamedTuple):
    """One completeness lane: where it is collected, and what it collects."""

    workflow: str
    command: str
    collected: tuple[str, ...]


LANES: dict[str, Lane] = {
    "tests/": Lane("ci-python.yml", r"-m pytest tests/ -v --tb=short", ("tests/",)),
    "tools/quality_ratchet_rust_tests/": Lane(
        "ci-rust.yml",
        r"python -m pytest tools/quality_ratchet_rust_tests$",
        ("tools/quality_ratchet_rust_tests/",),
    ),
    "formal/HaltMonitor/tests/": Lane(
        "ci-lean.yml",
        r"python -m pytest formal/HaltMonitor/tests$",
        ("formal/HaltMonitor/tests/",),
    ),
    "packages/director-ai-lite/tests/": Lane(
        "ci-lite.yml",
        r"cd packages/director-ai-lite\nPYTHONPATH=src pytest tests/ -q",
        ("packages/director-ai-lite/tests/",),
    ),
    "tools/quality_ratchet_julia_tests/": Lane(
        "ci-quality.yml",
        r'"\$QUALITY_PYTHON" -m pytest tools/quality_ratchet_julia_tests ',
        ("tools/quality_ratchet_julia_tests/",),
    ),
    "tools/quality_ratchet_required_tests/": Lane(
        "ci-quality.yml",
        r"^\s*tools/quality_ratchet_required_tests$",
        ("tools/quality_ratchet_required_tests/",),
    ),
    "tools/run_training_standalone_tests/": Lane(
        "ci-extras.yml",
        r"^\s*tools/run_training_standalone_tests/test_run_training_standalone\.py$",
        ("tools/run_training_standalone_tests/test_run_training_standalone.py",),
    ),
}
"""Each lane root mapped to its workflow, collecting command and collected paths."""
EXCLUSIONS = re.compile(
    r"(?<!\S)(?:--ignore(?:-glob)?(?:=|\s)|--deselect(?:=|\s)|-k(?:=|\s|$))", re.M
)
"""Pytest options that would let a lane's command skip files it seems to collect."""
NATIVE_FIXTURES = re.compile(r"\b(?:rust_tools|lean_binary|julia_binary)\b")
THIS_FILE = Path(__file__).resolve().relative_to(ROOT).as_posix()
"""This file names the native fixtures to search for them, so it is not scanned."""


def lane_refusals(
    lanes: Mapping[str, Lane],
    files: Sequence[str],
    scripts: Mapping[str, Sequence[str]],
) -> list[str]:
    """Return every way the declared lanes fail to collect the tracked test files.

    Parameters
    ----------
    lanes : Mapping[str, Lane]
        Lane roots mapped to their declarations.
    files : Sequence[str]
        Tracked test files, repository-relative.
    scripts : Mapping[str, Sequence[str]]
        Each workflow's ``run`` scripts.

    Returns
    -------
    list[str]
        One refusal per file outside exactly one lane, per file a lane's command
        does not name, per lane whose command is missing or excludes files, and
        per lane with no file. Empty means the lanes are complete and disjoint.
    """
    refusals = []
    for path in files:
        roots = [root for root in lanes if path.startswith(root)]
        if len(roots) != 1:
            refusals.append(f"{path} is in {len(roots)} lanes")
            continue
        collected = lanes[roots[0]].collected
        if not any(path == entry or path.startswith(entry) for entry in collected):
            refusals.append(
                f"{path} is under {roots[0]} but its command never names it"
            )
    for root, lane in lanes.items():
        pattern = re.compile(lane.command, re.M)
        steps = [
            script
            for script in scripts.get(lane.workflow, ())
            if pattern.search(script)
        ]
        if not steps:
            refusals.append(f"{root} has no collecting command in {lane.workflow}")
        refusals.extend(
            f"{root} is collected by a command that excludes files"
            for step in steps
            if EXCLUSIONS.search(step)
        )
        if not any(path.startswith(root) for path in files):
            refusals.append(f"{root} holds no test file")
    return refusals


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


def test_the_repository_lanes_are_complete_and_disjoint() -> None:
    """Collect each tracked test file in exactly one lane, by a command that names it."""
    scripts = {lane.workflow: _run_scripts(lane.workflow) for lane in LANES.values()}
    assert lane_refusals(LANES, _test_files(), scripts) == []


DIRECTORY_LANE = Lane("ci.yml", r"pytest suite/", ("suite/",))
FILE_LANE = Lane("ci.yml", r"pytest named/test_one\.py$", ("named/test_one.py",))
COMMANDS = ["pytest suite/", "pytest named/test_one.py"]


@pytest.mark.parametrize(
    ("files", "commands", "refusal"),
    [
        (
            ["suite/test_a.py", "named/test_one.py", "named/test_two.py"],
            COMMANDS,
            "named/test_two.py is under named/ but its command never names it",
        ),
        (
            ["suite/test_a.py", "named/test_one.py"],
            ["pytest suite/ --ignore=suite/test_a.py", "pytest named/test_one.py"],
            "suite/ is collected by a command that excludes files",
        ),
        (
            ["suite/test_a.py", "named/test_one.py"],
            ["pytest suite/ --ignore-glob=*_a.py", "pytest named/test_one.py"],
            "suite/ is collected by a command that excludes files",
        ),
        (
            ["suite/test_a.py", "named/test_one.py"],
            [
                "pytest suite/ --deselect suite/test_a.py::test_x",
                "pytest named/test_one.py",
            ],
            "suite/ is collected by a command that excludes files",
        ),
        (
            ["suite/test_a.py", "named/test_one.py"],
            ["pytest suite/ -k 'not slow'", "pytest named/test_one.py"],
            "suite/ is collected by a command that excludes files",
        ),
        (
            ["suite/test_a.py", "named/test_one.py", "loose/test_b.py"],
            COMMANDS,
            "loose/test_b.py is in 0 lanes",
        ),
        (
            ["suite/test_a.py", "named/test_one.py"],
            ["pytest suite/"],
            "named/ has no collecting command in ci.yml",
        ),
        (
            ["named/test_one.py"],
            COMMANDS,
            "suite/ holds no test file",
        ),
    ],
    ids=[
        "second-file-in-a-file-named-lane",
        "ignore",
        "ignore-glob",
        "deselect",
        "keyword-selection",
        "file-under-no-lane",
        "lane-command-missing",
        "empty-lane",
    ],
)
def test_each_completeness_gap_is_refused(
    files: list[str], commands: list[str], refusal: str
) -> None:
    """Refuse a file no command names, an excluding command, and a missing lane."""
    lanes = {"suite/": DIRECTORY_LANE, "named/": FILE_LANE}

    assert refusal in lane_refusals(lanes, files, {"ci.yml": commands})


def test_a_file_under_two_lanes_is_refused() -> None:
    """Refuse a file that two declared lane roots both contain."""
    lanes = {
        "suite/": DIRECTORY_LANE,
        "suite/deep/": Lane("ci.yml", r"pytest suite/deep", ("suite/deep/",)),
    }
    commands = ["pytest suite/", "pytest suite/deep"]

    refusals = lane_refusals(
        lanes, ["suite/test_a.py", "suite/deep/test_b.py"], {"ci.yml": commands}
    )

    assert refusals == ["suite/deep/test_b.py is in 2 lanes"]


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
