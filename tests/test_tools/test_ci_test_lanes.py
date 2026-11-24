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

import ast
import re
import subprocess
import tomllib
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
SELECTION = r"(?<!\S)(?:--ignore(?:-glob)?|--deselect|-k|-m)(?:=|\s|$)"
"""Pytest options that select or exclude tests: ``--ignore``, ``--deselect``, ``-k``, ``-m``."""
EXCLUSIONS = re.compile(r"(?<![\w.])pytest(?![\w.])[^\n]*?" + SELECTION, re.M)
"""Any selection option after a ``pytest`` token, so ``python -m pytest`` still passes."""
CONFTEST_EXCLUSIONS = frozenset(
    {
        "collect_ignore",
        "collect_ignore_glob",
        "pytest_ignore_collect",
        "pytest_collection_modifyitems",
    }
)
"""Conftest names that can drop collected files or tests behind a lane's command."""
DECLARED_EXCLUSIONS: frozenset[str] = frozenset()
"""Exclusions a lane needs, as ``path: name`` or ``path: option``; none today."""
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
            if EXCLUSIONS.search(step.replace("\\\n", " "))
        )
        if not any(path.startswith(root) for path in files):
            refusals.append(f"{root} holds no test file")
    return refusals


def config_refusals(
    pyprojects: Mapping[str, str],
    conftests: Mapping[str, str],
    declared: frozenset[str] = DECLARED_EXCLUSIONS,
) -> list[str]:
    """Return every pytest setting that would exclude tests behind a lane's command.

    Parameters
    ----------
    pyprojects : Mapping[str, str]
        Each tracked ``pyproject.toml`` with a ``[tool.pytest.ini_options]``
        table, by path.
    conftests : Mapping[str, str]
        Each tracked ``conftest.py``, by path.
    declared : frozenset[str]
        Exclusions a lane is allowed, each as ``path: name``.

    Returns
    -------
    list[str]
        One refusal per selection option in ``addopts``, per ``norecursedirs``,
        and per conftest name in :data:`CONFTEST_EXCLUSIONS` that is not declared.
    """
    refusals = []
    for path, text in pyprojects.items():
        options = (
            tomllib.loads(text).get("tool", {}).get("pytest", {}).get("ini_options", {})
        )
        addopts = options.get("addopts", "")
        joined = addopts if isinstance(addopts, str) else " ".join(map(str, addopts))
        for match in re.finditer(SELECTION, joined):
            if f"{path}: {match.group().strip()}" not in declared:
                refusals.append(
                    f"{path} addopts select tests with {match.group().strip()}"
                )
        if "norecursedirs" in options and f"{path}: norecursedirs" not in declared:
            refusals.append(f"{path} sets norecursedirs")
    for path, text in conftests.items():
        names = set()
        for node in ast.parse(text).body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                names.add(node.name)
            targets = node.targets if isinstance(node, ast.Assign) else []
            if isinstance(node, ast.AnnAssign | ast.AugAssign):
                targets = [node.target]
            names.update(
                target.id for target in targets if isinstance(target, ast.Name)
            )
        refusals.extend(
            f"{path} defines {name}"
            for name in sorted(names & CONFTEST_EXCLUSIONS)
            if f"{path}: {name}" not in declared
        )
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


def _tracked(name: str) -> list[str]:
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout.decode()
    return [
        path
        for path in filter(None, tracked.split("\0"))
        if path.split("/")[-1] == name
    ]


def test_the_repository_pytest_configuration_excludes_nothing() -> None:
    """Find no excluding option, ``norecursedirs`` or conftest exclusion today."""
    pyprojects = {
        path: (ROOT / path).read_text(encoding="utf-8")
        for path in _tracked("pyproject.toml")
        if "[tool.pytest.ini_options]" in (ROOT / path).read_text(encoding="utf-8")
    }
    conftests = {
        path: (ROOT / path).read_text(encoding="utf-8")
        for path in _tracked("conftest.py")
    }

    assert pyprojects and conftests
    assert config_refusals(pyprojects, conftests) == []


@pytest.mark.parametrize(
    ("commands", "refused"),
    [
        (["pytest suite/ -m 'not slow'", "pytest named/test_one.py"], True),
        (["pytest suite/ -m=slow", "pytest named/test_one.py"], True),
        (["python -m pytest suite/", "pytest named/test_one.py"], False),
        (
            [
                "coverage run -m pytest suite/ -v \\\n  -k 'not slow'",
                "pytest named/test_one.py",
            ],
            True,
        ),
    ],
    ids=[
        "marker-after-pytest",
        "marker-equals",
        "module-run-passes",
        "keyword-on-a-continuation",
    ],
)
def test_a_selection_after_the_pytest_token_is_refused(
    commands: list[str], refused: bool
) -> None:
    """Refuse ``-m`` or ``-k`` after ``pytest``, even on a continuation line."""
    lanes = {"suite/": DIRECTORY_LANE, "named/": FILE_LANE}
    refusals = lane_refusals(
        lanes, ["suite/test_a.py", "named/test_one.py"], {"ci.yml": commands}
    )

    assert (
        "suite/ is collected by a command that excludes files" in refusals
    ) is refused


PYTEST_TABLE = "[tool.pytest.ini_options]\n"


@pytest.mark.parametrize(
    ("pyproject", "conftest", "refusal"),
    [
        (
            PYTEST_TABLE + 'addopts = "-v --ignore=tests/test_x.py"\n',
            "",
            "pyproject.toml addopts select tests with --ignore=",
        ),
        (
            PYTEST_TABLE + 'addopts = ["-v", "-k", "not slow"]\n',
            "",
            "pyproject.toml addopts select tests with -k",
        ),
        (
            PYTEST_TABLE + 'addopts = "-m slow"\n',
            "",
            "pyproject.toml addopts select tests with -m",
        ),
        (
            PYTEST_TABLE + 'norecursedirs = ["tests/old"]\n',
            "",
            "pyproject.toml sets norecursedirs",
        ),
        (
            PYTEST_TABLE,
            'collect_ignore = ["test_x.py"]\n',
            "tests/conftest.py defines collect_ignore",
        ),
        (
            PYTEST_TABLE,
            'collect_ignore_glob: list[str] = ["*_x.py"]\n',
            "tests/conftest.py defines collect_ignore_glob",
        ),
        (
            PYTEST_TABLE,
            "def pytest_ignore_collect(collection_path, config):\n    return True\n",
            "tests/conftest.py defines pytest_ignore_collect",
        ),
        (
            PYTEST_TABLE,
            "def pytest_collection_modifyitems(config, items):\n    items.clear()\n",
            "tests/conftest.py defines pytest_collection_modifyitems",
        ),
    ],
    ids=[
        "addopts-ignore",
        "addopts-keyword",
        "addopts-marker",
        "norecursedirs",
        "collect-ignore",
        "collect-ignore-glob",
        "ignore-collect-hook",
        "collection-modify-hook",
    ],
)
def test_each_configuration_exclusion_is_refused(
    pyproject: str, conftest: str, refusal: str
) -> None:
    """Refuse an excluding ``addopts``, ``norecursedirs`` and each conftest exclusion."""
    refusals = config_refusals(
        {"pyproject.toml": pyproject}, {"tests/conftest.py": conftest}
    )

    assert refusal in refusals


def test_a_declared_exclusion_is_allowed() -> None:
    """Allow a conftest exclusion only when the lane table declares it."""
    conftest = {"tests/conftest.py": 'collect_ignore = ["test_x.py"]\n'}

    assert (
        config_refusals({}, conftest, frozenset({"tests/conftest.py: collect_ignore"}))
        == []
    )
    assert config_refusals({}, conftest) == ["tests/conftest.py defines collect_ignore"]
