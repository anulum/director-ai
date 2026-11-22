# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — verified Python engine artefacts for the notebook profile
"""Check both notebook-profile interpreters against the one pinned build.

The pytest lanes run setup-python builds, never the pinned build, so the
in-process checks here are the negative controls; the pinned build itself is
run through ``DIRECTOR_QUALITY_PYTHON``.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.quality_ratchet_python_support import quality_python
from tools.audit_ci_workflow_modularity import read_workflow
from tools.ci_workflow_inventory import object_mapping
from tools.quality_ratchet_python_engine import (
    ENGINE_FILES,
    PROBE_CODE,
    PYTHON_ARTEFACTS,
    probed_engine,
    python_platform,
    running_engine,
)

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "tools/install_quality_python.sh"
PINNED = PYTHON_ARTEFACTS["linux-x64"]


def probe(python: str) -> subprocess.CompletedProcess[str]:
    """Run the engine probe with an interpreter.

    Parameters
    ----------
    python : str
        The interpreter.

    Returns
    -------
    subprocess.CompletedProcess[str]
        The completed probe.
    """
    return subprocess.run(
        [python, "-I", "-B", "-c", PROBE_CODE],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def reported() -> dict[str, str]:
    """Return the files the pinned interpreter's probe reports.

    Returns
    -------
    dict[str, str]
        Each member of ``ENGINE_FILES`` and its resolved path.
    """
    values = json.loads(probe(quality_python()).stdout)[2:]
    return dict(zip(ENGINE_FILES, values, strict=True))


def report(*values: object) -> subprocess.CompletedProcess[str]:
    """Return a successful probe whose report holds the given values.

    Parameters
    ----------
    *values : object
        The report's members.

    Returns
    -------
    subprocess.CompletedProcess[str]
        The probe.
    """
    return subprocess.CompletedProcess([], 0, json.dumps(list(values)), "")


def test_the_pin_names_one_platform_and_four_digests() -> None:
    """List only linux-x64, with the archive and the three checked files."""
    assert set(PYTHON_ARTEFACTS) == {"linux-x64"}
    assert set(PINNED) == {"archive", *ENGINE_FILES}
    assert ENGINE_FILES == ("binary", "tokenize", "token")
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in PINNED.values())


@pytest.mark.parametrize(
    ("system", "machine", "name"),
    [
        ("linux", "x86_64", "linux-x64"),
        ("linux", "AMD64", "linux-x64"),
        ("darwin", "arm64", "darwin-arm64"),
        ("linux", "aarch64", "linux-arm64"),
        ("linux", "RISCV64", "linux-riscv64"),
    ],
)
def test_platforms_are_named_as_the_pin_keys_them(
    system: str, machine: str, name: str
) -> None:
    """Name known machines by their short form and keep any other as it is."""
    assert python_platform(system, machine) == name


def test_the_pinned_interpreter_passes_both_checks() -> None:
    """Accept the pinned build as the running and as the MyPy interpreter."""
    code = (
        "import json, subprocess, sys\n"
        "from tools.quality_ratchet_python_engine import "
        "PROBE_CODE, probed_engine, running_engine\n"
        "probe = subprocess.run([sys.executable, '-I', '-B', '-c', PROBE_CODE], "
        "capture_output=True, text=True, check=False)\n"
        "print(json.dumps([running_engine(), probed_engine(probe)]))\n"
    )
    result = subprocess.run(
        [quality_python(), "-B", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [["linux-x64", PINNED["binary"]]] * 2
    assert probed_engine(probe(quality_python())) == ("linux-x64", PINNED["binary"])


def test_the_lane_interpreter_is_refused_by_name() -> None:
    """Refuse the lane's own setup-python build in process and as a MyPy probe."""
    with pytest.raises(ValueError, match="The running Python binary differs"):
        running_engine()
    with pytest.raises(ValueError, match="The MyPy Python binary differs"):
        probed_engine(probe(sys.executable))


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (subprocess.CompletedProcess([], 1, "[]", ""), "probe failed"),
        (subprocess.CompletedProcess([], 0, "[]", "warning"), "probe failed"),
        (report("linux", "x86_64"), "invalid identity"),
        (subprocess.CompletedProcess([], 0, '{"a": 1}', ""), "invalid identity"),
        (report("linux", "x86_64", "", "t", "k"), "invalid identity"),
        (report("linux", "x86_64", 1, "t", "k"), "invalid identity"),
        (
            report("darwin", "arm64", "b", "t", "k"),
            "Python platform darwin-arm64 has no verified engine artefact",
        ),
    ],
)
def test_a_failed_malformed_or_unlisted_probe_is_refused(
    result: subprocess.CompletedProcess[str], message: str
) -> None:
    """Refuse a probe that failed, reported badly or names another platform."""
    with pytest.raises(ValueError, match=message):
        probed_engine(result)


@pytest.mark.parametrize("member", ["tokenize", "token"])
def test_a_changed_stdlib_module_is_refused_by_name(
    member: str, tmp_path: Path
) -> None:
    """Refuse the pinned binary beside a tokenize or token that differs by a byte."""
    files = reported()
    changed = tmp_path / f"{member}.py"
    changed.write_bytes(Path(files[member]).read_bytes() + b"\n")
    files[member] = str(changed)
    with pytest.raises(ValueError, match=f"The MyPy Python {member} differs"):
        probed_engine(report("linux", "x86_64", *files.values()))


def test_the_engine_check_has_no_switch() -> None:
    """Read no environment variable and take no argument that could relax it."""
    tree = ast.parse((ROOT / "tools/quality_ratchet_python_engine.py").read_text())
    nodes = list(ast.walk(tree))
    names = {node.attr for node in nodes if isinstance(node, ast.Attribute)}
    names |= {node.id for node in nodes if isinstance(node, ast.Name)}
    imported = {
        alias.name.split(".")[0]
        for node in nodes
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    assert not names & {"environ", "environb", "getenv"}
    assert "os" not in imported
    assert not inspect.signature(running_engine).parameters
    assert list(inspect.signature(probed_engine).parameters) == ["result"]


def test_one_committed_pin_feeds_every_job() -> None:
    """Pin the archive once, in the installer, with the engine's archive digest."""
    text = INSTALLER.read_text()
    assert re.findall(r"^python_sha256=([0-9a-f]{64})$", text, re.M) == [
        PINNED["archive"]
    ]
    assert re.findall(r"^python_url=(\S+)$", text, re.M) == [
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        "20261021/cpython-3.11.17%2B20261021-x86_64-unknown-linux-gnu-"
        "install_only_stripped.tar.gz"
    ]
    assert "sha256sum --check --strict" in text
    assert "--proto '=https' --tlsv1.2" in text and "--fail" in text
    assert "--require-hashes" in text and "--no-cache-dir" in text
    for workflow in (ROOT / ".github/workflows").glob("*.yml"):
        body = workflow.read_text()
        assert "python-build-standalone" not in body
        assert PINNED["archive"] not in body


def steps(path: str, job: str) -> list[dict[str, object]]:
    """Return a workflow job's steps.

    Parameters
    ----------
    path : str
        The workflow, relative to the repository.
    job : str
        The job.

    Returns
    -------
    list[dict[str, object]]
        Its steps, in order.
    """
    jobs = object_mapping(read_workflow(ROOT / path)["jobs"])
    raw = object_mapping(jobs[job])["steps"]
    assert isinstance(raw, list)
    return [object_mapping(step) for step in raw]


INSTALL = "bash tools/install_quality_python.sh"


def test_the_quality_job_runs_only_the_pinned_interpreter() -> None:
    """Install the pin first, with no setup-python or cache, then run the ratchet."""
    job = steps(".github/workflows/ci-quality.yml", "native-quality")
    uses = [str(step.get("uses", "")) for step in job]
    assert not any("setup-python" in use or "actions/cache" in use for use in uses)
    install = [i for i, step in enumerate(job) if step.get("run") == INSTALL]
    ratchet = [
        i for i, step in enumerate(job) if '"$QUALITY_PYTHON"' in str(step.get("run"))
    ]
    assert len(install) == 1 and ratchet and install[0] < min(ratchet)
    assert "if" not in job[install[0]]
    assert not job[install[0]].get("continue-on-error", False)


def test_every_partition_installs_the_pin_before_its_tests() -> None:
    """Give each pytest lane the same pinned interpreter before the tests run."""
    job = steps(".github/workflows/ci-python.yml", "test-partitions")
    install = [i for i, step in enumerate(job) if step.get("run") == INSTALL]
    tests = [
        i
        for i, step in enumerate(job)
        if step.get("name") == "Run the complete assigned test partition"
    ]
    assert len(install) == 1 and len(tests) == 1 and install[0] < tests[0]
    assert "if" not in job[install[0]]
    assert not job[install[0]].get("continue-on-error", False)


def test_the_pinned_files_are_the_ones_this_interpreter_reports() -> None:
    """Hash the pinned build's own files to the three pinned digests."""
    files = reported()
    assert {
        member: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for member, path in files.items()
    } == {member: PINNED[member] for member in ENGINE_FILES}
