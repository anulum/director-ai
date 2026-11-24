# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — the verified Node engine and the typescript-docs canaries
"""Run the real Node, helper and fixture; fault only what a wrapper must fake."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.quality_ratchet_baseline import parse_baseline
from tools.quality_ratchet_canaries import (
    CANARY_ROOT,
    fixture_digest,
    fixture_paths,
    run_profile_canaries,
)
from tools.quality_ratchet_node_engine import NODE_ARTEFACTS, node_engine
from tools.quality_ratchet_succession import FindingKey
from tools.quality_ratchet_typescript import (
    TYPESCRIPT_EXCLUDED,
    TYPESCRIPT_RULES,
    capture_typescript_documentation,
    typescript_sources,
)
from tools.quality_ratchet_typescript_canary import typescript_canary_findings

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = f"{CANARY_ROOT}/typescript-docs/canary.ts"
EXPECTED = (
    FindingKey(FIXTURE, "declaration-documentation", 9, 1, None),
    FindingKey(FIXTURE, "source-directive", 10, 3, None),
)


def real_node() -> str:
    """Return the Node on PATH, which the profile runs.

    Returns
    -------
    str
        Its path.
    """
    node = shutil.which("node")
    assert node is not None
    return node


def wrapper(tmp_path: Path, body: str) -> str:
    """Write an executable shell wrapper standing in for Node.

    Parameters
    ----------
    tmp_path : Path
        Directory for the wrapper.
    body : str
        Shell lines after the interpreter line.

    Returns
    -------
    str
        The wrapper's path.
    """
    path = tmp_path / "node-wrapper"
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def test_the_running_node_is_the_verified_linux_binary() -> None:
    """Probe the real Node: its platform and executed binary match the map."""
    platform, digest = node_engine(real_node(), ROOT, timeout=60.0)
    assert platform == "linux-x64"
    assert digest == NODE_ARTEFACTS["linux-x64"]["binary"]
    executed = Path(real_node()).resolve(strict=True).read_bytes()
    assert hashlib.sha256(executed).hexdigest() == digest


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("exit 3\n", "probe failed"),
        ("echo noise >&2\nprintf '[]'\n", "probe failed"),
        ('printf \'["linux", "x64"]\'\n', "invalid identity"),
        ("printf '{\"a\": 1}'\n", "invalid identity"),
        ('printf \'["linux", "", "/x"]\'\n', "invalid identity"),
        ('printf \'["linux", 64, "/x"]\'\n', "invalid identity"),
        ('printf \'["win32", "x64", "/x"]\'\n', "win32-x64 has no verified"),
    ],
)
def test_a_failed_or_unlisted_probe_is_refused(
    tmp_path: Path, body: str, message: str
) -> None:
    """Refuse a failed probe, a malformed identity, or an unlisted platform."""
    with pytest.raises(ValueError, match=message):
        node_engine(wrapper(tmp_path, body), ROOT, timeout=60.0)


def test_a_binary_other_than_the_verified_one_is_refused(tmp_path: Path) -> None:
    """Refuse a listed platform whose executed binary has another digest."""
    other = tmp_path / "other-node"
    other.write_bytes(b"not the verified release binary\n")
    body = f'printf \'["linux", "x64", "{other}"]\'\n'
    with pytest.raises(ValueError, match="differs from its verified artefact"):
        node_engine(wrapper(tmp_path, body), ROOT, timeout=60.0)


def test_the_probe_runs_in_the_repository_root(tmp_path: Path) -> None:
    """Run the probe where every other native call of the profile runs."""
    log = tmp_path / "probe-cwd.txt"
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ValueError, match="probe failed"):
        node_engine(wrapper(tmp_path, f'pwd > "{log}"\nexit 3\n'), root, timeout=60.0)
    assert log.read_text(encoding="utf-8").strip() == str(root)


def test_the_committed_identity_is_the_capture_of_this_tree() -> None:
    """Pin the whole preimage: the real capture has the committed identity."""
    snapshot = capture_typescript_documentation(ROOT, real_node())
    baseline = parse_baseline(
        (ROOT / "tools/quality_ratchet_baseline.json").read_bytes()
    )
    assert snapshot.identity == baseline.profiles["typescript-docs"].identity


def test_the_committed_fixture_fires_each_rule_once() -> None:
    """Run the real helper on the fixture: one positive for every rule."""
    paths = fixture_paths(ROOT, "typescript-docs")
    assert paths == (FIXTURE,)
    platform, engine, findings = typescript_canary_findings(
        ROOT, real_node(), paths, timeout=120.0
    )
    assert (platform, engine) == ("linux-x64", NODE_ARTEFACTS["linux-x64"]["binary"])
    assert findings == EXPECTED
    assert {key.rule for key in findings} == set(TYPESCRIPT_RULES)


def test_the_registered_runner_reports_the_whole_run() -> None:
    """Run the profile's registered runner as the ratchet does."""
    args = argparse.Namespace(node=real_node(), timeout=120.0)
    run = run_profile_canaries(ROOT, args, "typescript-docs")
    assert run.enabled_rules == TYPESCRIPT_RULES
    assert run.fixture_digest == fixture_digest(ROOT, "typescript-docs")
    assert run.platform == "linux-x64"
    assert run.engine_digest == NODE_ARTEFACTS["linux-x64"]["binary"]
    assert run.findings == EXPECTED


def test_enabled_rules_come_from_the_profile_not_the_findings(tmp_path: Path) -> None:
    """Name every enforced rule even when the fixture fires only one of them."""
    fixture = tmp_path / CANARY_ROOT / "typescript-docs"
    fixture.mkdir(parents=True)
    (fixture / "canary.ts").write_text(
        "export function lone(): number {\n  return 1;\n}\n", encoding="utf-8"
    )
    args = argparse.Namespace(node=real_node(), timeout=120.0)
    run = run_profile_canaries(tmp_path, args, "typescript-docs")
    assert run.enabled_rules == TYPESCRIPT_RULES
    assert {key.rule for key in run.findings} == {"declaration-documentation"}


def faulting_node(tmp_path: Path, fault: str) -> str:
    """Return a Node wrapper that runs the real Node and faults only the analysis.

    Parameters
    ----------
    tmp_path : Path
        Directory for the wrapper.
    fault : str
        The fault applied to the helper's analysis report.

    Returns
    -------
    str
        The wrapper's path.
    """
    path = tmp_path / "node-faulting-analysis"
    path.write_text(
        "#!" + sys.executable + "\n"
        "import json,subprocess,sys\n"
        "args=sys.argv[1:]\n"
        "r=subprocess.run([" + repr(real_node()) + ",*args],"
        "capture_output=True,text=True,timeout=60)\n"
        "out,err,rc=r.stdout,r.stderr,r.returncode\n"
        "if args[0]!='-e' and args[-1]!='--version':\n"
        " d=json.loads(out); fault=" + repr(fault) + "\n"
        " if fault=='report-shape': d['extra']=True\n"
        " if fault=='report-sources': d['sources']={}\n"
        " if fault=='finding-rule': d['findings'][0]['rule']='unknown'\n"
        " if fault=='finding-position': d['findings'][0]['column']=0\n"
        " if fault=='status': rc=0\n"
        " if fault=='exit': rc=2\n"
        " if fault=='stderr': err='noise'\n"
        " out=json.dumps(d)\n"
        "sys.stdout.write(out);sys.stderr.write(err);sys.exit(rc)\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return str(path)


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("report-shape", "not bound to complete source bytes"),
        ("report-sources", "not bound to complete source bytes"),
        ("finding-rule", "Invalid original native TypeScript documentation finding"),
        ("finding-position", "Invalid native diagnostic position"),
        ("status", "status disagrees with its findings"),
        ("exit", "did not completely analyse source"),
        ("stderr", "did not completely analyse source"),
    ],
)
def test_the_canary_refuses_exactly_where_the_adapter_does(
    tmp_path: Path, fault: str, message: str
) -> None:
    """Run the adapter's own analysis: its report checks refuse the canary (A-N1)."""
    with pytest.raises(ValueError, match=message):
        typescript_canary_findings(
            ROOT, faulting_node(tmp_path, fault), (FIXTURE,), timeout=120.0
        )


def test_a_canary_run_on_another_binary_is_refused(tmp_path: Path) -> None:
    """Refuse a canary run whose running process executes another binary."""
    other = tmp_path / "other-node"
    other.write_bytes(b"not the verified release binary\n")
    body = (
        f'if [ "$1" = "-e" ]; then printf \'["linux", "x64", "{other}"]\'; exit 0; fi\n'
        f'exec "{real_node()}" "$@"\n'
    )
    with pytest.raises(ValueError, match="differs from its verified artefact"):
        typescript_canary_findings(
            ROOT, wrapper(tmp_path, body), (FIXTURE,), timeout=120.0
        )


def git(root: Path, *arguments: str) -> None:
    """Run Git in a test repository.

    Parameters
    ----------
    root : Path
        Test repository.
    *arguments : str
        Git arguments.
    """
    subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)


def test_the_inventory_skips_the_canary_root_and_nothing_else(tmp_path: Path) -> None:
    """Skip a TypeScript file under the canary root; refuse one elsewhere."""
    git(tmp_path, "init")
    (tmp_path / "packages").mkdir()
    (tmp_path / "packages" / "a.ts").write_text("export {};\n", encoding="utf-8")
    canary = tmp_path / TYPESCRIPT_EXCLUDED[0] / "typescript-docs"
    canary.mkdir(parents=True)
    (canary / "canary.ts").write_text("export {};\n", encoding="utf-8")
    assert typescript_sources(tmp_path) == ("packages/a.ts",)
    (tmp_path / "tests" / "stray.ts").write_text("export {};\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the declared quality scope"):
        typescript_sources(tmp_path)


def test_no_profile_measures_a_canary_fixture_file() -> None:
    """Every fixture file is absent from all nine committed profiles."""
    fixtures = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / CANARY_ROOT).rglob("*")
        if path.is_file()
    }
    assert FIXTURE in fixtures
    baseline = parse_baseline(
        (ROOT / "tools/quality_ratchet_baseline.json").read_bytes()
    )
    assert len(baseline.profiles) == 9
    for snapshot in baseline.profiles.values():
        assert not fixtures & set(snapshot.sources)
    assert TYPESCRIPT_EXCLUDED == (CANARY_ROOT + "/",)
