# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — native TypeScript source, engine and deadline integrity
"""Bind real compiler results to complete sources, executing recipes and deadlines."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_quality_ratchet_typescript import (
    typescript_repository as typescript_repository,
)
from tools.ci_workflow_inventory import ROOT
from tools.quality_ratchet_contracts import snapshot_payload
from tools.quality_ratchet_typescript import (
    TYPESCRIPT_ENGINE_INPUTS,
    TYPESCRIPT_SUFFIXES,
    capture_typescript_documentation,
    typescript_sources,
)


def test_native_typescript_captures_the_actual_complete_checkout(
    tmp_path: Path,
) -> None:
    """Compare actual maintained compiler inputs with independent native Git paths.

    Parameters
    ----------
    tmp_path : Path
        Custody for the original full-cohort snapshot, including historical debt.
    """
    native = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            *("*" + suffix for suffix in sorted(TYPESCRIPT_SUFFIXES)),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=20,
    )
    paths = sorted({raw.decode() for raw in native.stdout.split(b"\0") if raw})
    expected = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths
    }
    snapshot = capture_typescript_documentation(ROOT, "node")
    assert dict(snapshot.sources) == expected
    assert snapshot.profile == "typescript-docs"
    assert snapshot.complete
    (tmp_path / "original-full-cohort.json").write_text(
        json.dumps(snapshot_payload(snapshot), sort_keys=True)
    )


@pytest.mark.parametrize("deadline", (float("nan"), float("inf"), -float("inf")))
def test_inventory_rejects_nonfinite_absolute_deadlines(
    typescript_repository: Path,
    deadline: float,
) -> None:
    """Reject an unbounded caller lifetime rather than passing it to native Git.

    Parameters
    ----------
    typescript_repository : Path
        Real committed source whose inventory would otherwise succeed.
    deadline : float
        Invalid absolute monotonic deadline.
    """
    with pytest.raises(ValueError, match="must be finite"):
        typescript_sources(typescript_repository, deadline=deadline)


def test_inventory_preserves_unrelated_names_and_refuses_broken_source_ownership(
    typescript_repository: Path,
) -> None:
    """Keep unrelated byte names while refusing undecodable or aliased TS inputs.

    Parameters
    ----------
    typescript_repository : Path
        Real Git inventory with committed and deliberately adverse pending inputs.
    """
    root = typescript_repository
    raw = os.fsencode(root) + b"/unrelated-\xff.dat"
    fd = os.open(raw, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    assert typescript_sources(root) == ("packages/contract/value.ts",)
    invalid = os.fsencode(root / "packages/contract") + b"/source-\xff.ts"
    fd = os.open(invalid, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    with pytest.raises(UnicodeDecodeError):
        typescript_sources(root)
    os.unlink(invalid)
    source = root / "packages/contract/value.ts"
    source.unlink()
    source.mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        typescript_sources(root)
    source.rmdir()
    source.parent.rmdir()
    source.parent.symlink_to(root / "absent-source-owner", target_is_directory=True)
    with pytest.raises(ValueError, match="Symbolic"):
        typescript_sources(root)


def test_native_engine_identity_and_concurrent_recipe_drift(
    typescript_repository: Path,
    tmp_path: Path,
) -> None:
    """Require identical recipes to agree and changed executing engines to refuse.

    Parameters
    ----------
    typescript_repository : Path
        Actual original Git source submitted to every real compiler execution.
    tmp_path : Path
        Isolated exact seven-input engine fixture and native-output custody.
    """
    recipe = tmp_path / "engine-recipe"
    for relative in TYPESCRIPT_ENGINE_INPUTS:
        target = recipe / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    (recipe / "packages/vercel-ai/node_modules").symlink_to(
        (ROOT / "packages/vercel-ai/node_modules").resolve(),
        target_is_directory=True,
    )
    program = (
        "import json,sys\nfrom pathlib import Path\n"
        "from tools.quality_ratchet_typescript import capture_typescript_documentation\n"
        "result=capture_typescript_documentation(Path(sys.argv[1]),sys.argv[2])\n"
        "print(json.dumps({'identity':result.identity,'sources':dict(result.sources)}))\n"
    )
    command = [sys.executable, "-B", "-c", program, str(typescript_repository)]
    original = capture_typescript_documentation(typescript_repository, "node")
    identical = subprocess.run(
        [*command, "node"],
        cwd=recipe,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert json.loads(identical.stdout)["identity"] == original.identity
    adapter = recipe / "tools/quality_ratchet_typescript.py"
    adapter.write_text(adapter.read_text() + "\n")
    changed = subprocess.run(
        [*command, "node"],
        cwd=recipe,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert json.loads(changed.stdout)["identity"] != original.identity
    assert json.loads(changed.stdout)["sources"] == dict(original.sources)
    node = shutil.which("node")
    assert node is not None
    wrapper = tmp_path / "node-original-before-engine-change"
    original_output = tmp_path / "original-engine-native.jsonl"
    wrapper.write_text(
        "#!"
        + sys.executable
        + "\nimport json,sys,subprocess\nfrom pathlib import Path\n"
        "args=sys.argv[1:]\nresult=subprocess.run(["
        + repr(node)
        + ",*args],capture_output=True,text=True,timeout=20)\n"
        "with Path(" + repr(str(original_output)) + ").open('a') as stream:\n"
        " stream.write(json.dumps({'args':args,'status':result.returncode,"
        "'stdout':result.stdout,'stderr':result.stderr})+'\\n')\n"
        "if args[-1]!='--version':\n"
        " path=Path(" + repr(str(adapter)) + ")\n"
        " path.write_text(path.read_text()+'\\n')\n"
        "sys.stdout.write(result.stdout);sys.stderr.write(result.stderr)\n"
        "sys.exit(result.returncode)\n"
    )
    wrapper.chmod(0o700)
    rejected = subprocess.run(
        [*command, str(wrapper)],
        cwd=recipe,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert rejected.returncode != 0
    assert "source or engine changed" in rejected.stderr
    originals = [json.loads(line) for line in original_output.read_text().splitlines()]
    assert len(originals) == 3
    assert originals[-1]["status"] == 0
    assert json.loads(originals[-1]["stdout"])["complete"] is True
    assert all(not item["stderr"] for item in originals)


def test_final_git_revalidation_cannot_outlive_the_shared_native_deadline(
    typescript_repository: Path,
    tmp_path: Path,
) -> None:
    """Reject delayed final Git custody after successful actual native analysis.

    Parameters
    ----------
    typescript_repository : Path
        Original real Git source retained throughout the test.
    tmp_path : Path
        Isolated wrappers, measured budget and original native carrier custody.
    """
    node = shutil.which("node")
    git = shutil.which("git")
    assert node is not None and git is not None
    binary = tmp_path / "bin"
    binary.mkdir()
    marker = tmp_path / "native-analysis-completed"
    delay = tmp_path / "measured-delay.json"
    original_git = tmp_path / "original-final-git.json"
    original_node = tmp_path / "original-final-node.json"
    node_wrapper = binary / "node-after-native-analysis"
    node_wrapper.write_text(
        "#!"
        + sys.executable
        + "\nimport json,subprocess,sys\nfrom pathlib import Path\n"
        "args=sys.argv[1:]\nr=subprocess.run(["
        + repr(node)
        + ",*args],capture_output=True,text=True,timeout=20)\n"
        "if args[-1]!='--version':\n"
        " Path(" + repr(str(original_node)) + ").write_text(r.stdout)\n"
        " Path(" + repr(str(marker)) + ").touch()\n"
        "sys.stdout.write(r.stdout);sys.stderr.write(r.stderr);sys.exit(r.returncode)\n"
    )
    git_wrapper = binary / "git"
    git_wrapper.write_text(
        "#!"
        + sys.executable
        + "\nimport json,subprocess,sys,time\nfrom pathlib import Path\n"
        "r=subprocess.run(["
        + repr(git)
        + ",*sys.argv[1:]],capture_output=True,text=True,timeout=20)\n"
        "if Path(" + repr(str(marker)) + ").exists():\n"
        " Path(" + repr(str(original_git)) + ").write_text(json.dumps({"
        "'status':r.returncode,'stdout':r.stdout,'stderr':r.stderr}))\n"
        " time.sleep(json.loads(Path(" + repr(str(delay)) + ").read_text())['delay'])\n"
        "sys.stdout.write(r.stdout);sys.stderr.write(r.stderr);sys.exit(r.returncode)\n"
    )
    node_wrapper.chmod(0o700)
    git_wrapper.chmod(0o700)
    program = (
        "import json,subprocess,sys,time\nfrom pathlib import Path\n"
        "from tools.quality_ratchet_typescript import capture_typescript_documentation\n"
        "root=Path(sys.argv[1]);start=time.monotonic()\n"
        "baseline=capture_typescript_documentation(root,'node')\n"
        "budget=(time.monotonic()-start)*8\n"
        "Path(sys.argv[3]).write_text(json.dumps({'delay':budget*4}))\n"
        "start=time.monotonic()\n"
        "try: capture_typescript_documentation(root,sys.argv[2],timeout=budget)\n"
        "except (ValueError,subprocess.TimeoutExpired) as exc:\n"
        " print(json.dumps({'refused':True,'error':str(exc),'budget':budget,"
        "'elapsed':time.monotonic()-start}));sys.exit(0)\n"
        "raise AssertionError('Accepted custody after its whole deadline')\n"
    )
    environment = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"])
    refused = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            program,
            str(typescript_repository),
            str(node_wrapper),
            str(delay),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=45,
    )
    assert refused.returncode == 0, refused.stderr
    result = json.loads(refused.stdout)
    assert result["refused"] is True
    assert result["elapsed"] <= result["budget"] + result["budget"] / 8
    assert result["elapsed"] < json.loads(delay.read_text())["delay"]
    assert json.loads(original_node.read_text())["complete"] is True
    actual_git = json.loads(original_git.read_text())
    assert actual_git["status"] == 0 and not actual_git["stderr"]
    assert Path(actual_git["stdout"].strip()).resolve() == typescript_repository
    (tmp_path / "original-deadline-result.json").write_text(refused.stdout)
