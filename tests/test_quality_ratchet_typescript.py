# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — real native TypeScript snapshot and ratchet contracts
"""Exercise original compiler processes and immutable source-linked allowances."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.quality_ratchet_comparison import compare_snapshots
from tools.quality_ratchet_typescript import (
    capture_typescript_documentation,
    typescript_sources,
)


@pytest.fixture
def typescript_repository(tmp_path: Path) -> Path:
    """Create a real maintained Git source root with an original declaration.

    Parameters
    ----------
    tmp_path : Path
        Actual isolated native runner's Samsung directory.

    Returns
    -------
    Path
        Original committed source repository.
    """
    root = tmp_path / "typescript-source"
    root.mkdir()
    subprocess.run(["git", "init"], cwd=root, capture_output=True, check=True)
    package = root / "packages/contract"
    package.mkdir(parents=True)
    (package / "value.ts").write_text(
        "/** Return the original admission value. */\n"
        "export function value(): number { return 1; }\n"
    )
    subprocess.run(
        ["git", "add", "packages/contract/value.ts"],
        cwd=root,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Native fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-m",
            "original source",
        ],
        cwd=root,
        capture_output=True,
        check=True,
    )
    return root


def test_native_typescript_changed_whole_file_cannot_retain_legacy_findings(
    typescript_repository: Path,
) -> None:
    """Reject changed-file historical debt instead of trading diagnostic counts.

    Parameters
    ----------
    typescript_repository : Path
        Real original maintained source repository.
    """
    root = typescript_repository
    path = root / "packages/contract/value.ts"
    path.write_text("export function value(): number { return 1; }\n")
    original = capture_typescript_documentation(root, "node")
    assert len(original.findings) == 1
    path.write_text("export function value(): number { return 2; }\n")
    changed = capture_typescript_documentation(root, "node")
    assert compare_snapshots(original, changed)
    path.write_text(
        "/** Return the corrected admission value. */\n"
        "export function value(): number { return 2; }\n"
    )
    clean = capture_typescript_documentation(root, "node")
    assert not compare_snapshots(original, clean)
    assert not clean.findings


def test_native_typescript_syntax_and_missing_tool_refuse(
    typescript_repository: Path,
) -> None:
    """Preserve actual native syntax failure and missing-process refusal.

    Parameters
    ----------
    typescript_repository : Path
        Real native source repository.
    """
    with pytest.raises(FileNotFoundError):
        capture_typescript_documentation(typescript_repository, "/absent/node")
    with pytest.raises((ValueError, subprocess.TimeoutExpired)):
        capture_typescript_documentation(
            typescript_repository,
            "node",
            timeout=0.0000001,
        )
    (typescript_repository / "packages/contract/value.ts").write_text(
        "export function invalid( {\n"
    )
    with pytest.raises(ValueError, match="completely analyse"):
        capture_typescript_documentation(typescript_repository, "node")


@pytest.mark.parametrize("timeout", (0.0, -1.0, float("nan"), float("inf")))
def test_typescript_requires_finite_positive_deadline(
    typescript_repository: Path,
    timeout: float,
) -> None:
    """Refuse an invalid analysis lifetime before any native execution.

    Parameters
    ----------
    typescript_repository : Path
        Actual selected source repository.
    timeout : float
        Deliberately invalid deadline.
    """
    with pytest.raises(ValueError, match="positive and finite"):
        capture_typescript_documentation(typescript_repository, "node", timeout=timeout)


def test_native_typescript_inventory_keeps_regular_complete_declared_sources(
    typescript_repository: Path,
) -> None:
    """Refuse aliases and unknown scopes while preserving actual pending deletions.

    Parameters
    ----------
    typescript_repository : Path
        Actual Git source repository.
    """
    root = typescript_repository
    relative = "packages/contract/value.ts"
    assert typescript_sources(root) == (relative,)
    (root / "packages/contract/alias.ts").symlink_to("value.ts")
    with pytest.raises(ValueError, match="Symbolic"):
        typescript_sources(root)
    (root / "packages/contract/alias.ts").unlink()
    (root / "unclassified.ts").write_text("const value = 1;")
    with pytest.raises(ValueError, match="outside"):
        typescript_sources(root)
    (root / "unclassified.ts").unlink()
    with pytest.raises(ValueError, match="actual Git root"):
        typescript_sources(root / "packages")
    (root / relative).unlink()
    with pytest.raises(ValueError, match="inventory is empty"):
        typescript_sources(root)


@pytest.mark.parametrize(
    "fault",
    (
        "node-version",
        "metadata-version",
        "metadata-shape",
        "metadata-json",
        "metadata-flag",
        "stderr",
        "exit",
        "report-shape",
        "report-sources",
        "report-complete",
        "report-json",
        "finding-members",
        "finding-rule",
        "finding-position",
        "status",
        "source-change",
    ),
)
def test_native_typescript_faults_refuse_after_original_real_execution(
    typescript_repository: Path,
    tmp_path: Path,
    fault: str,
) -> None:
    """Keep real process output before deliberate refusal-only carrier faults.

    Parameters
    ----------
    typescript_repository : Path
        Actual Git source submitted to the original compiler.
    tmp_path : Path
        Actual runner output and wrapper custody.
    fault : str
        Explicit negative corruption; never a successful native substitute.
    """
    node = shutil.which("node")
    assert node is not None
    root = typescript_repository
    (root / "packages/contract/value.ts").write_text(
        "export function value(): number { return 1; }\n"
    )
    wrapper = tmp_path / "node-original-before-fault"
    wrapper.write_text(
        "#!" + sys.executable + "\n"
        "import json,sys,subprocess\nfrom pathlib import Path\n"
        "args=sys.argv[1:]\n"
        "result=subprocess.run(["
        + repr(node)
        + ",*args],capture_output=True,text=True,timeout=20)\n"
        "with Path('native-original.jsonl').open('a') as stream:\n"
        " stream.write(json.dumps({'argv':args,'status':result.returncode,"
        "'stdout':result.stdout,'stderr':result.stderr})+'\\n')\n"
        "fault=" + repr(fault) + "\n"
        "output=result.stdout; error=result.stderr; status=result.returncode\n"
        "if args==['--version']:\n"
        " if fault=='node-version': output='v0.0.0\\n'\n"
        "elif args[-1]=='--version':\n"
        " data=json.loads(output)\n"
        " if fault=='metadata-version': data['typescript']='0.0.0'\n"
        " if fault=='metadata-shape': data['extra']=True\n"
        " if fault=='metadata-flag': data['compilerOptions']['noLib']=1\n"
        " output='{' if fault=='metadata-json' else json.dumps(data)\n"
        "else:\n"
        " data=json.loads(output); assert data['findings']\n"
        " if fault=='stderr': error='actual refusal fault\\n'\n"
        " if fault=='exit': status=2\n"
        " if fault=='report-shape': data['extra']=True\n"
        " if fault=='report-sources': data['sources']={}\n"
        " if fault=='report-complete': data['complete']=False\n"
        " if fault=='finding-members': del data['findings'][0]['message']\n"
        " if fault=='finding-rule': data['findings'][0]['rule']='unknown'\n"
        " if fault=='finding-position': data['findings'][0]['column']=0\n"
        " if fault=='status': status=0\n"
        " if fault=='source-change': Path('packages/contract/value.ts').write_text('export const changed = 2;')\n"
        " output='{' if fault=='report-json' else json.dumps(data)\n"
        "sys.stdout.write(output);sys.stderr.write(error);sys.exit(status)\n"
    )
    wrapper.chmod(0o700)
    with pytest.raises(ValueError):
        capture_typescript_documentation(root, str(wrapper), timeout=45)
    originals = [
        json.loads(line)
        for line in (root / "native-original.jsonl").read_text().splitlines()
    ]
    assert originals
    assert all(item["status"] in (0, 1) and not item["stderr"] for item in originals)
    assert all(item["stdout"] for item in originals)
