# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — actual native Lean ratchet source and CLI integration tests
"""Exercise isolated original import graphs and complete public Lean capture."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.quality_ratchet_comparison import compare_snapshots
from tools.quality_ratchet_lean import capture_lean_documentation, lean_sources


@pytest.fixture
def lean_binary() -> str:
    """Require a real installed native tool without skipping missing prerequisites.

    Returns
    -------
    str
        Actual pinned compiler executable selected by the native runner.
    """
    binary = os.environ.get("DIRECTOR_QUALITY_LEAN") or shutil.which("lean")
    assert binary is not None, "Native Lean is required"
    return binary


@pytest.fixture
def lean_repository(tmp_path: Path) -> Path:
    """Create a real Git source repository with the qualified library recipe.

    Parameters
    ----------
    tmp_path : Path
        Actual isolated runner workspace on the configured work disk.

    Returns
    -------
    Path
        Complete original source and native project configuration.
    """
    root = tmp_path / "lean-repository"
    project = root / "formal/HaltMonitor"
    project.mkdir(parents=True)
    (project / "lakefile.toml").write_text(
        'name = "HaltMonitor"\ndefaultTargets = ["HaltMonitor"]\n\n[[lean_lib]]\nname = "HaltMonitor"\n'
    )
    (project / "lean-toolchain").write_text("leanprover/lean4:v4.34.1\n")
    (project / "HaltMonitor.lean").write_text(
        "/-! Original admission library. -/\n/-- Original value. -/\ndef value : Nat := 1\n"
    )
    subprocess.run(["git", "init"], cwd=root, capture_output=True, check=True)
    return root


@pytest.mark.parametrize(
    "mode",
    [
        "unbound",
        "cycle",
        "elaboration",
        "toolchain",
        "project",
        "identity",
        "ambiguous",
    ],
)
def test_native_repository_refuses_unqualified_sources(
    lean_repository: Path,
    lean_binary: str,
    mode: str,
) -> None:
    """Refuse real import failures, changed recipes and ambiguous source modules.

    Parameters
    ----------
    lean_repository : Path
        Real original source and Git inventory.
    lean_binary : str
        Required native compiler.
    mode : str
        Concrete source or recipe failure applied to this isolated repository.
    """
    project = lean_repository / "formal/HaltMonitor"
    if mode == "unbound":
        (project / "HaltMonitor.lean").write_text("import AbsentOriginalModule\n")
    elif mode == "cycle":
        (project / "HaltMonitor").mkdir()
        (project / "HaltMonitor/Other.lean").write_text("import HaltMonitor\n")
        (project / "HaltMonitor.lean").write_text("import HaltMonitor.Other\n")
    elif mode == "elaboration":
        (project / "HaltMonitor.lean").write_text(
            "def invalid : Nat := originalNameDoesNotExist\n"
        )
    elif mode == "toolchain":
        (project / "lean-toolchain").write_text("leanprover/lean4:v0.0.0\n")
    elif mode == "project":
        with (project / "lakefile.toml").open("a") as stream:
            stream.write('\n[[require]]\nname = "unbound-package"\n')
    elif mode == "identity":
        (project / "bad-name.lean").write_text("/-! Invalid source module. -/\n")
    else:
        (project / "QualityRatchet.lean").write_text("/-! First native module. -/\n")
        helper = lean_repository / "tools/quality_ratchet_lean"
        helper.mkdir(parents=True)
        (helper / "QualityRatchet.lean").write_text("/-! Ambiguous native module. -/\n")
    with pytest.raises(ValueError):
        capture_lean_documentation(lean_repository, lean_binary, timeout=180)


@pytest.mark.parametrize("mode", ["outside", "symlink", "empty", "subroot"])
def test_native_inventory_keeps_every_source_boundary(
    lean_repository: Path,
    tmp_path: Path,
    mode: str,
) -> None:
    """Refuse undeclared, indirect, absent and incorrectly rooted source inventories.

    Parameters
    ----------
    lean_repository : Path
        Actual complete native Git repository.
    tmp_path : Path
        Actual isolated runner workspace.
    mode : str
        Concrete source ownership boundary violation.
    """
    path = lean_repository / "formal/HaltMonitor/HaltMonitor.lean"
    root = lean_repository
    if mode == "outside":
        (root / "outside.lean").write_text("def outside : Nat := 1\n")
    elif mode == "symlink":
        target = tmp_path / "external.lean"
        target.write_text("def external : Nat := 1\n")
        path.unlink()
        path.symlink_to(target)
    elif mode == "empty":
        path.unlink()
    else:
        root = root / "formal"
    with pytest.raises(ValueError):
        lean_sources(root)


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf"), 1e-323])
def test_native_deadline_is_positive_finite_and_executable(
    lean_repository: Path,
    lean_binary: str,
    timeout: float,
) -> None:
    """Refuse invalid and already exhausted actual analysis deadlines.

    Parameters
    ----------
    lean_repository : Path
        Real maintained source repository.
    lean_binary : str
        Actual pinned compiler executable.
    timeout : float
        Invalid or physically exhausted whole native deadline.
    """
    with pytest.raises(ValueError, match="deadline"):
        capture_lean_documentation(lean_repository, lean_binary, timeout=timeout)


def test_missing_native_tool_is_visible(lean_repository: Path) -> None:
    """Refuse a missing native tool instead of returning a clean source cohort.

    Parameters
    ----------
    lean_repository : Path
        Actual maintained source root.
    """
    with pytest.raises(ValueError, match="executable is unavailable"):
        capture_lean_documentation(lean_repository, "director-native-lean-missing")


def test_changed_source_cannot_introduce_a_real_compiler_warning(
    lean_repository: Path,
    lean_binary: str,
) -> None:
    """Require zero new native warnings even when source docs are complete.

    Parameters
    ----------
    lean_repository : Path
        Actual original native Git source repository.
    lean_binary : str
        Actual pinned compiler executable.
    """
    original = capture_lean_documentation(lean_repository, lean_binary)
    path = lean_repository / "formal/HaltMonitor/HaltMonitor.lean"
    with path.open("a") as stream:
        stream.write(
            "set_option linter.unusedVariables true\n/-- Original warned value. -/\ndef warned (unused : Nat) : Nat := 1\n"
        )
    current = capture_lean_documentation(lean_repository, lean_binary)
    assert len(current.findings) == 1
    assert current.findings[0].rule == "compiler-warning"
    assert current.findings[0].path == "formal/HaltMonitor/HaltMonitor.lean"
    assert current.advisories
    assert compare_snapshots(original, current) == (
        "New or modified source has native findings: formal/HaltMonitor/HaltMonitor.lean",
    )


@pytest.mark.parametrize(
    "attack",
    [
        "version",
        "header",
        "header-import-array",
        "header-import-identity",
        "inspection",
        "warning-severity",
        "warning-source",
        "warning-duplicate",
        "information",
        "binary-drift",
    ],
)
def test_public_native_transport_refuses_lost_qualification(
    lean_repository: Path,
    lean_binary: str,
    tmp_path: Path,
    attack: str,
) -> None:
    """Retain actual native output before deliberate transport and tool corruption.

    Parameters
    ----------
    lean_repository : Path
        Original maintained native source repository.
    lean_binary : str
        Actual pinned compiler executable.
    tmp_path : Path
        Original process output custody and isolated proxy tool.
    attack : str
        Concrete transport or source-identity failure introduced after execution.
    """
    path = lean_repository / "formal/HaltMonitor/HaltMonitor.lean"
    if attack.startswith("warning-"):
        with path.open("a") as stream:
            stream.write(
                "set_option linter.unusedVariables true\n/-- Original warned value. -/\ndef warned (unused : Nat) : Nat := 1\n"
            )
    elif attack == "information":
        with path.open("a") as stream:
            stream.write("#check Nat\n")
    native = shutil.which(lean_binary)
    assert native is not None
    custody = tmp_path / "native-originals.jsonl"
    wrapper = tmp_path / "native-lean-original-before-corruption"
    wrapper.write_text(
        "#!" + sys.executable + " -B\n"
        '"""Retain actual native output before a deliberate admission attack."""\n'
        "import sys,subprocess,json,base64\nfrom pathlib import Path\n"
        "args=sys.argv[1:]; attack=" + repr(attack) + "\n"
        "result=subprocess.run(["
        + repr(native)
        + ",*args],capture_output=True,timeout=120)\n"
        "with Path(" + repr(str(custody)) + ").open('a') as stream:\n"
        " stream.write(json.dumps({'argv':args,'code':result.returncode,'stdout_b64':base64.b64encode(result.stdout).decode(),'stderr_b64':base64.b64encode(result.stderr).decode()})+'\\n')\n"
        "output=result.stdout; error=result.stderr\n"
        "if attack=='version' and args==['--version']: output=b'unqualified version\\n'\n"
        "if '--run' in args and 'header' in args and attack.startswith('header'):\n"
        " row=json.loads(output)\n"
        " if attack=='header': row['source']='lost original source'\n"
        " elif attack=='header-import-array': row['imports']={}\n"
        " else: row['imports']=['../unbound-module']\n"
        " output=json.dumps(row).encode()\n"
        "if attack=='inspection' and '--run' in args and 'inspect' in args: error=b'original inspection transport lost'\n"
        "if attack.startswith('warning-') and '-o' in args:\n"
        " rows=[json.loads(line) for line in output.splitlines()]\n"
        " for row in rows:\n"
        "  if attack=='warning-severity': row['severity']='error'\n"
        "  elif attack=='warning-source': row['fileName']='unbound-source.lean'\n"
        " if attack=='warning-duplicate': rows=rows+rows\n"
        " output=('\\n'.join(json.dumps(row) for row in rows)+'\\n').encode()\n"
        "if attack=='binary-drift' and args==['--version']:\n"
        " with Path(__file__).open('a') as stream: stream.write('\\n# Actual tool drift.\\n')\n"
        "sys.stdout.buffer.write(output);sys.stderr.buffer.write(error)\n"
        "raise SystemExit(result.returncode)\n"
    )
    wrapper.chmod(0o700)
    if attack == "warning-duplicate":
        snapshot = capture_lean_documentation(
            lean_repository, str(wrapper), timeout=180
        )
        assert (
            len(snapshot.findings) == 1
            and snapshot.findings[0].rule == "compiler-warning"
        )
        assert snapshot.advisories
    else:
        with pytest.raises(ValueError):
            capture_lean_documentation(lean_repository, str(wrapper), timeout=180)
    records = [json.loads(line) for line in custody.read_text().splitlines()]
    assert records and all(row["code"] == 0 for row in records)
    assert all(row["stdout_b64"] or row["argv"][0] == "--json" for row in records)


def test_deleted_tracked_source_is_absent_from_the_live_inventory(
    lean_repository: Path,
) -> None:
    """Measure actual file deletion without retaining a nonexistent source entry.

    Parameters
    ----------
    lean_repository : Path
        Real Git repository with independently tracked source entries.
    """
    subprocess.run(
        ["git", "add", "."], cwd=lean_repository, check=True, capture_output=True
    )
    project = lean_repository / "formal/HaltMonitor"
    (project / "HaltMonitor.lean").unlink()
    (project / "Other.lean").write_text("/-! Surviving original source. -/\n")
    assert lean_sources(lean_repository) == ("formal/HaltMonitor/Other.lean",)
