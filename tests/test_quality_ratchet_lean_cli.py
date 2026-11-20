# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — shared Lean quality CLI integration
"""Exercise the maintained shared quality CLI with the actual Lean compiler."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from formal.HaltMonitor.tests.test_quality_ratchet_lean import (
    lean_binary as lean_binary,
)
from formal.HaltMonitor.tests.test_quality_ratchet_lean import (
    lean_repository as lean_repository,
)
from tools.quality_ratchet_contracts import decode_quality_json, parse_snapshot


def test_public_cli_compiles_real_unchanged_dependencies(
    lean_repository: Path,
    lean_binary: str,
    tmp_path: Path,
) -> None:
    """Reach complete native capture through the public CLI and original imports.

    Parameters
    ----------
    lean_repository : Path
        Original native Git source root.
    lean_binary : str
        Actual pinned compiler executable.
    tmp_path : Path
        Actual runner output custody.
    """
    project = lean_repository / "formal/HaltMonitor"
    (project / "HaltMonitor").mkdir()
    (project / "HaltMonitor/Value.lean").write_text(
        "/-! Original dependency. -/\n/-- Dependency value. -/\ndef dependencyValue : Nat := 1\n"
    )
    path = project / "HaltMonitor.lean"
    path.write_text(
        "import HaltMonitor.Value\n/-! Original admission library. -/\n/-- Original dependent value. -/\ndef value : Nat := dependencyValue\n"
    )
    subprocess.run(
        ["git", "add", "."], cwd=lean_repository, capture_output=True, check=True
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
            "original native sources",
        ],
        cwd=lean_repository,
        capture_output=True,
        check=True,
    )
    before = path.read_bytes()
    output = tmp_path / "original-snapshot.json"
    environment = dict(
        os.environ,
        LEAN_PATH=str(tmp_path / "unavailable-caller-cache"),
        LEAN_SYSROOT=str(tmp_path / "unavailable-caller-toolchain"),
    )
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "tools.check_quality_ratchet",
            "capture",
            "--root",
            str(lean_repository),
            "--profile",
            "lean-docs",
            "--lean",
            lean_binary,
            "--timeout",
            "180",
            "--output",
            str(output),
        ],
        capture_output=True,
        env=environment,
        timeout=210,
    )
    assert process.returncode == 0, process.stderr.decode()
    snapshot = parse_snapshot(decode_quality_json(output.read_bytes()))
    assert (
        snapshot.profile == "lean-docs"
        and not snapshot.findings
        and not snapshot.advisories
    )
    assert set(snapshot.sources) == {
        "formal/HaltMonitor/HaltMonitor.lean",
        "formal/HaltMonitor/HaltMonitor/Value.lean",
    }
    assert path.read_bytes() == before
    assert not (project / ".lake").exists()
    assert not list(project.rglob("*.olean"))
    baseline = lean_repository / "tools/quality_ratchet_baseline.json"
    baseline.parent.mkdir()
    for command in ("record", "check"):
        arguments = [
            sys.executable,
            "-B",
            "-m",
            "tools.check_quality_ratchet",
            command,
            "--root",
            str(lean_repository),
            "--profile",
            "lean-docs",
            "--lean",
            lean_binary,
            "--timeout",
            "180",
        ]
        if command == "record":
            arguments.extend(["--output", str(baseline)])
        result = subprocess.run(
            arguments, capture_output=True, env=environment, timeout=210
        )
        assert result.returncode == 0, result.stderr.decode()
    assert baseline.is_file()
    assert path.read_bytes() == before
