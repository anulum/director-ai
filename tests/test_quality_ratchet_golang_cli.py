# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — real native Go documentation ratchet contracts
"""Preserve native Go baseline adoption through the complete quality CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.quality_ratchet_contracts import decode_quality_json, parse_snapshot


@pytest.fixture
def go_tools() -> tuple[str, str]:
    """Resolve required real native tools without skipping missing prerequisites.

    Returns
    -------
    tuple[str, str]
        Actual Go and Revive executables, verified by each native capture.
    """
    go = shutil.which("go")
    revive = os.environ.get("DIRECTOR_QUALITY_REVIVE") or shutil.which("revive")
    assert go is not None and revive is not None, "Go and Revive are required"
    return go, revive


@pytest.fixture
def go_repository(tmp_path: Path) -> Path:
    """Create a real Git repository with a documented private Go declaration.

    Parameters
    ----------
    tmp_path : Path
        Actual native runner's isolated Samsung workspace.

    Returns
    -------
    Path
        Real maintained source, native module metadata and original Git commit.
    """
    root = tmp_path / "go-quality"
    root.mkdir()
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    module = root / "gateway/go"
    module.mkdir(parents=True)
    (module / "go.mod").write_text("module native-fixture\n\ngo 1.27.1\n")
    (module / "go.sum").write_text("")
    (module / "value.go").write_text(
        "// Package fixture provides an actual value.\npackage fixture\n"
        "// value returns the original count.\nfunc value() int {return 1}\n"
    )
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
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
        cwd=root,
        check=True,
        capture_output=True,
    )
    return root


def test_public_cli_records_original_go_baseline_and_refuses_regression(
    go_repository: Path,
    go_tools: tuple[str, str],
    tmp_path: Path,
) -> None:
    """Exercise the real public record/check entrypoints and immutable Git baseline.

    Parameters
    ----------
    go_repository : Path
        Actual original Git source repository.
    go_tools : tuple[str, str]
        Verified native Go and Revive.
    tmp_path : Path
        Actual runner's isolated evidence directory.
    """
    command = [
        sys.executable,
        "-B",
        "-m",
        "tools.check_quality_ratchet",
        "--root",
        str(go_repository),
        "--profile",
        "go-docs",
        "--go",
        go_tools[0],
        "--revive",
        go_tools[1],
    ]
    baseline = tmp_path / "original.json"
    original = subprocess.run(
        [*command, "record", "--output", str(baseline)],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert original.returncode == 0, original.stderr
    assert json.loads(original.stdout)["profile"] == "go-docs"
    installed = go_repository / "tools/quality_ratchet_baseline.json"
    installed.parent.mkdir()
    installed.write_bytes(baseline.read_bytes())
    subprocess.run(
        ["git", "add", "."], cwd=go_repository, check=True, capture_output=True
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
            "adopt original source baseline",
        ],
        cwd=go_repository,
        check=True,
        capture_output=True,
    )
    output = tmp_path / "actual.json"
    captured = subprocess.run(
        [*command, "capture", "--output", str(output)],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert captured.returncode == 0, captured.stderr
    assert parse_snapshot(decode_quality_json(output.read_bytes())).findings == ()
    (go_repository / "gateway/go/value.go").write_text(
        "package fixture\nfunc value() {}\n"
    )
    refused = subprocess.run(
        [*command, "check"],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert refused.returncode == 1, refused.stderr
    assert json.loads(refused.stdout)["refusals"]
