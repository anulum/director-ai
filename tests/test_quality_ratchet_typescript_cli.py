# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — real native TypeScript snapshot and ratchet contracts
"""Preserve the complete native TypeScript CLI baseline regression for its owner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.test_quality_ratchet_typescript import (
    typescript_repository as typescript_repository,
)


def test_public_typescript_cli_records_and_checks_exact_original_baseline(
    typescript_repository: Path,
    tmp_path: Path,
) -> None:
    """Exercise actual native capture, immutable record and changed-file refusal.

    Parameters
    ----------
    typescript_repository : Path
        Actual original source repository.
    tmp_path : Path
        Actual isolated output custody directory.
    """
    root = typescript_repository
    command = [
        sys.executable,
        "-B",
        "-m",
        "tools.check_quality_ratchet",
        "--root",
        str(root),
        "--profile",
        "typescript-docs",
    ]
    baseline = tmp_path / "original.json"
    native = subprocess.run(
        [*command, "record", "--output", str(baseline)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert native.returncode == 0, native.stderr
    assert json.loads(native.stdout)["profile"] == "typescript-docs"
    installed = root / "tools/quality_ratchet_baseline.json"
    installed.parent.mkdir()
    installed.write_bytes(baseline.read_bytes())
    subprocess.run(
        ["git", "add", "tools/quality_ratchet_baseline.json"],
        cwd=root,
        check=True,
        capture_output=True,
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
            "adopt original baseline",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    output = tmp_path / "current.json"
    captured = subprocess.run(
        [*command, "capture", "--output", str(output)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert captured.returncode == 0, captured.stderr
    assert json.loads(output.read_text())["complete"] is True
    clean = subprocess.run(
        [*command, "check"], capture_output=True, text=True, timeout=30
    )
    assert clean.returncode == 0, clean.stderr
    (root / "packages/contract/value.ts").write_text(
        "export function value(): number { return 2; }\n"
    )
    rejected = subprocess.run(
        [*command, "check"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert rejected.returncode == 1, rejected.stderr
    assert json.loads(rejected.stdout)["refusals"]
