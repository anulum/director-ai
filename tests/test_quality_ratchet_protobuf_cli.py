# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — original native Protobuf documentation and public CLI tests
"""Exercise the shared quality CLI's native Protobuf profile."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.test_quality_ratchet_protobuf import (
    DOCUMENTED_SCHEMA,
)
from tests.test_quality_ratchet_protobuf import (
    protobuf_repository as protobuf_repository,
)


def test_public_protobuf_cli_original_baseline_and_changed_file_refusal(
    protobuf_repository: Path,
    tmp_path: Path,
) -> None:
    """Exercise real capture, immutable record, check and regression status.

    Parameters
    ----------
    protobuf_repository : Path
        Actual original native source repository.
    tmp_path : Path
        Explicit original output custody.
    """
    root = protobuf_repository
    command = [
        sys.executable,
        "-B",
        "-m",
        "tools.check_quality_ratchet",
        "--root",
        str(root),
        "--profile",
        "protobuf-docs",
        "--protoc",
        "protoc",
    ]
    baseline = tmp_path / "original-protobuf.json"
    recorded = subprocess.run(
        [*command, "record", "--output", str(baseline)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert recorded.returncode == 0, recorded.stderr
    installed = root / "tools/quality_ratchet_baseline.json"
    installed.parent.mkdir()
    installed.write_bytes(baseline.read_bytes())
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
            "adopt original profile",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    captured = tmp_path / "current.json"
    original = subprocess.run(
        [*command, "capture", "--output", str(captured)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert original.returncode == 0 and json.loads(captured.read_text())["complete"]
    check = subprocess.run(
        [*command, "check"], capture_output=True, text=True, timeout=20
    )
    assert check.returncode == 0, check.stderr
    (root / "proto/admission.proto").write_text(
        DOCUMENTED_SCHEMA.replace("// Original caller text.\n", "")
    )
    check = subprocess.run(
        [*command, "check"], capture_output=True, text=True, timeout=20
    )
    assert check.returncode == 1 and json.loads(check.stdout)["refusals"]
