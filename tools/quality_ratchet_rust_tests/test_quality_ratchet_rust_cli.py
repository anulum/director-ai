# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — real native Rust documentation ratchet contracts
"""Exercise the shared Rust quality CLI with the exact native toolchain."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.quality_ratchet_rust_tests.test_quality_ratchet_rust import (
    ROOT,
)
from tools.quality_ratchet_rust_tests.test_quality_ratchet_rust import (
    rust_repository as rust_repository,
)
from tools.quality_ratchet_rust_tests.test_quality_ratchet_rust import (
    rust_tools as rust_tools,
)


def test_public_cli_records_actual_capture_and_refuses_unqualified_source(
    rust_repository: Path, rust_tools: tuple[str, str]
) -> None:
    """Exercise genuine capture, baseline record and source refusal through literal CLI."""
    cargo, rustc = rust_tools
    common = [
        sys.executable,
        "-m",
        "tools.check_quality_ratchet",
        "--root",
        str(rust_repository),
        "--profile",
        "rust-docs",
        "--cargo",
        cargo,
        "--rustc",
        rustc,
        "--timeout",
        "120",
    ]
    baseline = rust_repository / "tools/quality_ratchet_baseline.json"
    baseline.parent.mkdir()
    for operation in ("capture", "record"):
        output = baseline if operation == "record" else rust_repository / "capture.json"
        result = subprocess.run(
            [*common, operation, "--output", str(output)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=140,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        summary = json.loads(result.stdout)
        assert (
            summary["complete"] and summary["sources"] == 1 and summary["findings"] == 0
        )
        assert output.is_file()
    result = subprocess.run(
        [*common, "check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=140,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    subprocess.run(
        ["git", "add", str(baseline.relative_to(rust_repository))],
        cwd=rust_repository,
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
            "retain exact original native baseline",
        ],
        cwd=rust_repository,
        check=True,
        capture_output=True,
    )
    source = rust_repository / "backfire-kernel/crates/backfire-core/src/lib.rs"
    before = source.read_text()
    source.write_text(before.replace("/// Return the current count.\n", ""))
    result = subprocess.run(
        [*common, "check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=140,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    assert json.loads(result.stdout)["refusals"]
    source.write_text(before)
    source.write_text(source.read_text() + "opaque_declarations!();\n")
    result = subprocess.run(
        [*common, "capture", "--output", str(rust_repository / "refused.json")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=140,
        check=False,
    )
    assert result.returncode == 2 and "completely analyse" in result.stderr
    assert not (rust_repository / "refused.json").exists()
