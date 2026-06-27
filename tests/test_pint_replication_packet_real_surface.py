# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - PINT replication packet real-surface tests
"""Real subprocess coverage for the PINT replication packet validator CLI."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_pint_replication_packet.py"


def _run_validator(root: Path) -> subprocess.CompletedProcess[str]:
    """Run the production PINT replication validator CLI for ``root``."""
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(root)],
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )


def test_pint_replication_validator_cli_accepts_live_packet() -> None:
    """The production CLI should validate the checked-in packet and cases."""
    result = _run_validator(ROOT)

    assert result.returncode == 0
    assert result.stdout == "pint_replication_packet_ok\n"
    assert result.stderr == ""


def test_pint_replication_validator_cli_reports_packet_errors(
    tmp_path: Path,
) -> None:
    """The production CLI should return non-zero and print validation errors."""
    benchmark_dir = tmp_path / "benchmarks"
    benchmark_dir.mkdir()
    packet = benchmark_dir / "pint_replication_packet.toml"
    shutil.copyfile(ROOT / "benchmarks" / "pint_replication_packet.toml", packet)
    shutil.copyfile(
        ROOT / "benchmarks" / "pint_seed_cases.jsonl",
        benchmark_dir / "pint_seed_cases.jsonl",
    )
    packet.write_text(
        packet.read_text(encoding="utf-8").replace(
            "public_score_claim = false",
            "public_score_claim = true",
        ),
        encoding="utf-8",
    )

    result = _run_validator(tmp_path)

    assert result.returncode == 1
    assert result.stdout == ""
    assert (
        "benchmarks/pint_replication_packet.toml: seed replication packet "
        "must not set public_score_claim=true"
    ) in result.stderr
