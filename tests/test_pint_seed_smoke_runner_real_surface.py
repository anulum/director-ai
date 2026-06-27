# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - PINT seed smoke runner real-surface tests
"""Real subprocess coverage for the PINT seed smoke runner CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_pint_seed_smoke.py"


def test_pint_seed_smoke_cli_writes_non_public_smoke_result(
    tmp_path: Path,
) -> None:
    """The production CLI should write claim-bounded smoke evidence."""
    output = tmp_path / "pint_seed_smoke.json"

    result = subprocess.run(
        [sys.executable, str(RUNNER), str(ROOT), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )

    assert result.returncode == 0
    assert result.stdout == f"pint_seed_smoke_written {output}\n"
    assert result.stderr == ""
    payload = cast(dict[str, object], json.loads(output.read_text(encoding="utf-8")))
    assert payload["result_type"] == "pint_seed_smoke"
    assert payload["public_score_claim"] is False
    assert payload["official_pint_score"] is False
    assert payload["benchmark_eligible"] is False
    assert payload["seed_cases"] == "benchmarks/pint_seed_cases.jsonl"
    assert payload["total_cases"] == 10
    assert payload["correct"] == 10
    assert "not an official PINT score" in cast(str, payload["claim_boundary"])
    per_case = cast(list[dict[str, object]], payload["per_case"])
    assert per_case
    assert "input" not in per_case[0]


def test_pint_seed_smoke_cli_reports_missing_repository_packet(
    tmp_path: Path,
) -> None:
    """The production CLI should return non-zero when validation fails."""
    output = tmp_path / "pint_seed_smoke.json"

    result = subprocess.run(
        [sys.executable, str(RUNNER), str(tmp_path), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert (
        "benchmarks/pint_replication_packet.toml: missing PINT replication packet"
    ) in result.stderr
    assert not output.exists()
