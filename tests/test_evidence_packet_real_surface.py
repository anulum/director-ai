# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — evidence packet real-surface tests
"""Real CLI coverage for Director-AI evidence packet emission."""

from __future__ import annotations

import json
import os

# The real-surface checks intentionally invoke the local Python CLI.
import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import cast

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

ROOT = Path(__file__).resolve().parents[1]


def _cli_env() -> dict[str, str]:
    """Return a deterministic local environment for CLI subprocess runs."""
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "DIRECTOR_FORCE_CPU": "1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": str(ROOT / "src"),
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return env


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the public ``director-ai`` module entry point with ``args``."""
    # Fixed local module invocation; shell remains false.
    return subprocess.run(  # nosec B603
        [sys.executable, "-m", "director_ai.cli", *args],
        cwd=ROOT,
        env=_cli_env(),
        text=True,
        capture_output=True,
        check=False,
        timeout=30.0,
    )


def _read_packet(path: Path) -> dict[str, object]:
    """Read a generated evidence packet from ``path``."""
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def test_evidence_packet_unit_guard_has_real_cli_companion() -> None:
    """The helper-heavy packet guard should be backed by public CLI coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_evidence_packet.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_evidence_packet_real_surface.py" in category


def test_evidence_packet_cli_emits_and_verifies_packet(tmp_path: Path) -> None:
    """The public evidence CLI should emit a packet accepted by verify CLI."""
    output_dir = tmp_path / "evidence"

    emit = _run_cli("evidence", "--emit", str(output_dir))

    assert emit.returncode == 0, emit.stdout + emit.stderr
    assert "grounded answer approved:    True" in emit.stdout
    assert "hallucinated answer blocked: True" in emit.stdout

    packet_path = output_dir / "evidence_packet.json"
    packet = _read_packet(packet_path)
    content = cast(dict[str, object], packet["content"])
    checks = cast(dict[str, object], content["checks"])
    integrity = cast(dict[str, object], packet["integrity"])
    assert content["schema_version"] == "director.evidence_packet.v1"
    assert isinstance(content["knowledge_base_size"], int)
    assert content["knowledge_base_size"] >= 5
    assert checks == {
        "grounded_approved": True,
        "hallucinated_blocked": True,
    }
    assert integrity["algorithm"] == "sha256"
    assert isinstance(integrity["digest"], str)
    assert len(integrity["digest"]) == 64

    verify_dir = _run_cli("verify-evidence", str(output_dir))
    assert verify_dir.returncode == 0, verify_dir.stdout + verify_dir.stderr
    assert "Evidence packet VERIFIED" in verify_dir.stdout

    verify_file = _run_cli("verify-evidence", str(packet_path))
    assert verify_file.returncode == 0, verify_file.stdout + verify_file.stderr
    assert "Evidence packet VERIFIED" in verify_file.stdout
