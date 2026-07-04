# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real server and CLI coverage for audit-chain verification."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="fastapi required for server route tests")

from fastapi.testclient import TestClient

from director_ai.core.config import DirectorConfig
from director_ai.server import create_app
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def _audit_chain_client(audit_log: Path) -> TestClient:
    """Build a real FastAPI app that writes the structured audit log."""
    config = DirectorConfig(
        mode="general",
        scorer_backend="lite",
        use_nli=True,
        coherence_threshold=0.0,
        hard_limit=0.0,
        soft_limit=0.0,
        adaptive_threshold_enabled=False,
        hybrid_retrieval=False,
        reranker_enabled=False,
        retrieval_abstention_threshold=0.0,
        audit_log_path=str(audit_log),
    )
    return TestClient(create_app(config))


def _read_records(audit_log: Path) -> list[dict[str, object]]:
    """Read compact JSONL audit records from the production file format."""
    return [
        json.loads(line)
        for line in audit_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_records(audit_log: Path, records: list[dict[str, object]]) -> None:
    """Write JSONL audit records after deliberate tamper mutation."""
    audit_log.write_text(
        "\n".join(json.dumps(record, separators=(",", ":")) for record in records)
        + "\n",
        encoding="utf-8",
    )


def _audit_hmac_material() -> str:
    """Return deterministic test HMAC material for audit-chain verification."""
    return "audit-chain-real-surface-" + "hmac-material-which-is-long-enough"


def _verify_audit(
    audit_log: Path, key_material: str
) -> subprocess.CompletedProcess[str]:
    """Run the production CLI verifier in a real subprocess."""
    env = {
        **os.environ,
        "DIRECTOR_FORCE_CPU": "1",
    }
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "director_ai.cli",
            "verify-audit",
            str(audit_log),
            "--secret",
            key_material,
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=10,
    )


def test_audit_chain_unit_guard_declares_this_companion() -> None:
    """The legacy audit-chain unit guard should declare this companion."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_audit_chain.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_audit_chain_real_surface.py" in reason


def test_public_review_audit_log_verifies_and_rejects_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public review should write a CLI-verifiable tamper-evident chain."""
    key_material = _audit_hmac_material()
    monkeypatch.setenv("DIRECTOR_AUDIT_HMAC_SECRET", key_material)
    audit_log = tmp_path / "review-audit.jsonl"

    with _audit_chain_client(audit_log) as client:
        first = client.post(
            "/v1/review",
            headers={"X-Tenant-ID": "tenant-alpha"},
            json={
                "prompt": "Which audit control is required?",
                "response": "Every review decision is written to the audit log.",
            },
        )
        second = client.post(
            "/v1/review",
            headers={"X-Tenant-ID": "tenant-alpha"},
            json={
                "prompt": "How is tamper evidence checked?",
                "response": "The verifier replays the JSONL hash chain.",
            },
        )

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    records = _read_records(audit_log)
    assert len(records) == 2
    assert records[0]["tenant_id"] == "tenant-alpha"
    assert records[0]["query_hash"] != "Which audit control is required?"
    assert records[0]["prev_hash"] == "0" * 64
    assert records[1]["prev_hash"] == records[0]["entry_hash"]

    clean = _verify_audit(audit_log, key_material)
    assert clean.returncode == 0, clean.stderr
    assert f"Audit chain VERIFIED: {audit_log}" in clean.stdout

    records[1]["approved"] = not bool(records[1]["approved"])
    _write_records(audit_log, records)

    tampered = _verify_audit(audit_log, key_material)
    assert tampered.returncode == 1
    assert f"Audit chain TAMPERED at entry 1: {audit_log}" in tampered.stdout
