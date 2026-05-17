# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - external security run validator tests

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_external_security_run.py"
SPEC = importlib.util.spec_from_file_location(
    "validate_external_security_run", VALIDATOR
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
validate_run = MODULE.validate_run


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _valid_evidence(root: Path) -> None:
    (root / "http_transcripts").mkdir(parents=True)
    (root / "http_transcripts" / "stream-redacted.json").write_text(
        '{"path":"/v1/stream","redacted":true}',
        encoding="utf-8",
    )
    _write_json(
        root / "environment.json",
        {
            "target_commit": "abc1234",
            "director_ai_version": "3.14.0",
            "python": "3.12",
            "platform": "linux",
            "enabled_extras": ["server"],
            "config_fingerprint": "sha256:abc",
            "tester": "independent-lab",
            "started_at": "2026-04-29T10:00:00Z",
            "completed_at": "2026-04-29T11:00:00Z",
        },
    )
    (root / "websocket_frames.jsonl").write_text(
        "\n".join(
            json.dumps({"type": item, "session_id": f"sid-{item}"})
            for item in ("accepted", "rejected", "halted", "cancelled")
        ),
        encoding="utf-8",
    )
    (root / "tenant_matrix.csv").write_text(
        "tenant,surface,action,expected_status,actual_status\n"
        "tenant-a,/v1/tenants,read,200,200\n"
        "tenant-b,/v1/tenants/tenant-a/facts,cross-tenant read,403,403\n",
        encoding="utf-8",
    )
    (root / "ingestion_matrix.csv").write_text(
        "tenant,case,expected_status,actual_status\ntenant-a,valid text,200,200\n",
        encoding="utf-8",
    )
    (root / "physical_matrix.csv").write_text(
        "tenant,case,expected_decision,actual_decision\n"
        "tenant-a,budget exhausted,block,block\n",
        encoding="utf-8",
    )
    (root / "attestation_matrix.csv").write_text(
        "issuer,case,expected_status,actual_status\n"
        "issuer-a,tampered passport,rejected,rejected\n",
        encoding="utf-8",
    )
    (root / "contract_matrix.csv").write_text(
        "boundary,case,expected_status,actual_status\n"
        "python-proto,safety event roundtrip,pass,pass\n",
        encoding="utf-8",
    )
    (root / "evidence.txt").write_text("redacted replay", encoding="utf-8")
    (root / "findings.jsonl").write_text(
        json.dumps(
            {
                "severity": "low",
                "surface": "/v1/stream",
                "reproduction": "replay script",
                "evidence_path": "evidence.txt",
            }
        ),
        encoding="utf-8",
    )
    (root / "summary.md").write_text(
        "target_commit: abc1234\n"
        "- streaming_interception: pass\n"
        "- multi_tenant_isolation: pass\n"
        "- knowledge_ingestion: pass\n"
        "- physical_hooks: pass\n"
        "- attestation: pass\n"
        "- cross_language_trust_boundary: pass\n",
        encoding="utf-8",
    )


def test_validate_external_security_run_accepts_complete_evidence(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)

    assert validate_run(tmp_path) == []


def test_validate_external_security_run_rejects_missing_frames(tmp_path: Path) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "websocket_frames.jsonl").write_text(
        json.dumps({"type": "accepted", "session_id": "sid-1"}),
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert "missing frame types" in errors[0]


def test_validate_external_security_run_rejects_unredacted_markers(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "http_transcripts" / "bad.txt").write_text(
        "Authorization: bearer live-token",
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert "unredacted sensitive marker" in errors[0]


def test_validate_external_security_run_rejects_finding_path_traversal(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    outside = tmp_path.parent / "outside-evidence.txt"
    outside.write_text("outside", encoding="utf-8")
    (tmp_path / "findings.jsonl").write_text(
        json.dumps(
            {
                "severity": "medium",
                "surface": "/v1/stream",
                "reproduction": "replay script",
                "evidence_path": "../outside-evidence.txt",
            }
        ),
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert "escapes evidence root" in errors[0]


def test_validate_external_security_run_rejects_empty_http_transcripts(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "http_transcripts" / "stream-redacted.json").unlink()

    errors = validate_run(tmp_path)

    assert errors
    assert "http_transcripts must contain" in errors[0]


def test_validate_external_security_run_rejects_weak_tenant_matrix(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "tenant_matrix.csv").write_text(
        "tenant,surface,action,expected_status,actual_status\n"
        "tenant-a,/v1/tenants,read,200,200\n",
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert "tenant_matrix.csv must include at least two tenants" in errors[0]


def test_validate_external_security_run_rejects_summary_commit_mismatch(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "summary.md").write_text(
        "target_commit: different\n"
        "- streaming_interception: pass\n"
        "- multi_tenant_isolation: pass\n"
        "- knowledge_ingestion: pass\n"
        "- physical_hooks: pass\n"
        "- attestation: pass\n"
        "- cross_language_trust_boundary: pass\n",
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert "summary.md target_commit must match environment.json" in errors[0]
