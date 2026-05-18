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
            "target_commit": "0123456789abcdef0123456789abcdef01234567",
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
                "severity": "info",
                "track_id": "streaming_interception",
                "surface": "/v1/stream",
                "reproduction": "replay script",
                "evidence_path": "evidence.txt",
            }
        ),
        encoding="utf-8",
    )
    (root / "summary.md").write_text(
        "target_commit: 0123456789abcdef0123456789abcdef01234567\n"
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


def test_validate_external_security_run_rejects_required_file_symlink_escape(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    outside = tmp_path.parent / "outside-environment.json"
    outside.write_text(
        (tmp_path / "environment.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "environment.json").unlink()
    (tmp_path / "environment.json").symlink_to(outside)

    errors = validate_run(tmp_path)

    assert errors
    assert "environment.json escapes evidence root" in errors[0]


def test_validate_external_security_run_rejects_missing_frames(tmp_path: Path) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "websocket_frames.jsonl").write_text(
        json.dumps({"type": "accepted", "session_id": "sid-1"}),
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert "missing frame types" in errors[0]


def test_validate_external_security_run_rejects_frame_without_session_id(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "websocket_frames.jsonl").write_text(
        "\n".join(
            json.dumps({"type": item, "session_id": ""})
            for item in ("accepted", "rejected", "halted", "cancelled")
        ),
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert "websocket_frames.jsonl:1 session_id must be a non-empty string" in errors[0]


def test_validate_external_security_run_rejects_unknown_frame_type(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "websocket_frames.jsonl").write_text(
        "\n".join(
            json.dumps({"type": item, "session_id": f"sid-{item}"})
            for item in ("accepted", "rejected", "halted", "cancelled", "leaked")
        ),
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert "websocket_frames.jsonl:5 unknown frame type: leaked" in errors[0]


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
                "track_id": "streaming_interception",
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


def test_validate_external_security_run_rejects_blank_http_transcript_file(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "http_transcripts" / "stream-redacted.json").write_text(
        "   ",
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert "http_transcripts/stream-redacted.json must be non-empty" in errors[0]


def test_validate_external_security_run_rejects_transcript_symlink_escape(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    outside = tmp_path.parent / "outside-transcript.json"
    outside.write_text('{"path":"/v1/stream","redacted":true}', encoding="utf-8")
    transcript = tmp_path / "http_transcripts" / "stream-redacted.json"
    transcript.unlink()
    transcript.symlink_to(outside)

    errors = validate_run(tmp_path)

    assert errors
    assert "http_transcripts/stream-redacted.json escapes evidence root" in errors[0]


def test_validate_external_security_run_rejects_extra_file_symlink_escape(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    outside = tmp_path.parent / "outside-extra.txt"
    outside.write_text("redacted auxiliary evidence", encoding="utf-8")
    (tmp_path / "auxiliary.txt").symlink_to(outside)

    errors = validate_run(tmp_path)

    assert errors
    assert "auxiliary.txt escapes evidence root" in errors[0]


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


def test_validate_external_security_run_rejects_matrix_result_mismatch(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "tenant_matrix.csv").write_text(
        "tenant,surface,action,expected_status,actual_status\n"
        "tenant-a,/v1/tenants,read,200,200\n"
        "tenant-b,/v1/tenants/tenant-a/facts,cross-tenant read,403,200\n",
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert (
        "tenant_matrix.csv:3 actual_status=200 does not match expected_status=403"
        in errors[0]
    )


def test_validate_external_security_run_rejects_decision_matrix_mismatch(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "physical_matrix.csv").write_text(
        "tenant,case,expected_decision,actual_decision\n"
        "tenant-a,budget exhausted,block,warn\n",
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert (
        "physical_matrix.csv:2 actual_decision=warn "
        "does not match expected_decision=block"
    ) in errors[0]


def test_validate_external_security_run_rejects_blank_required_matrix_cell(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "ingestion_matrix.csv").write_text(
        "tenant,case,expected_status,actual_status\ntenant-a,valid text,,\n",
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert "ingestion_matrix.csv:2 actual_status must be non-empty" in errors[0]


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
    assert (
        "summary.md target_commit line must be exactly 'target_commit: <sha>'"
        in errors[0]
    )


def test_validate_external_security_run_rejects_target_commit_without_separator(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "summary.md").write_text(
        "target_commit:0123456789abcdef0123456789abcdef01234567\n"
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
    assert (
        "summary.md target_commit line must be exactly 'target_commit: <sha>'"
        in errors[0]
    )


def test_validate_external_security_run_rejects_unknown_target_commit_alias(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "summary.md").write_text(
        "target_commit: 0123456789abcdef0123456789abcdef01234567\n"
        "target_commit_sha: 0123456789abcdef0123456789abcdef01234567\n"
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
    assert "summary.md unknown target commit line: target_commit_sha" in errors[0]


def test_validate_external_security_run_rejects_short_target_commit(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    env = json.loads((tmp_path / "environment.json").read_text(encoding="utf-8"))
    env["target_commit"] = "abc1234"
    _write_json(tmp_path / "environment.json", env)
    (tmp_path / "summary.md").write_text(
        "target_commit: abc1234\n"
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
    assert "environment.json target_commit must be a full git SHA" in errors[0]


def test_validate_external_security_run_rejects_summary_without_track_status(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "summary.md").write_text(
        "target_commit: 0123456789abcdef0123456789abcdef01234567\n"
        "- streaming_interception\n"
        "- multi_tenant_isolation: pass\n"
        "- knowledge_ingestion: pass\n"
        "- physical_hooks: pass\n"
        "- attestation: pass\n"
        "- cross_language_trust_boundary: pass\n",
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert "summary.md missing status for track id: streaming_interception" in errors[0]


def test_validate_external_security_run_rejects_finding_without_known_track(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "findings.jsonl").write_text(
        json.dumps(
            {
                "severity": "low",
                "track_id": "invented_track",
                "surface": "/v1/stream",
                "reproduction": "replay script",
                "evidence_path": "evidence.txt",
            }
        ),
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert "findings.jsonl:1 has unknown track_id: invented_track" in errors[0]


def test_validate_external_security_run_rejects_finding_without_known_surface(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "findings.jsonl").write_text(
        json.dumps(
            {
                "severity": "low",
                "track_id": "streaming_interception",
                "surface": "/v1/not-in-packet",
                "reproduction": "replay script",
                "evidence_path": "evidence.txt",
            }
        ),
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert (
        "findings.jsonl:1 surface is not declared for track "
        "streaming_interception: /v1/not-in-packet"
    ) in errors[0]


def test_validate_external_security_run_rejects_failed_track_without_finding(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "findings.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "summary.md").write_text(
        "target_commit: 0123456789abcdef0123456789abcdef01234567\n"
        "- streaming_interception: fail\n"
        "- multi_tenant_isolation: pass\n"
        "- knowledge_ingestion: pass\n"
        "- physical_hooks: pass\n"
        "- attestation: pass\n"
        "- cross_language_trust_boundary: pass\n",
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert "summary.md failed track has no finding: streaming_interception" in errors[0]


def test_validate_external_security_run_rejects_failed_track_with_only_info_findings(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "findings.jsonl").write_text(
        json.dumps(
            {
                "severity": "info",
                "track_id": "streaming_interception",
                "surface": "/v1/stream",
                "reproduction": "replay script",
                "evidence_path": "evidence.txt",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "summary.md").write_text(
        "target_commit: 0123456789abcdef0123456789abcdef01234567\n"
        "- streaming_interception: fail\n"
        "- multi_tenant_isolation: pass\n"
        "- knowledge_ingestion: pass\n"
        "- physical_hooks: pass\n"
        "- attestation: pass\n"
        "- cross_language_trust_boundary: pass\n",
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert (
        "summary.md failed track has no non-info finding: streaming_interception"
        in errors[0]
    )


def test_validate_external_security_run_rejects_passed_track_with_non_info_finding(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "findings.jsonl").write_text(
        json.dumps(
            {
                "severity": "low",
                "track_id": "streaming_interception",
                "surface": "/v1/stream",
                "reproduction": "replay script",
                "evidence_path": "evidence.txt",
            }
        ),
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert (
        "summary.md passed track has non-info finding: streaming_interception"
        in errors[0]
    )


def test_validate_external_security_run_rejects_noncanonical_finding_severity(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "findings.jsonl").write_text(
        json.dumps(
            {
                "severity": "LOW",
                "track_id": "streaming_interception",
                "surface": "/v1/stream",
                "reproduction": "replay script",
                "evidence_path": "evidence.txt",
            }
        ),
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert "findings.jsonl:1 has invalid severity" in errors[0]


def test_validate_external_security_run_rejects_high_finding_without_disposition(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "findings.jsonl").write_text(
        json.dumps(
            {
                "severity": "high",
                "track_id": "streaming_interception",
                "surface": "/v1/stream",
                "reproduction": "replay script",
                "evidence_path": "evidence.txt",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "summary.md").write_text(
        "target_commit: 0123456789abcdef0123456789abcdef01234567\n"
        "- streaming_interception: fail\n"
        "- multi_tenant_isolation: pass\n"
        "- knowledge_ingestion: pass\n"
        "- physical_hooks: pass\n"
        "- attestation: pass\n"
        "- cross_language_trust_boundary: pass\n",
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert (
        "findings.jsonl:1 high finding requires fix_commit or accepted_risk"
        in errors[0]
    )


def test_validate_external_security_run_accepts_high_finding_with_fix_commit(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "findings.jsonl").write_text(
        json.dumps(
            {
                "severity": "high",
                "track_id": "streaming_interception",
                "surface": "/v1/stream",
                "reproduction": "replay script",
                "evidence_path": "evidence.txt",
                "fix_commit": "0123456789abcdef0123456789abcdef01234567",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "summary.md").write_text(
        "target_commit: 0123456789abcdef0123456789abcdef01234567\n"
        "- streaming_interception: fail\n"
        "- multi_tenant_isolation: pass\n"
        "- knowledge_ingestion: pass\n"
        "- physical_hooks: pass\n"
        "- attestation: pass\n"
        "- cross_language_trust_boundary: pass\n",
        encoding="utf-8",
    )

    assert validate_run(tmp_path) == []


def test_validate_external_security_run_rejects_short_finding_fix_commit(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "findings.jsonl").write_text(
        json.dumps(
            {
                "severity": "critical",
                "track_id": "streaming_interception",
                "surface": "/v1/stream",
                "reproduction": "replay script",
                "evidence_path": "evidence.txt",
                "fix_commit": "abc1234",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "summary.md").write_text(
        "target_commit: 0123456789abcdef0123456789abcdef01234567\n"
        "- streaming_interception: fail\n"
        "- multi_tenant_isolation: pass\n"
        "- knowledge_ingestion: pass\n"
        "- physical_hooks: pass\n"
        "- attestation: pass\n"
        "- cross_language_trust_boundary: pass\n",
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert "findings.jsonl:1 fix_commit must be a full git SHA" in errors[0]


def test_validate_external_security_run_accepts_critical_finding_with_accepted_risk(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "findings.jsonl").write_text(
        json.dumps(
            {
                "severity": "critical",
                "track_id": "streaming_interception",
                "surface": "/v1/stream",
                "reproduction": "replay script",
                "evidence_path": "evidence.txt",
                "accepted_risk": "documented exception reviewed by owner",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "summary.md").write_text(
        "target_commit: 0123456789abcdef0123456789abcdef01234567\n"
        "- streaming_interception: fail\n"
        "- multi_tenant_isolation: pass\n"
        "- knowledge_ingestion: pass\n"
        "- physical_hooks: pass\n"
        "- attestation: pass\n"
        "- cross_language_trust_boundary: pass\n",
        encoding="utf-8",
    )

    assert validate_run(tmp_path) == []


def test_validate_external_security_run_rejects_terse_accepted_risk(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "findings.jsonl").write_text(
        json.dumps(
            {
                "severity": "critical",
                "track_id": "streaming_interception",
                "surface": "/v1/stream",
                "reproduction": "replay script",
                "evidence_path": "evidence.txt",
                "accepted_risk": "ok",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "summary.md").write_text(
        "target_commit: 0123456789abcdef0123456789abcdef01234567\n"
        "- streaming_interception: fail\n"
        "- multi_tenant_isolation: pass\n"
        "- knowledge_ingestion: pass\n"
        "- physical_hooks: pass\n"
        "- attestation: pass\n"
        "- cross_language_trust_boundary: pass\n",
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert (
        "findings.jsonl:1 accepted_risk must describe owner and rationale" in errors[0]
    )


def test_validate_external_security_run_rejects_reversed_run_window(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    env = json.loads((tmp_path / "environment.json").read_text(encoding="utf-8"))
    env["started_at"] = "2026-04-29T11:00:00Z"
    env["completed_at"] = "2026-04-29T10:00:00Z"
    _write_json(tmp_path / "environment.json", env)

    errors = validate_run(tmp_path)

    assert errors
    assert "environment.json completed_at must be after started_at" in errors[0]


def test_validate_external_security_run_rejects_blank_environment_identity(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    env = json.loads((tmp_path / "environment.json").read_text(encoding="utf-8"))
    env["tester"] = " "
    _write_json(tmp_path / "environment.json", env)

    errors = validate_run(tmp_path)

    assert errors
    assert "environment.json tester must be a non-empty string" in errors[0]


def test_validate_external_security_run_rejects_blank_enabled_extra(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    env = json.loads((tmp_path / "environment.json").read_text(encoding="utf-8"))
    env["enabled_extras"] = ["server", " "]
    _write_json(tmp_path / "environment.json", env)

    errors = validate_run(tmp_path)

    assert errors
    assert "environment.json enabled_extras[1] must be a non-empty string" in errors[0]


def test_validate_external_security_run_rejects_summary_commit_prefix_only(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "summary.md").write_text(
        "target_commit: 0123456789abcdef0123456789abcdef01234567-extra\n"
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
    assert (
        "summary.md target_commit line must be exactly 'target_commit: <sha>'"
        in errors[0]
    )


def test_validate_external_security_run_rejects_duplicate_summary_status(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "summary.md").write_text(
        "target_commit: 0123456789abcdef0123456789abcdef01234567\n"
        "- streaming_interception: pass\n"
        "- streaming_interception: fail\n"
        "- multi_tenant_isolation: pass\n"
        "- knowledge_ingestion: pass\n"
        "- physical_hooks: pass\n"
        "- attestation: pass\n"
        "- cross_language_trust_boundary: pass\n",
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert (
        "summary.md duplicate status for track id: streaming_interception" in errors[0]
    )


def test_validate_external_security_run_rejects_summary_status_without_separator(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "summary.md").write_text(
        "target_commit: 0123456789abcdef0123456789abcdef01234567\n"
        "- streaming_interception:pass\n"
        "- multi_tenant_isolation: pass\n"
        "- knowledge_ingestion: pass\n"
        "- physical_hooks: pass\n"
        "- attestation: pass\n"
        "- cross_language_trust_boundary: pass\n",
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert (
        "summary.md malformed status line for track id: streaming_interception"
        in errors[0]
    )


def test_validate_external_security_run_rejects_unknown_summary_status_track(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "summary.md").write_text(
        "target_commit: 0123456789abcdef0123456789abcdef01234567\n"
        "- streaming_interception: pass\n"
        "- invented_track: pass\n"
        "- multi_tenant_isolation: pass\n"
        "- knowledge_ingestion: pass\n"
        "- physical_hooks: pass\n"
        "- attestation: pass\n"
        "- cross_language_trust_boundary: pass\n",
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert "summary.md unknown status track id: invented_track" in errors[0]


def test_validate_external_security_run_rejects_pass_summary_status_with_reason(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "summary.md").write_text(
        "target_commit: 0123456789abcdef0123456789abcdef01234567\n"
        "- streaming_interception: pass manually inspected\n"
        "- multi_tenant_isolation: pass\n"
        "- knowledge_ingestion: pass\n"
        "- physical_hooks: pass\n"
        "- attestation: pass\n"
        "- cross_language_trust_boundary: pass\n",
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert (
        "summary.md pass track has unexpected reason: streaming_interception"
        in errors[0]
    )


def test_validate_external_security_run_rejects_fail_summary_status_with_reason(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "findings.jsonl").write_text(
        json.dumps(
            {
                "severity": "low",
                "track_id": "streaming_interception",
                "surface": "/v1/stream",
                "reproduction": "replay script",
                "evidence_path": "evidence.txt",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "summary.md").write_text(
        "target_commit: 0123456789abcdef0123456789abcdef01234567\n"
        "- streaming_interception: fail see finding\n"
        "- multi_tenant_isolation: pass\n"
        "- knowledge_ingestion: pass\n"
        "- physical_hooks: pass\n"
        "- attestation: pass\n"
        "- cross_language_trust_boundary: pass\n",
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert (
        "summary.md fail track has unexpected reason: streaming_interception"
        in errors[0]
    )


def test_validate_external_security_run_rejects_noncanonical_summary_status(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "summary.md").write_text(
        "target_commit: 0123456789abcdef0123456789abcdef01234567\n"
        "- streaming_interception: PASS\n"
        "- multi_tenant_isolation: pass\n"
        "- knowledge_ingestion: pass\n"
        "- physical_hooks: pass\n"
        "- attestation: pass\n"
        "- cross_language_trust_boundary: pass\n",
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert (
        "summary.md invalid status for track id streaming_interception: PASS"
        in errors[0]
    )


def test_validate_external_security_run_rejects_blocked_summary_without_reason(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "summary.md").write_text(
        "target_commit: 0123456789abcdef0123456789abcdef01234567\n"
        "- streaming_interception: blocked\n"
        "- multi_tenant_isolation: pass\n"
        "- knowledge_ingestion: pass\n"
        "- physical_hooks: pass\n"
        "- attestation: pass\n"
        "- cross_language_trust_boundary: pass\n",
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert (
        "summary.md blocked track missing reason: streaming_interception" in errors[0]
    )


def test_validate_external_security_run_rejects_empty_finding_fields(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "findings.jsonl").write_text(
        json.dumps(
            {
                "severity": "low",
                "track_id": "streaming_interception",
                "surface": " ",
                "reproduction": "replay script",
                "evidence_path": "evidence.txt",
            }
        ),
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert "findings.jsonl:1 surface must be a non-empty string" in errors[0]


def test_validate_external_security_run_rejects_directory_evidence_path(
    tmp_path: Path,
) -> None:
    _valid_evidence(tmp_path)
    (tmp_path / "finding-evidence").mkdir()
    (tmp_path / "findings.jsonl").write_text(
        json.dumps(
            {
                "severity": "low",
                "track_id": "streaming_interception",
                "surface": "/v1/stream",
                "reproduction": "replay script",
                "evidence_path": "finding-evidence",
            }
        ),
        encoding="utf-8",
    )

    errors = validate_run(tmp_path)

    assert errors
    assert "findings.jsonl:1 evidence path must be a file" in errors[0]
