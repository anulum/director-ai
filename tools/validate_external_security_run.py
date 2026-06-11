# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - external security run evidence validator

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "security" / "external_security_test_packet.toml"
DENY_TOKENS = ("authorization", "x-api-key", "cookie", "set-cookie", "bearer ")
SEVERITIES = {"info", "low", "medium", "high", "critical"}
TRACK_STATUSES = {"pass", "fail", "blocked", "skipped"}
SUMMARY_STATUS_RE = re.compile(
    r"^- (?P<track_id>[a-z0-9_]+): (?P<status>[A-Za-z]+)(?: (?P<reason>.+))?$"
)
SUMMARY_STATUS_PREFIX_RE = re.compile(r"^-\s*(?P<track_id>[a-z0-9_]+):")
REQUIRED_FRAME_TYPES = {"accepted", "rejected", "halted", "cancelled"}
FULL_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
TARGET_COMMIT_RE = re.compile(r"^target_commit: (?P<commit>[0-9a-f]{40})$")


def _load_packet() -> dict[str, Any]:
    return tomllib.loads(PACKET.read_text(encoding="utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        records.append(item)
    return records


def _assert_no_denied_tokens(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    for token in DENY_TOKENS:
        if token in text:
            raise ValueError(f"{path} contains unredacted sensitive marker: {token}")


def _assert_csv_columns(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = required - columns
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        rows = list(reader)
        if not rows:
            raise ValueError(f"{path} must contain at least one row")
        for row_number, row in enumerate(rows, 2):
            for column in sorted(required):
                if not row.get(column, "").strip():
                    raise ValueError(f"{path}:{row_number} {column} must be non-empty")
        return rows


def _normalise_cell(value: str | None) -> str:
    return str(value or "").strip().lower()


def _assert_expected_actual_match(
    path: Path,
    rows: list[dict[str, str]],
    *,
    expected_column: str,
    actual_column: str,
) -> None:
    for row_number, row in enumerate(rows, 2):
        expected = _normalise_cell(row.get(expected_column))
        actual = _normalise_cell(row.get(actual_column))
        if actual != expected:
            raise ValueError(
                f"{path}:{row_number} {actual_column}={row.get(actual_column, '')} "
                f"does not match {expected_column}={row.get(expected_column, '')}"
            )


def _assert_http_transcripts(root: Path) -> None:
    transcript_dir = root / "http_transcripts"
    files = [path for path in transcript_dir.rglob("*") if path.is_file()]
    if not files:
        raise ValueError("http_transcripts must contain at least one redacted file")
    for path in files:
        _assert_resolved_inside_root(root, path, str(path.relative_to(root)))
        if not path.read_text(encoding="utf-8", errors="replace").strip():
            raise ValueError(f"{path.relative_to(root)} must be non-empty")


def _assert_resolved_inside_root(root: Path, path: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes evidence root") from exc


def _parse_utc_timestamp(path: Path, field: str, value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{path.name} {field} must be an ISO-8601 UTC timestamp")
    try:
        return datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ValueError(
            f"{path.name} {field} must be an ISO-8601 UTC timestamp"
        ) from exc


def _require_non_empty_string(path: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value.strip()


def _has_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _require_full_git_sha(path: str, value: object) -> str:
    commit = _require_non_empty_string(path, value)
    if not FULL_GIT_SHA_RE.fullmatch(commit):
        raise ValueError(f"{path} must be a full git SHA")
    return commit


def _require_accepted_risk_detail(path: str, value: object) -> str:
    detail = _require_non_empty_string(path, value)
    words = [word for word in re.split(r"\s+", detail) if word]
    if len(words) < 5:
        raise ValueError(f"{path} must describe owner and rationale")
    return detail


def _validate_environment(root: Path) -> dict[str, Any]:
    path = root / "environment.json"
    env = _read_json(path)
    required = {
        "target_commit",
        "director_ai_version",
        "python",
        "platform",
        "enabled_extras",
        "config_fingerprint",
        "tester",
        "started_at",
        "completed_at",
    }
    missing = required - set(env)
    if missing:
        raise ValueError(f"environment.json missing fields: {sorted(missing)}")
    for field in (
        "target_commit",
        "director_ai_version",
        "python",
        "platform",
        "config_fingerprint",
        "tester",
    ):
        _require_non_empty_string(f"environment.json {field}", env[field])
    _require_full_git_sha("environment.json target_commit", env["target_commit"])
    if not isinstance(env["enabled_extras"], list):
        raise ValueError("environment.json enabled_extras must be a list")
    for index, extra in enumerate(env["enabled_extras"]):
        _require_non_empty_string(f"environment.json enabled_extras[{index}]", extra)
    started_at = _parse_utc_timestamp(path, "started_at", env["started_at"])
    completed_at = _parse_utc_timestamp(path, "completed_at", env["completed_at"])
    if completed_at <= started_at:
        raise ValueError("environment.json completed_at must be after started_at")
    return env


def _validate_frames(path: Path) -> None:
    records = _read_jsonl(path)
    if not records:
        raise ValueError(f"{path} must contain at least one frame")
    for index, item in enumerate(records, 1):
        frame_type = _require_non_empty_string(
            f"{path.name}:{index} type", item.get("type")
        )
        if frame_type not in REQUIRED_FRAME_TYPES:
            raise ValueError(f"{path.name}:{index} unknown frame type: {frame_type}")
        _require_non_empty_string(
            f"{path.name}:{index} session_id", item.get("session_id")
        )
    frame_types = {str(item.get("type", "")) for item in records}
    missing = REQUIRED_FRAME_TYPES - frame_types
    if missing:
        raise ValueError(f"{path} missing frame types: {sorted(missing)}")


def _validate_findings(root: Path, packet: dict[str, Any]) -> tuple[set[str], set[str]]:
    findings = _read_jsonl(root / "findings.jsonl")
    resolved_root = root.resolve()
    known_tracks = {str(track["id"]) for track in packet["test_tracks"]}
    surfaces_by_track = {
        str(track["id"]): {str(surface) for surface in track["surfaces"]}
        for track in packet["test_tracks"]
    }
    finding_tracks: set[str] = set()
    non_info_finding_tracks: set[str] = set()
    for index, finding in enumerate(findings, 1):
        required = {"severity", "track_id", "surface", "reproduction", "evidence_path"}
        missing = required - set(finding)
        if missing:
            raise ValueError(
                f"findings.jsonl:{index} missing fields: {sorted(missing)}"
            )
        severity = finding["severity"]
        if severity not in SEVERITIES:
            raise ValueError(f"findings.jsonl:{index} has invalid severity")
        if severity in {"high", "critical"} and not (
            _has_non_empty_string(finding.get("fix_commit"))
            or _has_non_empty_string(finding.get("accepted_risk"))
        ):
            raise ValueError(
                f"findings.jsonl:{index} {severity} finding requires "
                "fix_commit or accepted_risk"
            )
        if _has_non_empty_string(finding.get("fix_commit")):
            _require_full_git_sha(
                f"findings.jsonl:{index} fix_commit", finding["fix_commit"]
            )
        if _has_non_empty_string(finding.get("accepted_risk")):
            _require_accepted_risk_detail(
                f"findings.jsonl:{index} accepted_risk", finding["accepted_risk"]
            )
        track_id = _require_non_empty_string(
            f"findings.jsonl:{index} track_id", finding["track_id"]
        )
        if track_id not in known_tracks:
            raise ValueError(f"findings.jsonl:{index} has unknown track_id: {track_id}")
        surface = _require_non_empty_string(
            f"findings.jsonl:{index} surface", finding["surface"]
        )
        if surface not in surfaces_by_track[track_id]:
            raise ValueError(
                f"findings.jsonl:{index} surface is not declared for track "
                f"{track_id}: {surface}"
            )
        _require_non_empty_string(
            f"findings.jsonl:{index} reproduction", finding["reproduction"]
        )
        evidence_path = _require_non_empty_string(
            f"findings.jsonl:{index} evidence_path", finding["evidence_path"]
        )
        finding_tracks.add(track_id)
        evidence = (root / evidence_path).resolve()
        try:
            evidence.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                f"findings.jsonl:{index} evidence path escapes evidence root: {evidence}"
            ) from exc
        if not evidence.exists():
            raise ValueError(
                f"findings.jsonl:{index} evidence path not found: {evidence}"
            )
        if not evidence.is_file():
            raise ValueError(f"findings.jsonl:{index} evidence path must be a file")
        if severity != "info":
            non_info_finding_tracks.add(track_id)
    return finding_tracks, non_info_finding_tracks


def _validate_summary(
    root: Path,
    packet: dict[str, Any],
    env: dict[str, Any],
    finding_tracks: set[str],
    non_info_finding_tracks: set[str],
) -> None:
    summary = (root / "summary.md").read_text(encoding="utf-8")
    summary_lines = [line.strip() for line in summary.splitlines()]
    for line in summary_lines:
        if line.startswith("target_commit") and not line.startswith("target_commit:"):
            raise ValueError(
                f"summary.md unknown target commit line: {line.split(':', 1)[0]}"
            )
    target_commit_lines = [
        line for line in summary_lines if line.startswith("target_commit:")
    ]
    if len(target_commit_lines) != 1:
        raise ValueError("summary.md must contain exactly one target_commit line")
    target_commit_match = TARGET_COMMIT_RE.fullmatch(target_commit_lines[0])
    if not target_commit_match:
        raise ValueError(
            "summary.md target_commit line must be exactly 'target_commit: <sha>'"
        )
    summary_commit = target_commit_match.group("commit")
    if summary_commit != str(env["target_commit"]):
        raise ValueError("summary.md target_commit must match environment.json")
    known_tracks = {str(track["id"]) for track in packet["test_tracks"]}
    for line in summary_lines:
        match = SUMMARY_STATUS_PREFIX_RE.match(line)
        if match and match.group("track_id") not in known_tracks:
            raise ValueError(
                f"summary.md unknown status track id: {match.group('track_id')}"
            )
    for track in packet["test_tracks"]:
        track_id = str(track["id"])
        if track_id not in summary:
            raise ValueError(f"summary.md missing track id: {track_id}")
        status_prefix = f"- {track_id}:"
        matching_lines = [
            line for line in summary_lines if line.startswith(status_prefix)
        ]
        if not matching_lines:
            raise ValueError(f"summary.md missing status for track id: {track_id}")
        if len(matching_lines) > 1:
            raise ValueError(f"summary.md duplicate status for track id: {track_id}")
        status_match = SUMMARY_STATUS_RE.fullmatch(matching_lines[0])
        if not status_match:
            raise ValueError(
                f"summary.md malformed status line for track id: {track_id}"
            )
        status = status_match.group("status")
        reason = status_match.group("reason")
        if status not in TRACK_STATUSES:
            raise ValueError(
                f"summary.md invalid status for track id {track_id}: {status}"
            )
        if status == "pass" and track_id in non_info_finding_tracks:
            raise ValueError(
                f"summary.md passed track has non-info finding: {track_id}"
            )
        if status == "fail" and track_id not in finding_tracks:
            raise ValueError(f"summary.md failed track has no finding: {track_id}")
        if status == "fail" and track_id not in non_info_finding_tracks:
            raise ValueError(
                f"summary.md failed track has no non-info finding: {track_id}"
            )
        if status in {"pass", "fail"} and reason:
            raise ValueError(
                f"summary.md {status} track has unexpected reason: {track_id}"
            )
        if status in {"blocked", "skipped"} and not reason:
            raise ValueError(f"summary.md {status} track missing reason: {track_id}")


def _validate_tenant_matrix(root: Path) -> None:
    path = root / "tenant_matrix.csv"
    rows = _assert_csv_columns(
        path,
        {"tenant", "surface", "action", "expected_status", "actual_status"},
    )
    _assert_expected_actual_match(
        path,
        rows,
        expected_column="expected_status",
        actual_column="actual_status",
    )
    tenants = {row["tenant"].strip() for row in rows if row.get("tenant", "").strip()}
    if len(tenants) < 2:
        raise ValueError("tenant_matrix.csv must include at least two tenants")

    denied_statuses = {"401", "403", "404", "409", "422", "blocked", "rejected"}
    has_denied_case = any(
        row.get("expected_status", "").strip().lower() in denied_statuses
        for row in rows
    )
    if not has_denied_case:
        raise ValueError("tenant_matrix.csv must include a denied isolation case")


def validate_run(root: Path) -> list[str]:
    packet = _load_packet()
    errors: list[str] = []

    for item in packet["required_outputs"]:
        raw_path = str(item["path"])
        relative = Path(raw_path)
        path = root / relative.relative_to("security-validation")
        if raw_path.endswith("/"):
            if not path.is_dir():
                errors.append(f"missing directory: {path}")
            else:
                try:
                    _assert_resolved_inside_root(
                        root, path, str(path.relative_to(root))
                    )
                except ValueError as exc:
                    errors.append(str(exc))
        elif not path.is_file():
            errors.append(f"missing file: {path}")
        else:
            try:
                _assert_resolved_inside_root(root, path, str(path.relative_to(root)))
            except ValueError as exc:
                errors.append(str(exc))
    if errors:
        return errors

    try:
        env = _validate_environment(root)
        _validate_frames(root / "websocket_frames.jsonl")
        _assert_http_transcripts(root)
        _validate_tenant_matrix(root)
        ingestion_path = root / "ingestion_matrix.csv"
        _assert_expected_actual_match(
            ingestion_path,
            _assert_csv_columns(
                ingestion_path,
                {"tenant", "case", "expected_status", "actual_status"},
            ),
            expected_column="expected_status",
            actual_column="actual_status",
        )
        physical_path = root / "physical_matrix.csv"
        _assert_expected_actual_match(
            physical_path,
            _assert_csv_columns(
                physical_path,
                {"tenant", "case", "expected_decision", "actual_decision"},
            ),
            expected_column="expected_decision",
            actual_column="actual_decision",
        )
        attestation_path = root / "attestation_matrix.csv"
        _assert_expected_actual_match(
            attestation_path,
            _assert_csv_columns(
                attestation_path,
                {"issuer", "case", "expected_status", "actual_status"},
            ),
            expected_column="expected_status",
            actual_column="actual_status",
        )
        contract_path = root / "contract_matrix.csv"
        _assert_expected_actual_match(
            contract_path,
            _assert_csv_columns(
                contract_path,
                {"boundary", "case", "expected_status", "actual_status"},
            ),
            expected_column="expected_status",
            actual_column="actual_status",
        )
        finding_tracks, non_info_finding_tracks = _validate_findings(root, packet)
        _validate_summary(root, packet, env, finding_tracks, non_info_finding_tracks)
        for path in root.rglob("*"):
            if path.is_file():
                _assert_resolved_inside_root(root, path, str(path.relative_to(root)))
                _assert_no_denied_tokens(path)
    except ValueError as exc:
        errors.append(str(exc))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "evidence_dir",
        nargs="?",
        default="security-validation",
        help="Directory containing external security test evidence.",
    )
    args = parser.parse_args(argv)

    errors = validate_run(Path(args.evidence_dir))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("external security evidence accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
