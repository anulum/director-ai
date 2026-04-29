# SPDX-License-Identifier: AGPL-3.0-or-later
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
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "security" / "external_security_test_packet.toml"
DENY_TOKENS = ("authorization", "x-api-key", "cookie", "set-cookie", "bearer ")
SEVERITIES = {"info", "low", "medium", "high", "critical"}


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


def _assert_csv_columns(path: Path, required: set[str]) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = required - columns
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        rows = list(reader)
        if not rows:
            raise ValueError(f"{path} must contain at least one row")


def _validate_environment(root: Path) -> None:
    env = _read_json(root / "environment.json")
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
    if not isinstance(env["enabled_extras"], list):
        raise ValueError("environment.json enabled_extras must be a list")


def _validate_frames(path: Path) -> None:
    records = _read_jsonl(path)
    if not records:
        raise ValueError(f"{path} must contain at least one frame")
    frame_types = {str(item.get("type", "")) for item in records}
    required = {"accepted", "rejected", "halted", "cancelled"}
    missing = required - frame_types
    if missing:
        raise ValueError(f"{path} missing frame types: {sorted(missing)}")


def _validate_findings(root: Path) -> None:
    findings = _read_jsonl(root / "findings.jsonl")
    for index, finding in enumerate(findings, 1):
        required = {"severity", "surface", "reproduction", "evidence_path"}
        missing = required - set(finding)
        if missing:
            raise ValueError(
                f"findings.jsonl:{index} missing fields: {sorted(missing)}"
            )
        if str(finding["severity"]).lower() not in SEVERITIES:
            raise ValueError(f"findings.jsonl:{index} has invalid severity")
        evidence = root / str(finding["evidence_path"])
        if not evidence.exists():
            raise ValueError(
                f"findings.jsonl:{index} evidence path not found: {evidence}"
            )


def _validate_summary(root: Path, packet: dict[str, Any]) -> None:
    summary = (root / "summary.md").read_text(encoding="utf-8")
    for track in packet["test_tracks"]:
        track_id = str(track["id"])
        if track_id not in summary:
            raise ValueError(f"summary.md missing track id: {track_id}")
    if "target_commit" not in summary:
        raise ValueError("summary.md must include target_commit")


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
        elif not path.is_file():
            errors.append(f"missing file: {path}")
    if errors:
        return errors

    try:
        _validate_environment(root)
        _validate_frames(root / "websocket_frames.jsonl")
        _assert_csv_columns(
            root / "tenant_matrix.csv",
            {"tenant", "surface", "action", "expected_status", "actual_status"},
        )
        _assert_csv_columns(
            root / "ingestion_matrix.csv",
            {"tenant", "case", "expected_status", "actual_status"},
        )
        _assert_csv_columns(
            root / "physical_matrix.csv",
            {"tenant", "case", "expected_decision", "actual_decision"},
        )
        _assert_csv_columns(
            root / "attestation_matrix.csv",
            {"issuer", "case", "expected_status", "actual_status"},
        )
        _assert_csv_columns(
            root / "contract_matrix.csv",
            {"boundary", "case", "expected_status", "actual_status"},
        )
        _validate_findings(root)
        _validate_summary(root, packet)
        for path in root.rglob("*"):
            if path.is_file():
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
