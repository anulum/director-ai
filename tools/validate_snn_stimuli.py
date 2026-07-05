#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Validate, migrate, and write canonical SNN stimulus records."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn, cast

REQUIRED_KEYS = frozenset({"content", "project", "actor", "timestamp"})
OPTIONAL_KEYS = frozenset({"entities", "kind", "source_ref"})
ALLOWED_KEYS = REQUIRED_KEYS | OPTIONAL_KEYS
DEFAULT_STIMULUS_DIR = Path("04_ARCANE_SAPIENCE/snn_stimuli")
FILENAME_TIMESTAMP_RE = re.compile(r"^[A-Za-z0-9_.-]+_(?P<epoch>[0-9]{10})\.json$")
ACTOR_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")
MIN_CONTENT_LENGTH = 15


@dataclass(frozen=True)
class StimulusIssue:
    """A validation or migration issue tied to one stimulus file."""

    path: Path
    message: str

    def format(self, root: Path) -> str:
        """Return a deterministic, repository-friendly issue string."""
        try:
            display_path = self.path.relative_to(root)
        except ValueError:
            display_path = self.path
        return f"{display_path.as_posix()}: {self.message}"


def validate_stimulus_file(path: Path) -> list[StimulusIssue]:
    """Validate one JSON stimulus file against the canonical schema."""
    try:
        payload = _read_json(path)
    except ValueError as exc:
        return [StimulusIssue(path, str(exc))]

    mapping = _as_string_mapping(payload)
    if mapping is None:
        return [StimulusIssue(path, "record must be a JSON object with string keys")]

    return _validate_mapping(mapping, path)


def validate_stimulus_dir(stimulus_dir: Path) -> list[StimulusIssue]:
    """Validate all JSON stimulus files in a stimulus directory."""
    if not stimulus_dir.exists():
        return [StimulusIssue(stimulus_dir, "stimulus directory is missing")]
    if not stimulus_dir.is_dir():
        return [StimulusIssue(stimulus_dir, "stimulus path is not a directory")]

    issues: list[StimulusIssue] = []
    for path in _stimulus_files(stimulus_dir):
        issues.extend(validate_stimulus_file(path))
    return issues


def canonicalise_stimulus_record(payload: object, path: Path) -> dict[str, object]:
    """Return a canonical record for a current or legacy stimulus payload."""
    mapping = _as_string_mapping(payload)
    if mapping is None:
        _raise_canonicalise_error(path, "record must be a JSON object with string keys")

    content = _legacy_content(mapping, path)
    project = _first_non_empty_string(mapping, ("project", "repo"))
    actor = _first_non_empty_string(mapping, ("actor", "agent", "source"))
    timestamp = _canonical_timestamp(mapping.get("timestamp"), path)

    if project is None:
        _raise_canonicalise_error(path, "missing project or repo")
    if actor is None:
        _raise_canonicalise_error(path, "missing actor, agent, or source")

    record: dict[str, object] = {
        "content": content,
        "project": project,
        "actor": actor,
        "timestamp": timestamp,
    }

    entities = _string_sequence(mapping.get("entities"))
    if entities:
        record["entities"] = entities

    record["kind"] = _first_non_empty_string(mapping, ("kind",)) or "session_evidence"

    source_ref = _first_non_empty_string(mapping, ("source_ref", "commit"))
    if source_ref is not None:
        record["source_ref"] = source_ref

    return record


def migrate_stimulus_dir(
    stimulus_dir: Path, *, apply: bool
) -> tuple[int, list[StimulusIssue]]:
    """Migrate legacy stimulus files to the canonical schema."""
    if not stimulus_dir.exists():
        return 0, [StimulusIssue(stimulus_dir, "stimulus directory is missing")]
    if not stimulus_dir.is_dir():
        return 0, [StimulusIssue(stimulus_dir, "stimulus path is not a directory")]

    migrated = 0
    issues: list[StimulusIssue] = []
    for path in _stimulus_files(stimulus_dir):
        if not validate_stimulus_file(path):
            continue
        try:
            payload = _read_json(path)
            record = canonicalise_stimulus_record(payload, path)
        except ValueError as exc:
            issues.append(StimulusIssue(path, str(exc)))
            continue

        record_issues = _validate_mapping(record, path)
        if record_issues:
            issues.extend(record_issues)
            continue

        migrated += 1
        if apply:
            _write_json_atomic(path, record, overwrite=True)

    return migrated, issues


def build_stimulus_record(
    *,
    content: str,
    project: str,
    actor: str,
    timestamp: str | None,
    entities: Sequence[str],
    kind: str | None,
    source_ref: str | None,
) -> dict[str, object]:
    """Build a canonical SNN stimulus record from explicit CLI fields."""
    if timestamp is None:
        timestamp = datetime.now(tz=UTC).isoformat()
    record: dict[str, object] = {
        "content": content,
        "project": project,
        "actor": actor,
        "timestamp": _canonical_timestamp(timestamp, Path("<cli>")),
    }
    if entities:
        record["entities"] = list(entities)
    if kind is not None:
        record["kind"] = kind
    if source_ref is not None:
        record["source_ref"] = source_ref
    return record


def write_stimulus_record(output_path: Path, record: Mapping[str, object]) -> None:
    """Write one canonical stimulus record without overwriting an existing file."""
    issues = _validate_mapping(record, output_path)
    if issues:
        message = "; ".join(issue.message for issue in issues)
        raise ValueError(message)
    _write_json_atomic(output_path, record, overwrite=False)


def main(argv: list[str] | None = None) -> int:
    """Run the SNN stimulus validation, migration, or writer CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        return _handle_validate(args.stimulus_dir)
    if args.command == "migrate":
        return _handle_migrate(args.stimulus_dir, apply=args.apply)
    if args.command == "write":
        return _handle_write(args)

    parser.error(f"unknown command: {args.command}")
    return 2


def _read_json(path: Path) -> object:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read JSON file: {exc}") from exc

    try:
        payload: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc
    return payload


def _as_string_mapping(payload: object) -> Mapping[str, object] | None:
    if not isinstance(payload, dict):
        return None
    if not all(isinstance(key, str) for key in payload):
        return None
    return cast(Mapping[str, object], payload)


def _validate_mapping(mapping: Mapping[str, object], path: Path) -> list[StimulusIssue]:
    issues: list[StimulusIssue] = []
    keys = set(mapping)
    missing = sorted(REQUIRED_KEYS - keys)
    unexpected = sorted(keys - ALLOWED_KEYS)
    if missing:
        issues.append(StimulusIssue(path, f"missing keys: {', '.join(missing)}"))
    if unexpected:
        issues.append(StimulusIssue(path, f"unexpected keys: {', '.join(unexpected)}"))

    content = mapping.get("content")
    if "content" in mapping and (
        not isinstance(content, str) or len(content.strip()) < MIN_CONTENT_LENGTH
    ):
        issues.append(
            StimulusIssue(
                path,
                f"content must be a string with at least {MIN_CONTENT_LENGTH} characters",
            )
        )

    for key in ("project", "actor"):
        value = mapping.get(key)
        if key in mapping and (not isinstance(value, str) or not value.strip()):
            issues.append(StimulusIssue(path, f"{key} must be a non-empty string"))

    if "timestamp" in mapping and not _timestamp_is_valid(mapping.get("timestamp")):
        issues.append(
            StimulusIssue(
                path, "timestamp must be an ISO-8601 string or positive epoch"
            )
        )

    entities = mapping.get("entities")
    if entities is not None and not _string_sequence(entities):
        issues.append(StimulusIssue(path, "entities must be a non-empty string list"))

    for key in ("kind", "source_ref"):
        value = mapping.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            issues.append(StimulusIssue(path, f"{key} must be a non-empty string"))

    return issues


def _timestamp_is_valid(value: object) -> bool:
    try:
        _canonical_timestamp(value, Path("<validation>"))
    except ValueError:
        return False
    return True


def _canonical_timestamp(value: object, path: Path) -> str:
    if value is None:
        return _timestamp_from_filename(path)
    if isinstance(value, bool):
        _raise_canonicalise_error(path, "timestamp must not be boolean")
    if isinstance(value, int | float):
        if value <= 0:
            _raise_canonicalise_error(path, "timestamp epoch must be positive")
        return datetime.fromtimestamp(float(value), tz=UTC).isoformat()
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            _raise_canonicalise_error(path, "timestamp must be non-empty")
        if normalized.endswith("Z"):
            normalized = f"{normalized[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(
                f"invalid timestamp for {path.as_posix()}: {value}"
            ) from exc
        return parsed.isoformat()
    _raise_canonicalise_error(path, "timestamp must be an ISO string or epoch")


def _timestamp_from_filename(path: Path) -> str:
    match = FILENAME_TIMESTAMP_RE.match(path.name)
    if match is None:
        _raise_canonicalise_error(path, "missing timestamp and filename epoch")
    epoch = int(match.group("epoch"))
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat()


def _legacy_content(mapping: Mapping[str, object], path: Path) -> str:
    direct = _first_non_empty_string(mapping, ("content", "text"))
    if direct is not None:
        return _content_or_error(direct, path)

    task = _first_non_empty_string(mapping, ("task",))
    signals = _string_sequence(mapping.get("signals"))
    verification = _string_sequence(mapping.get("verification"))

    pieces: list[str] = []
    if task is not None and signals:
        pieces.append(f"{task}: {'; '.join(signals)}")
    elif task is not None:
        pieces.append(task)
    elif signals:
        pieces.append("; ".join(signals))

    if verification:
        pieces.append(f"verification: {'; '.join(verification)}")

    if not pieces:
        compact = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
        pieces.append(f"migrated legacy stimulus: {compact}")

    return _content_or_error("; ".join(pieces), path)


def _content_or_error(value: str, path: Path) -> str:
    content = value.strip()
    if len(content) < MIN_CONTENT_LENGTH:
        _raise_canonicalise_error(
            path,
            f"content must be at least {MIN_CONTENT_LENGTH} characters",
        )
    return content


def _string_sequence(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return []
        items.append(item.strip())
    return items


def _first_non_empty_string(
    mapping: Mapping[str, object],
    keys: Sequence[str],
) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _stimulus_files(stimulus_dir: Path) -> list[Path]:
    return sorted(stimulus_dir.glob("*.json"))


def _write_json_atomic(
    output_path: Path,
    record: Mapping[str, object],
    *,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {output_path}")
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    tmp_path.write_text(
        f"{json.dumps(record, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )
    tmp_path.replace(output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate, migrate, and write canonical SNN stimulus records.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate stimulus JSON files")
    validate.add_argument(
        "stimulus_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_STIMULUS_DIR,
        help="stimulus directory to validate",
    )

    migrate = subparsers.add_parser("migrate", help="migrate legacy stimulus files")
    migrate.add_argument(
        "stimulus_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_STIMULUS_DIR,
        help="stimulus directory to migrate",
    )
    migrate.add_argument(
        "--apply",
        action="store_true",
        help="rewrite legacy files in place",
    )

    write = subparsers.add_parser("write", help="write one canonical stimulus record")
    write.add_argument(
        "--stimulus-dir",
        type=Path,
        default=DEFAULT_STIMULUS_DIR,
        help="directory used when --output is omitted",
    )
    write.add_argument("--output", type=Path, help="explicit output JSON path")
    write.add_argument("--content", required=True, help="canonical content payload")
    write.add_argument("--project", required=True, help="project identifier")
    write.add_argument("--actor", required=True, help="writer actor identifier")
    write.add_argument("--timestamp", help="ISO-8601 timestamp; defaults to UTC now")
    write.add_argument("--entity", action="append", default=[], help="related entity")
    write.add_argument("--kind", help="stimulus kind")
    write.add_argument("--source-ref", help="source document or commit reference")
    return parser


def _handle_validate(stimulus_dir: Path) -> int:
    issues = validate_stimulus_dir(stimulus_dir)
    if issues:
        _print_issues(stimulus_dir, issues)
        return 1
    print(f"snn_stimuli_ok: {len(_stimulus_files(stimulus_dir))} files")
    return 0


def _handle_migrate(stimulus_dir: Path, *, apply: bool) -> int:
    migrated, issues = migrate_stimulus_dir(stimulus_dir, apply=apply)
    if issues:
        _print_issues(stimulus_dir, issues)
        return 1
    verb = "snn_stimuli_migrated" if apply else "snn_stimuli_migration_plan"
    print(f"{verb}: {migrated} files")
    return 0


def _handle_write(args: argparse.Namespace) -> int:
    output = args.output or _next_output_path(args.stimulus_dir, args.actor)
    record = build_stimulus_record(
        content=args.content,
        project=args.project,
        actor=args.actor,
        timestamp=args.timestamp,
        entities=args.entity,
        kind=args.kind,
        source_ref=args.source_ref,
    )
    try:
        write_stimulus_record(output, record)
    except (OSError, ValueError) as exc:
        print(f"write failed: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {output.as_posix()}")
    return 0


def _next_output_path(stimulus_dir: Path, actor: str) -> Path:
    slug = ACTOR_SLUG_RE.sub("_", actor.strip().lower()).strip("_") or "stimulus"
    epoch = int(time.time())
    candidate = stimulus_dir / f"{slug}_{epoch}.json"
    counter = 1
    while candidate.exists():
        candidate = stimulus_dir / f"{slug}_{epoch}_{counter}.json"
        counter += 1
    return candidate


def _print_issues(root: Path, issues: Sequence[StimulusIssue]) -> None:
    for issue in issues:
        print(issue.format(root), file=sys.stderr)


def _raise_canonicalise_error(path: Path, message: str) -> NoReturn:
    raise ValueError(f"{path.as_posix()}: {message}")


if __name__ == "__main__":
    raise SystemExit(main())
