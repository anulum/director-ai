# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — required native quality CLI tests
"""Exercise the complete required gate with real language tools and Git history."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from formal.HaltMonitor.tests.test_quality_ratchet_lean import (
    lean_binary as lean_binary,
)
from formal.HaltMonitor.tests.test_quality_ratchet_lean import (
    lean_repository as lean_repository,
)
from tests.quality_ratchet_python_support import quality_python
from tests.test_quality_ratchet_cli import commit_repository
from tests.test_quality_ratchet_golang_cli import go_repository as go_repository
from tests.test_quality_ratchet_golang_cli import go_tools as go_tools
from tests.test_quality_ratchet_typescript import (
    typescript_repository as typescript_repository,
)
from tests.test_quality_ratchet_typing_cli import typing_repository as typing_repository
from tools.ci_workflow_inventory import object_mapping
from tools.quality_ratchet_baseline import (
    BASELINE_PATH,
    Baseline,
    parse_baseline,
    resolve_commit,
)
from tools.quality_ratchet_contracts import parse_snapshot, snapshot_payload
from tools.quality_ratchet_gate import (
    POLICY_PATH,
    POLICY_SCHEMA,
    qualify_required_record,
    required_baseline_scope,
    required_input_state,
    verify_required_captures,
)
from tools.quality_ratchet_julia_tests.test_native_julia_report import (
    julia_binary as julia_binary,
)
from tools.quality_ratchet_profiles import REQUIRED_PROFILES, capture_profile
from tools.quality_ratchet_rust_tests.test_quality_ratchet_rust import (
    rust_repository as rust_repository,
)
from tools.quality_ratchet_rust_tests.test_quality_ratchet_rust import (
    rust_tools as rust_tools,
)

ROOT = Path(__file__).resolve().parents[2]


def write_policy(root: Path, *, profiles: tuple[str, ...] = REQUIRED_PROFILES) -> Path:
    """Install the real policy envelope bound to this fixture's immutable HEAD."""
    output = root / POLICY_PATH
    output.write_text(
        f'schema = "{POLICY_SCHEMA}"\n'
        f'origin_commit = "{resolve_commit(root, "HEAD")}"\n'
        f"profiles = {json.dumps(profiles)}\n"
    )
    return output


@pytest.fixture
def required_repository(
    typing_repository: Path,
    go_repository: Path,
    typescript_repository: Path,
    lean_repository: Path,
    rust_repository: Path,
) -> Path:
    """Assemble the original committed fixtures for every maintained native profile."""
    root = typing_repository
    (root / "requirements/ci-lint.txt").write_text("ruff==0.16.10\n")
    for source, directory in (
        (go_repository, "gateway"),
        (typescript_repository, "packages"),
        (lean_repository, "formal"),
        (rust_repository, "backfire-kernel"),
    ):
        shutil.copytree(source / directory, root / directory, dirs_exist_ok=True)
    shutil.copytree(ROOT / "tools/julia_tuner", root / "tools/julia_tuner")
    proto = root / "proto"
    proto.mkdir()
    (proto / "record.proto").write_text(
        "// Original admission wire contract.\n"
        'syntax = "proto3";\n'
        "package admission;\n"
        "// Actual original value.\n"
        "message Record {\n  // Original count.\n  uint32 count = 1;\n}\n"
    )
    notebooks = root / "notebooks"
    notebooks.mkdir()
    (notebooks / "record.ipynb").write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {
                    "kernelspec": {"language": "python", "name": "python3"},
                    "language_info": {"name": "python"},
                },
                "cells": [
                    {
                        "cell_type": "code",
                        "id": "record",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": "amount: int = 3",
                    }
                ],
            }
        )
    )
    commit_repository(root)
    write_policy(root)
    return root


def require_cli(
    root: Path,
    command: str,
    *arguments: str,
    executables: tuple[tuple[str, str], ...] = (),
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the public required-mode command using the actual native executables."""
    tool_arguments = [argument for pair in executables for argument in pair]
    return subprocess.run(
        [
            quality_python(),
            "-B",
            "-m",
            "tools.check_quality_ratchet",
            "--all",
            "--root",
            str(root),
            command,
            *tool_arguments,
            *arguments,
        ],
        cwd=ROOT,
        capture_output=True,
        env=environment,
        text=True,
        timeout=150,
        check=False,
    )


def test_required_first_record_and_committed_check(
    required_repository: Path,
    go_tools: tuple[str, str],
    lean_binary: str,
    rust_tools: tuple[str, str],
    julia_binary: str,
    tmp_path: Path,
) -> None:
    """Record all actual profiles and reject changed-file debt through the same gate."""
    root = required_repository
    authority = resolve_commit(root, "HEAD")
    executables = (
        ("--go", go_tools[0]),
        ("--revive", go_tools[1]),
        ("--lean", lean_binary),
        ("--cargo", rust_tools[0]),
        ("--rustc", rust_tools[1]),
        ("--julia", julia_binary),
    )
    record = tmp_path / "record.json"
    result = require_cli(
        root, "record", "--output", str(record), executables=executables
    )
    assert result.returncode == 0, result.stderr
    baseline = object_mapping(json.loads(record.read_bytes()))
    assert baseline["origin_commit"] == authority
    assert set(object_mapping(baseline["profiles"])) == set(REQUIRED_PROFILES)
    assert all(
        object_mapping(item)["complete"] is True
        for item in object_mapping(baseline["profiles"]).values()
    )
    shutil.copyfile(record, root / BASELINE_PATH)
    uncommitted = require_cli(root, "check", executables=executables)
    assert (
        uncommitted.returncode == 2 and "no trusted Git authority" in uncommitted.stderr
    )
    commit_repository(root)
    current = require_cli(root, "check", executables=executables)
    assert current.returncode == 0, current.stderr
    assert set(
        object_mapping(object_mapping(json.loads(current.stdout))["profiles"])
    ) == set(REQUIRED_PROFILES)
    _verify_authority_changes_during_capture(root, tmp_path, executables)
    (root / "src/legacy.py").write_text("def missing(value):\n    return value + 1\n")
    notebook = root / "notebooks/record.ipynb"
    original_cells = notebook.read_text()
    notebook.write_text(original_cells.replace("amount: int = 3", "amount: str = 3"))
    launcher = root / "tools/julia_tuner/bin/tune_threshold.jl"
    original_launcher = launcher.read_text()
    launcher.write_text(
        original_launcher.replace('help = "RNG seed (omit for non-deterministic)"', "")
    )
    regression = require_cli(root, "check", executables=executables)
    notebook.write_text(original_cells)
    launcher.write_text(original_launcher)
    assert regression.returncode == 1, regression.stderr
    violations = object_mapping(json.loads(regression.stdout))["refusals"]
    assert isinstance(violations, list)
    assert (
        "python-docs: New or modified source has native findings: src/legacy.py"
        in violations
    )
    assert (
        "python-types: New or modified source has native findings: src/legacy.py"
        in violations
    )
    assert (
        "notebook-types: New or modified source has native findings: notebooks/record.ipynb"
        in violations
    )
    assert (
        "julia-docs: New or modified source has native findings: tools/julia_tuner/bin/tune_threshold.jl"
        in violations
    )
    (root / "src/legacy.py").write_text(
        '"""Qualified original module."""\n'
        "def missing(value: int) -> int:\n"
        '    """Return the provided value."""\n'
        "    return value\n"
    )
    record2 = tmp_path / "reduced.json"
    reduced = require_cli(
        root, "record", "--output", str(record2), executables=executables
    )
    assert reduced.returncode == 0, reduced.stderr
    updated = object_mapping(json.loads(record2.read_bytes()))
    assert updated["origin_commit"] == authority
    for profile in ("python-docs", "python-types"):
        assert not object_mapping(object_mapping(updated["profiles"])[profile])[
            "findings"
        ]
    _verify_refusals_after_native_record(root, record2)
    captured = tmp_path / "capture.json"
    capture = require_cli(
        root, "capture", "--output", str(captured), executables=executables
    )
    assert capture.returncode == 0, capture.stderr
    assert (
        object_mapping(json.loads(captured.read_bytes()))["schema"]
        == "quality-ratchet.capture.v1"
    )
    shutil.copyfile(captured, root / BASELINE_PATH)
    wrong_authority = require_cli(root, "check", executables=executables)
    assert (
        wrong_authority.returncode == 2 and "baseline schema" in wrong_authority.stderr
    )
    policy = root / POLICY_PATH
    original_policy = policy.read_text()
    policy.write_text(original_policy + "# Changed policy\n")
    drift = require_cli(root, "check", executables=executables)
    assert (
        drift.returncode == 2 and "policy differs from the trusted base" in drift.stderr
    )
    policy.write_text(original_policy)
    profiles = object_mapping(baseline["profiles"])
    del profiles["protobuf-docs"]
    baseline["profiles"] = profiles
    (root / BASELINE_PATH).write_text(json.dumps(baseline))
    commit_repository(root)
    incomplete_authority = require_cli(root, "check", executables=executables)
    assert incomplete_authority.returncode == 2
    assert (
        "Required baseline authority or profile set changed"
        in incomplete_authority.stderr
    )


def test_required_capture_refuses_missing_profile(
    typing_repository: Path, tmp_path: Path
) -> None:
    """Refuse a required policy with an omitted native profile before output publication."""
    root = typing_repository
    write_policy(root, profiles=REQUIRED_PROFILES[:-1])
    output = tmp_path / "capture.json"
    result = require_cli(root, "capture", "--output", str(output))
    assert (
        result.returncode == 2 and "incomplete required quality policy" in result.stderr
    )
    assert not output.exists()


def test_native_dispatch_rejects_unregistered_profile(typing_repository: Path) -> None:
    """Reject an unregistered profile at the public native dispatch entry point."""
    with pytest.raises(ValueError, match="Unknown native quality profile"):
        capture_profile(typing_repository, Namespace(timeout=120.0), "unknown-docs")


def test_required_initial_record_refuses_dirty_historical_checkout(
    typing_repository: Path, tmp_path: Path
) -> None:
    """Refuse to record native historical authority from a modified Git checkout."""
    root = typing_repository
    write_policy(root)
    (root / "src/legacy.py").write_text("def missing(value):\n    return 7\n")
    output = tmp_path / "record.json"
    result = require_cli(root, "record", "--output", str(output))
    assert result.returncode == 2 and "clean historical checkout" in result.stderr
    assert not output.exists()


def test_required_initial_record_refuses_rewritten_committed_policy(
    typing_repository: Path, tmp_path: Path
) -> None:
    """Reject replacing a policy blob on the commit selected for initial adoption."""
    root = typing_repository
    write_policy(root)
    commit_repository(root)
    write_policy(root)

    output = tmp_path / "record.json"
    result = require_cli(root, "record", "--output", str(output))
    assert result.returncode == 2
    assert "Initial required quality policy differs from its origin" in result.stderr
    assert not output.exists()


def test_required_capture_publishes_nothing_if_a_native_tool_refuses(
    required_repository: Path,
    go_tools: tuple[str, str],
    tmp_path: Path,
) -> None:
    """An unavailable native executable must name its profile and fail closed."""
    output = tmp_path / "incomplete-capture.json"
    result = require_cli(
        required_repository,
        "capture",
        "--output",
        str(output),
        executables=(
            ("--go", go_tools[0]),
            ("--revive", go_tools[1]),
            ("--node", str(tmp_path / "unavailable-node")),
        ),
    )
    assert result.returncode == 2 and "typescript-docs:" in result.stderr
    assert not output.exists()


def _verify_refusals_after_native_record(root: Path, record: Path) -> None:
    """Check changed native inputs without substituting a fabricated collector result."""
    actual = parse_baseline(record.read_bytes())
    incomplete_payload = snapshot_payload(actual.profiles["protobuf-docs"])
    incomplete_payload["complete"] = False
    incomplete = Baseline(
        actual.origin_commit,
        actual.profiles | {"protobuf-docs": parse_snapshot(incomplete_payload)},
    )
    with pytest.raises(ValueError, match="Required baseline contains an incomplete"):
        required_baseline_scope(incomplete, actual.origin_commit)

    before = required_input_state(root)
    verify_required_captures(root, actual, before)

    configuration = root / "requirements/ci-lint.txt"
    original_config = configuration.read_bytes()
    configuration.write_bytes(original_config + b"\n# Configuration changed\n")
    with pytest.raises(ValueError, match="inventory or configuration changed"):
        verify_required_captures(root, actual, before)
    configuration.write_bytes(original_config)

    source = root / "src/legacy.py"
    original_source = source.read_bytes()
    source.write_bytes(original_source + b"\n# Source changed\n")
    with pytest.raises(
        ValueError, match="source bytes changed between native profiles"
    ):
        verify_required_captures(root, actual, before)
    source.write_bytes(original_source)

    with pytest.raises(ValueError, match="adopted during initial recording"):
        qualify_required_record(root, resolve_commit(root, "HEAD"), actual)

    historical = root.parent / "historical-authority"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(historical), actual.origin_commit],
        cwd=root,
        check=True,
        capture_output=True,
    )
    policy = write_policy(historical)
    try:
        with pytest.raises(ValueError, match="capture does not match original source"):
            qualify_required_record(historical, actual.origin_commit, actual)
    finally:
        policy.unlink()
        subprocess.run(
            ["git", "worktree", "remove", str(historical)],
            cwd=root,
            check=True,
            capture_output=True,
        )


def _verify_authority_changes_during_capture(
    root: Path, runtime: Path, executables: tuple[tuple[str, str], ...]
) -> None:
    """Reject both authority changes after the CLI starts native collection."""
    policy = root / POLICY_PATH
    baseline = root / BASELINE_PATH
    payload = object_mapping(json.loads(baseline.read_bytes()))
    changed_profiles = object_mapping(payload["profiles"])
    del changed_profiles["protobuf-docs"]
    payload["profiles"] = changed_profiles

    for path, changed, diagnostic in (
        (
            policy,
            policy.read_bytes() + b"\n# Changed during collection\n",
            "Required quality policy changed during analysis",
        ),
        (
            baseline,
            json.dumps(payload).encode(),
            "Required quality candidate baseline changed during analysis",
        ),
    ):
        native_temporary = runtime / f"{path.stem}-native"
        native_temporary.mkdir()
        environment = dict(os.environ, TMPDIR=str(native_temporary))
        original = path.read_bytes()
        with ThreadPoolExecutor(max_workers=1) as worker:
            pending = worker.submit(
                require_cli,
                root,
                "check",
                executables=executables,
                environment=environment,
            )
            deadline = time.monotonic() + 120
            while not any(
                native_temporary.glob("quality-rust-parser-*/.rustc_info.json")
            ):
                assert not pending.done(), (
                    f"Native check ended before Cargo ran: {pending.result().stderr}"
                )
                assert time.monotonic() < deadline, (
                    "Timed out observing the native Cargo build"
                )
                time.sleep(0.025)
            try:
                path.write_bytes(changed)
                result = pending.result()
            finally:
                path.write_bytes(original)
        assert result.returncode == 2 and diagnostic in result.stderr
        assert not result.stdout
