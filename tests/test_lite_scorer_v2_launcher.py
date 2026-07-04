# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 launcher tests

from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools" / "launch_lite_scorer_v2_training.py"
SPEC = importlib.util.spec_from_file_location(
    "launch_lite_scorer_v2_training", LAUNCHER
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

LaunchError = MODULE.LaunchError
find_train_argv = MODULE.find_train_argv
launch_lite_scorer_v2_training = MODULE.launch_lite_scorer_v2_training
resolve_lite_scorer_v2_run_root = MODULE.resolve_lite_scorer_v2_run_root


def test_find_train_argv_returns_only_the_planned_training_command() -> None:
    plan = {
        "commands": [
            {"name": "build_heldout", "argv": ["build"]},
            {"name": "train", "argv": ["uv", "run", "python", "train.py"]},
            {"name": "evaluate", "argv": ["eval"]},
        ]
    }

    assert find_train_argv(plan) == ["uv", "run", "python", "train.py"]


def test_find_train_argv_rejects_missing_training_command() -> None:
    plan = {"commands": [{"name": "evaluate", "argv": ["eval"]}]}

    try:
        find_train_argv(plan)
    except LaunchError as exc:
        assert str(exc) == "run plan does not contain a train command"
    else:
        raise AssertionError("expected LaunchError")


def test_find_train_argv_rejects_malformed_training_commands() -> None:
    """Malformed run plans should fail before any subprocess launch."""
    cases = [
        (
            {"commands": "train"},
            "run plan does not contain a commands list",
        ),
        (
            {"commands": [{"name": "train", "argv": [""]}]},
            "run plan train command must be a non-empty argv list",
        ),
    ]

    for plan, message in cases:
        try:
            find_train_argv(plan)
        except LaunchError as exc:
            assert str(exc) == message
        else:
            raise AssertionError("expected LaunchError")


def test_launcher_module_rejects_missing_planner_spec(monkeypatch: Any) -> None:
    """The script should fail closed when its planner module cannot be loaded."""
    original_spec_from_file_location = importlib.util.spec_from_file_location
    launcher_spec = original_spec_from_file_location(
        "launch_lite_scorer_v2_training_missing_planner",
        LAUNCHER,
    )
    assert launcher_spec is not None
    assert launcher_spec.loader is not None

    def fake_spec_from_file_location(
        name: str,
        location: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if name == "plan_lite_scorer_v2_run":
            return None
        return original_spec_from_file_location(name, location, *args, **kwargs)

    monkeypatch.setattr(
        importlib.util,
        "spec_from_file_location",
        fake_spec_from_file_location,
    )
    module = importlib.util.module_from_spec(launcher_spec)

    try:
        launcher_spec.loader.exec_module(module)
    except RuntimeError as exc:
        assert "cannot load Lite Scorer v2 planner" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_launcher_process_probe_handles_kernel_outcomes(monkeypatch: Any) -> None:
    """Process probing should handle missing, denied, and visible PIDs."""
    assert MODULE.is_process_running(0) is False

    def missing_process(_pid: int, _signal: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(MODULE.os, "kill", missing_process)
    assert MODULE.is_process_running(101) is False

    def denied_process(_pid: int, _signal: int) -> None:
        raise PermissionError

    monkeypatch.setattr(MODULE.os, "kill", denied_process)
    assert MODULE.is_process_running(101) is True

    def visible_process(_pid: int, _signal: int) -> None:
        return None

    monkeypatch.setattr(MODULE.os, "kill", visible_process)
    assert MODULE.is_process_running(101) is True


def test_launcher_run_root_falls_back_to_local_coordination(tmp_path: Path) -> None:
    """Non-code-repo roots should keep durable state under the provided root."""
    assert resolve_lite_scorer_v2_run_root(tmp_path) == (
        tmp_path.resolve() / ".coordination" / "runs" / "DIRECTOR-AI"
    )


def test_launcher_code_repo_without_monorepo_coordination_falls_back(
    tmp_path: Path,
) -> None:
    """Code-repo-shaped temp roots without monorepo state should stay local."""
    repo = tmp_path / "aaa_God_of_the_Math_Collection" / "03_CODE" / "DIRECTOR-AI"
    repo.mkdir(parents=True)

    assert resolve_lite_scorer_v2_run_root(repo) == (
        repo.resolve() / ".coordination" / "runs" / "DIRECTOR-AI"
    )


def test_launcher_active_run_scan_ignores_incomplete_pid_files(tmp_path: Path) -> None:
    """Active-run discovery should ignore pidless and malformed run directories."""
    run_root = tmp_path / ".coordination" / "runs" / "DIRECTOR-AI"
    (run_root / "lite_scorer_v2_train_no_pid").mkdir(parents=True)
    invalid = run_root / "lite_scorer_v2_train_invalid_pid"
    invalid.mkdir()
    (invalid / "pid").write_text("not-a-pid\n", encoding="utf-8")
    live = run_root / "lite_scorer_v2_train_live"
    live.mkdir()
    (live / "pid").write_text("2026\n", encoding="utf-8")

    assert MODULE._active_training_runs(
        tmp_path,
        lambda pid: pid == 2026,
    ) == [(live, 2026)]

    assert (
        MODULE._active_training_runs(
            tmp_path,
            lambda _pid: False,
        )
        == []
    )


def test_launcher_writes_metadata_and_starts_new_process_session(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeProcess:
        pid = 424242

    def fake_plan(root: Path, manifest: Path) -> tuple[dict[str, Any], list[str]]:
        assert root == tmp_path.resolve()
        assert manifest == tmp_path / "manifest.toml"
        return {
            "schema_version": "1.0.0",
            "plan_id": "lite-scorer-v2-run-plan",
            "commands": [
                {
                    "name": "train",
                    "argv": ["uv", "run", "--frozen", "python", "train.py"],
                }
            ],
        }, []

    def fake_popen(
        args: list[str],
        *,
        cwd: Path,
        stdin: Any,
        stdout: Any,
        stderr: int,
        start_new_session: bool,
    ) -> FakeProcess:
        calls.append(
            {
                "args": args,
                "cwd": cwd,
                "stdin": stdin,
                "stdout": stdout,
                "stderr": stderr,
                "start_new_session": start_new_session,
            }
        )
        return FakeProcess()

    monkeypatch.setattr(MODULE, "build_lite_scorer_v2_run_plan", fake_plan)
    monkeypatch.setattr(MODULE.os, "getsid", lambda pid: 919191)

    result = launch_lite_scorer_v2_training(
        tmp_path,
        manifest=tmp_path / "manifest.toml",
        timestamp="2026-05-18T024000",
        popen=fake_popen,
        process_running=lambda _pid: False,
    )

    run_dir = (
        tmp_path
        / ".coordination"
        / "runs"
        / "DIRECTOR-AI"
        / ("lite_scorer_v2_train_2026-05-18T024000")
    )
    assert result == {
        "run_dir": run_dir.as_posix(),
        "pid": 424242,
        "session_id": 919191,
        "log": (run_dir / "train.log").as_posix(),
    }
    assert calls[0]["cwd"] == tmp_path.resolve()
    assert calls[0]["start_new_session"] is True
    assert calls[0]["args"][:3] == ["bash", "-c", MODULE.LAUNCH_WRAPPER]
    assert calls[0]["args"][-4:] == ["uv", "run", "--frozen", "python", "train.py"][-4:]
    assert (run_dir / "pid").read_text(encoding="utf-8") == "424242\n"
    assert json.loads((run_dir / "metadata.json").read_text(encoding="utf-8")) == {
        "command": ["uv", "run", "--frozen", "python", "train.py"],
        "pid": 424242,
        "plan_id": "lite-scorer-v2-run-plan",
        "public_score_claim": False,
        "run_id": "lite_scorer_v2_train_2026-05-18T024000",
        "session_id": 919191,
    }
    assert (run_dir / "command.txt").read_text(encoding="utf-8") == (
        "uv run --frozen python train.py\n"
    )


def test_launcher_uses_generated_timestamp_and_records_missing_session(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Launch metadata should tolerate processes gone before session probing."""

    class FakeProcess:
        pid = 313131

    def fake_plan(_root: Path, _manifest: Path) -> tuple[dict[str, Any], list[str]]:
        return {
            "schema_version": "1.0.0",
            "plan_id": "lite-scorer-v2-run-plan",
            "commands": [
                {"name": "train", "argv": ["uv", "run", "python", "train.py"]}
            ],
        }, []

    def fake_popen(
        _args: list[str],
        *,
        cwd: Path,
        stdin: Any,
        stdout: Any,
        stderr: int,
        start_new_session: bool,
    ) -> FakeProcess:
        assert cwd == tmp_path.resolve()
        assert stdin == MODULE.subprocess.DEVNULL
        assert stdout is not None
        assert stderr == MODULE.subprocess.STDOUT
        assert start_new_session is True
        return FakeProcess()

    def missing_session(_pid: int) -> int:
        raise ProcessLookupError

    monkeypatch.setattr(MODULE, "build_lite_scorer_v2_run_plan", fake_plan)
    monkeypatch.setattr(MODULE.os, "getsid", missing_session)

    result = launch_lite_scorer_v2_training(
        tmp_path,
        manifest=Path("manifest.toml"),
        popen=fake_popen,
        process_running=lambda _pid: False,
    )

    run_dir = Path(result["run_dir"])
    assert run_dir.name.startswith("lite_scorer_v2_train_")
    assert result["session_id"] is None
    assert (run_dir / "metadata.json").is_file()


def test_launcher_rejects_existing_live_training_pid(tmp_path: Path) -> None:
    live_dir = (
        tmp_path
        / ".coordination"
        / "runs"
        / "DIRECTOR-AI"
        / ("lite_scorer_v2_train_2026-05-18T010000")
    )
    live_dir.mkdir(parents=True)
    (live_dir / "pid").write_text("101\n", encoding="utf-8")

    try:
        launch_lite_scorer_v2_training(
            tmp_path,
            timestamp="2026-05-18T024500",
            process_running=lambda pid: pid == 101,
        )
    except LaunchError as exc:
        assert str(exc) == (
            "active Lite Scorer v2 training run already exists: "
            f"{live_dir.as_posix()} (pid 101)"
        )
    else:
        raise AssertionError("expected LaunchError")


def test_launcher_rejects_planner_errors(tmp_path: Path, monkeypatch: Any) -> None:
    """Planner validation errors should stop the durable launcher."""

    def fake_plan(_root: Path, _manifest: Path) -> tuple[dict[str, Any], list[str]]:
        return {}, ["missing teacher artifact", "missing student base"]

    monkeypatch.setattr(MODULE, "build_lite_scorer_v2_run_plan", fake_plan)

    try:
        launch_lite_scorer_v2_training(
            tmp_path,
            manifest=Path("manifest.toml"),
            process_running=lambda _pid: False,
        )
    except LaunchError as exc:
        assert str(exc) == "missing teacher artifact; missing student base"
    else:
        raise AssertionError("expected LaunchError")


def test_launcher_rejects_existing_run_directory(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A deterministic timestamp must not overwrite an existing run directory."""
    run_dir = (
        tmp_path
        / ".coordination"
        / "runs"
        / "DIRECTOR-AI"
        / "lite_scorer_v2_train_2026-07-04T123000"
    )
    run_dir.mkdir(parents=True)

    def fake_plan(_root: Path, _manifest: Path) -> tuple[dict[str, Any], list[str]]:
        return {
            "schema_version": "1.0.0",
            "plan_id": "lite-scorer-v2-run-plan",
            "commands": [
                {"name": "train", "argv": ["uv", "run", "python", "train.py"]}
            ],
        }, []

    monkeypatch.setattr(MODULE, "build_lite_scorer_v2_run_plan", fake_plan)

    try:
        launch_lite_scorer_v2_training(
            tmp_path,
            manifest=Path("manifest.toml"),
            timestamp="2026-07-04T123000",
            process_running=lambda _pid: False,
        )
    except LaunchError as exc:
        assert str(exc) == f"run directory already exists: {run_dir.as_posix()}"
    else:
        raise AssertionError("expected LaunchError")


def test_launcher_run_root_uses_monorepo_coordination_for_code_repo(
    tmp_path: Path,
) -> None:
    """Real Samsung-style code checkouts should use monorepo coordination."""
    monorepo = tmp_path / "aaa_God_of_the_Math_Collection"
    repo = monorepo / "03_CODE" / "DIRECTOR-AI"
    repo.mkdir(parents=True)
    (monorepo / ".coordination").mkdir()

    assert resolve_lite_scorer_v2_run_root(repo) == (
        monorepo / ".coordination" / "runs" / "DIRECTOR-AI"
    )


def test_launcher_main_prints_json_on_success(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """The CLI entrypoint should emit the launcher result as JSON."""

    def fake_launch(
        root: Path,
        *,
        manifest: Path,
        timestamp: str | None,
    ) -> dict[str, object]:
        assert root == tmp_path
        assert manifest == tmp_path / "manifest.toml"
        assert timestamp == "2026-07-04T124000"
        return {"log": "train.log", "pid": 7, "run_dir": "run", "session_id": None}

    monkeypatch.setattr(MODULE, "launch_lite_scorer_v2_training", fake_launch)
    stdout = StringIO()

    with redirect_stdout(stdout):
        exit_code = MODULE.main(
            [
                str(tmp_path),
                "--manifest",
                str(tmp_path / "manifest.toml"),
                "--timestamp",
                "2026-07-04T124000",
            ]
        )

    assert exit_code == 0
    assert json.loads(stdout.getvalue()) == {
        "log": "train.log",
        "pid": 7,
        "run_dir": "run",
        "session_id": None,
    }


def test_launcher_main_reports_launch_errors(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """The CLI entrypoint should write launcher failures to stderr."""

    def fake_launch(
        _root: Path,
        *,
        manifest: Path,
        timestamp: str | None,
    ) -> dict[str, object]:
        assert manifest == MODULE.MANIFEST
        assert timestamp is None
        raise LaunchError("launch rejected")

    monkeypatch.setattr(MODULE, "launch_lite_scorer_v2_training", fake_launch)
    stderr = StringIO()

    with redirect_stderr(stderr):
        exit_code = MODULE.main([str(tmp_path)])

    assert exit_code == 1
    assert stderr.getvalue() == "launch rejected\n"
