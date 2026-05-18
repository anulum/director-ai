# SPDX-License-Identifier: AGPL-3.0-or-later
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
    assert calls[0]["args"][:3] == ["bash", "-lc", MODULE.LAUNCH_WRAPPER]
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
