# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 export runner tests

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_lite_scorer_v2_export.py"
SPEC = importlib.util.spec_from_file_location("run_lite_scorer_v2_export", RUNNER)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

ExportError = MODULE.ExportError
find_export_argv = MODULE.find_export_argv
run_lite_scorer_v2_export = MODULE.run_lite_scorer_v2_export


def test_find_export_argv_returns_export_onnx_command() -> None:
    plan = {
        "commands": [
            {"name": "train", "argv": ["train"]},
            {"name": "export_onnx", "argv": ["director-ai", "export"]},
        ]
    }

    assert find_export_argv(plan) == ["director-ai", "export"]


def test_export_runner_refuses_when_training_is_not_export_ready(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        MODULE,
        "inspect_lite_scorer_v2_training",
        lambda *args, **kwargs: {
            "state": "running",
            "export_ready": False,
            "run_dir": ".coordination/runs/DIRECTOR-AI/run",
        },
    )

    try:
        run_lite_scorer_v2_export(tmp_path)
    except ExportError as exc:
        assert str(exc) == (
            "Lite Scorer v2 training is not export-ready: state=running, "
            "export_ready=False"
        )
    else:
        raise AssertionError("expected ExportError")


def test_export_runner_uses_plan_command_and_verifies_onnx_artifact(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, Any]] = []
    artifact = tmp_path / "MODELS" / "lite-scorer-v2" / "onnx" / "model_quantized.onnx"

    monkeypatch.setattr(
        MODULE,
        "inspect_lite_scorer_v2_training",
        lambda *args, **kwargs: {
            "state": "completed",
            "export_ready": True,
            "run_dir": ".coordination/runs/DIRECTOR-AI/run",
        },
    )
    monkeypatch.setattr(
        MODULE,
        "build_lite_scorer_v2_run_plan",
        lambda root, manifest: (
            {
                "plan_id": "lite-scorer-v2-run-plan",
                "commands": [
                    {
                        "name": "export_onnx",
                        "argv": ["uv", "run", "--frozen", "director-ai", "export"],
                    }
                ],
                "generated_paths": {
                    "onnx_artifact": "MODELS/lite-scorer-v2/onnx/model_quantized.onnx"
                },
            },
            [],
        ),
    )

    def fake_run(
        argv: list[str],
        *,
        cwd: Path,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append({"argv": argv, "cwd": cwd, "check": check})
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"onnx-bytes")
        return subprocess.CompletedProcess(argv, 0)

    result = run_lite_scorer_v2_export(tmp_path, run_command=fake_run)

    assert calls == [
        {
            "argv": ["uv", "run", "--frozen", "director-ai", "export"],
            "cwd": tmp_path.resolve(),
            "check": True,
        }
    ]
    assert result == {
        "schema_version": "1.0.0",
        "command": ["uv", "run", "--frozen", "director-ai", "export"],
        "onnx_artifact": {
            "path": "MODELS/lite-scorer-v2/onnx/model_quantized.onnx",
            "exists": True,
            "size_bytes": len(b"onnx-bytes"),
        },
        "public_score_claim": False,
        "status": "recorded",
    }


def test_export_runner_fails_when_command_does_not_create_artifact(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        MODULE,
        "inspect_lite_scorer_v2_training",
        lambda *args, **kwargs: {
            "state": "completed",
            "export_ready": True,
            "run_dir": ".coordination/runs/DIRECTOR-AI/run",
        },
    )
    monkeypatch.setattr(
        MODULE,
        "build_lite_scorer_v2_run_plan",
        lambda root, manifest: (
            {
                "plan_id": "lite-scorer-v2-run-plan",
                "commands": [{"name": "export_onnx", "argv": ["export"]}],
                "generated_paths": {"onnx_artifact": "MODELS/missing.onnx"},
            },
            [],
        ),
    )

    try:
        run_lite_scorer_v2_export(
            tmp_path,
            run_command=lambda argv, *, cwd, check: subprocess.CompletedProcess(
                argv,
                0,
            ),
        )
    except ExportError as exc:
        assert str(exc) == (
            "ONNX export command completed but MODELS/missing.onnx is missing"
        )
    else:
        raise AssertionError("expected ExportError")
