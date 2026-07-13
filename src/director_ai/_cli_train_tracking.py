# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Training Experiment & Registry CLI

"""CLI commands for experiment-run tracking and the trained-model registry."""

from __future__ import annotations

import json
import sys
from typing import Any


def _cmd_train_runs(args: list[str]) -> None:
    """List tracked experiment runs, optionally ranked by a metric."""
    from director_ai.core.training.experiment_tracker import ExperimentTracker

    opts = _parse_runs_args(args)
    tracker = ExperimentTracker(str(opts["dir"]))
    runs = tracker.list_runs(
        backend=_opt_str(opts, "backend"),
        state=_opt_str(opts, "state"),
    )
    payload: dict[str, Any] = {
        "runs": [run.to_dict() for run in runs],
    }
    metric = _opt_str(opts, "metric")
    if metric:
        payload["ranking"] = [
            {"run_id": run_id, metric: value}
            for run_id, value in tracker.compare(metric)
        ]
        best = tracker.best_run(metric)
        payload["best"] = best.to_dict() if best else None
    print(json.dumps(payload, indent=2, sort_keys=True))


def _parse_runs_args(args: list[str]) -> dict[str, str | None]:
    opts: dict[str, str | None] = {
        "dir": "",
        "backend": None,
        "state": None,
        "metric": None,
    }
    aliases = {
        "--dir": "dir",
        "--experiment-dir": "dir",
        "--backend": "backend",
        "--state": "state",
        "--metric": "metric",
    }
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in aliases and i + 1 < len(args):
            opts[aliases[arg]] = args[i + 1]
            i += 2
            continue
        print(f"Unknown train runs option: {arg}")
        sys.exit(1)
    if not opts["dir"]:
        print("Error: --dir is required")
        sys.exit(1)
    return opts


def _cmd_train_registry(args: list[str]) -> None:
    """Inspect or mutate the trained-model registry."""
    from director_ai.core.training.trained_model_registry import (
        TrainedModelRegistry,
    )

    opts = _parse_registry_args(args)
    registry = TrainedModelRegistry(str(opts["dir"]))
    try:
        payload = _run_registry_action(registry, opts)
    except (KeyError, ValueError) as exc:
        message = exc.args[0] if exc.args else exc
        print(f"Error: {message}")
        sys.exit(1)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _run_registry_action(
    registry: Any,
    opts: dict[str, object],
) -> dict[str, Any]:
    if opts["register"]:
        return {"registered": _register_from_run(registry, opts).to_dict()}

    model = _opt_str(opts, "model")
    version = _opt_str(opts, "version")
    if opts["promote"]:
        record = registry.promote(
            _require(model, "--model"),
            int(_require(version, "--version")),
            benchmark_evidence=_load_evidence(_require_evidence(opts)),
        )
        return {"promoted": record.to_dict()}
    if opts["retire"]:
        record = registry.retire(
            _require(model, "--model"),
            int(_require(version, "--version")),
        )
        return {"retired": record.to_dict()}
    if model and version:
        return {"model": registry.get(model, int(version)).to_dict()}
    if model:
        production = registry.production(model)
        return {
            "name": model,
            "versions": [record.to_dict() for record in registry.list_versions(model)],
            "production": production.to_dict() if production else None,
        }
    return {
        "models": {
            name: [record.to_dict() for record in registry.list_versions(name)]
            for name in registry.list_models()
        },
    }


def _register_from_run(registry: Any, opts: dict[str, object]) -> Any:
    from director_ai.core.training.dataset_fingerprint import DatasetFingerprint
    from director_ai.core.training.experiment_tracker import ExperimentTracker

    runs_dir = _opt_str(opts, "runs_dir")
    run_id = _opt_str(opts, "from_run")
    tracker = ExperimentTracker(_require(runs_dir, "--runs-dir"))
    run = tracker.get(_require(run_id, "--from-run"))
    base_model = str(run.spec.get("resolved_base_model") or run.spec["base_model"])
    return registry.register(
        name=_require(_opt_str(opts, "name"), "--name"),
        artifact_uri=_require(_opt_str(opts, "artifact"), "--artifact"),
        base_model_id=base_model,
        dataset_fingerprint=DatasetFingerprint.from_dict(run.dataset_fingerprint),
        metrics=run.metrics,
        run_id=run.run_id,
        base_model_revision=_opt_str(opts, "base_revision") or "",
        config_hash=run.config_hash,
    )


def _parse_registry_args(args: list[str]) -> dict[str, object]:
    opts: dict[str, object] = {
        "dir": "",
        "model": None,
        "version": None,
        "promote": False,
        "retire": False,
        "register": False,
        "evidence": None,
        "name": None,
        "artifact": None,
        "from_run": None,
        "runs_dir": None,
        "base_revision": None,
    }
    aliases = {
        "--dir": "dir",
        "--registry-dir": "dir",
        "--model": "model",
        "--version": "version",
        "--evidence": "evidence",
        "--name": "name",
        "--artifact": "artifact",
        "--from-run": "from_run",
        "--runs-dir": "runs_dir",
        "--base-revision": "base_revision",
    }
    flags = {
        "--promote": "promote",
        "--retire": "retire",
        "--register": "register",
    }
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in flags:
            opts[flags[arg]] = True
            i += 1
            continue
        if arg in aliases and i + 1 < len(args):
            opts[aliases[arg]] = args[i + 1]
            i += 2
            continue
        print(f"Unknown train registry option: {arg}")
        sys.exit(1)
    if not opts["dir"]:
        print("Error: --dir is required")
        sys.exit(1)
    return opts


def _load_evidence(path: str) -> dict[str, Any]:
    from pathlib import Path

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Error: cannot read benchmark evidence {path}: {exc}")
        sys.exit(1)
    if not isinstance(payload, dict):
        print(f"Error: benchmark evidence must be a JSON object: {path}")
        sys.exit(1)
    return payload


def _require_evidence(opts: dict[str, object]) -> str:
    evidence = _opt_str(opts, "evidence")
    if not evidence:
        print("Error: --promote requires --evidence <benchmark.json>")
        sys.exit(1)
    return evidence


def _require(value: str | None, option: str) -> str:
    if not value:
        print(f"Error: {option} is required")
        sys.exit(1)
    return value


def _opt_str(opts: dict[str, object] | dict[str, str | None], key: str) -> str | None:
    value = opts[key]
    return value if isinstance(value, str) else None
