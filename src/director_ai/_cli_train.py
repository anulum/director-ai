# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Managed Training CLI

"""CLI commands for managed local and cloud training jobs."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast


def _cmd_train(args: list[str]) -> None:
    """Dispatch managed training commands."""

    if not args or args[0] in ("-h", "--help", "help"):
        _print_train_help()
        return

    subcommand = args[0]
    if subcommand == "submit":
        _cmd_train_submit(args[1:])
        return
    if subcommand == "models":
        _cmd_train_models(args[1:])
        return
    if subcommand == "benchmark-models":
        _cmd_train_benchmark_models(args[1:])
        return
    if subcommand == "sweep":
        _cmd_train_sweep(args[1:])
        return
    if subcommand == "harvest":
        _cmd_train_harvest(args[1:])
        return

    print(f"Unknown train subcommand: {subcommand}")
    _print_train_help()
    sys.exit(1)


def _cmd_train_submit(args: list[str]) -> None:
    """Submit or dry-run a managed training job."""

    from director_ai.core.training.jobs import (
        TrainingHardware,
        TrainingJobSpec,
        build_internal_suite_spec,
        shell_join,
        submission_to_json,
        submit_training_job,
    )

    opts = _parse_submit_args(args)
    backend = _as_str(opts, "backend")
    caller = _as_str(opts, "caller")
    dataset_uri = _as_str(opts, "dataset_uri")
    output_uri = _as_str(opts, "output_uri")
    project = _as_optional_str(opts, "project")
    region = _as_str(opts, "region")
    image = _as_str(opts, "image")
    suite = _as_str(opts, "suite")
    hardware = TrainingHardware(
        machine_type=_as_str(opts, "machine"),
        accelerator_type=_as_str(opts, "gpu"),
        accelerator_count=int(_as_str(opts, "gpu_count")),
        boot_disk_gb=int(_as_str(opts, "boot_disk_gb")),
    )

    if suite:
        spec = build_internal_suite_spec(
            suite=suite,
            dataset_uri=dataset_uri,
            output_uri=output_uri,
            project=project,
            region=region,
            container_image_uri=image,
            hardware=hardware,
        )
    else:
        spec = TrainingJobSpec(
            display_name=_as_str(opts, "display_name"),
            caller=caller,
            dataset_uri=dataset_uri,
            output_uri=output_uri,
            eval_uri=_as_optional_str(opts, "eval_uri"),
            project=project,
            region=region,
            base_model=_as_str(opts, "base_model"),
            allow_experimental_model=bool(opts["allow_experimental_model"]),
            epochs=int(_as_str(opts, "epochs")),
            batch_size=int(_as_str(opts, "batch_size")),
            learning_rate=float(_as_str(opts, "learning_rate")),
            timeout_minutes=int(_as_str(opts, "timeout_minutes")),
            container_image_uri=image,
            service_account=_as_optional_str(opts, "service_account"),
            network=_as_optional_str(opts, "network"),
            hardware=hardware,
        )

    dry_run = not opts["execute"]
    try:
        submission = submit_training_job(spec, backend=backend, dry_run=bool(dry_run))
    except Exception as exc:
        print(f"Error: training job submission failed: {exc}")
        sys.exit(1)
    print(submission_to_json(submission))
    command = submission.request.get("command")
    if command:
        print(f"\nCommand: {shell_join(command)}")


def _parse_submit_args(args: list[str]) -> dict[str, str | bool | None]:
    opts: dict[str, str | bool | None] = {
        "backend": "vertex",
        "caller": "product",
        "display_name": "director-ai-managed-training",
        "dataset_uri": "",
        "eval_uri": None,
        "output_uri": "",
        "project": None,
        "region": "us-central1",
        "base_model": "yaxili96/FactCG-DeBERTa-v3-Large",
        "allow_experimental_model": False,
        "epochs": "3",
        "batch_size": "16",
        "learning_rate": "2e-5",
        "timeout_minutes": "180",
        "image": "python:3.12-slim",
        "service_account": None,
        "network": None,
        "machine": "g2-standard-8",
        "gpu": "NVIDIA_L4",
        "gpu_count": "1",
        "boot_disk_gb": "100",
        "suite": "",
        "execute": False,
    }
    aliases = {
        "--backend": "backend",
        "--caller": "caller",
        "--display-name": "display_name",
        "--dataset-uri": "dataset_uri",
        "--train-uri": "dataset_uri",
        "--eval-uri": "eval_uri",
        "--output-uri": "output_uri",
        "--project": "project",
        "--region": "region",
        "--base-model": "base_model",
        "--model": "base_model",
        "--epochs": "epochs",
        "--batch-size": "batch_size",
        "--lr": "learning_rate",
        "--timeout-min": "timeout_minutes",
        "--image": "image",
        "--service-account": "service_account",
        "--network": "network",
        "--machine": "machine",
        "--gpu": "gpu",
        "--gpu-count": "gpu_count",
        "--boot-disk-gb": "boot_disk_gb",
        "--suite": "suite",
    }

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--execute":
            opts["execute"] = True
            i += 1
            continue
        if arg == "--dry-run":
            opts["execute"] = False
            i += 1
            continue
        if arg == "--allow-experimental-model":
            opts["allow_experimental_model"] = True
            i += 1
            continue
        if arg in aliases and i + 1 < len(args):
            opts[aliases[arg]] = args[i + 1]
            i += 2
            continue
        print(f"Unknown or incomplete option: {arg}")
        sys.exit(1)

    if not opts["dataset_uri"]:
        print("Error: --dataset-uri is required")
        sys.exit(1)
    if not opts["output_uri"]:
        print("Error: --output-uri is required")
        sys.exit(1)
    return opts


def _cmd_train_sweep(args: list[str]) -> None:
    """Plan or submit a managed fine-tuning scenario sweep."""

    import json

    from director_ai.core.training.jobs import (
        TrainingHardware,
        submission_to_json,
        submit_training_job,
    )
    from director_ai.core.training.sweeps import (
        TrainingDatasetSplit,
        build_training_sweep_plan,
    )

    opts = _parse_sweep_args(args)
    train_sets = cast(dict[str, str], opts["train_sets"])
    eval_sets = cast(dict[str, str], opts["eval_sets"])
    models = cast(list[str], opts["models"])
    epochs = cast(list[str], opts["epochs"])
    batch_sizes = cast(list[str], opts["batch_sizes"])
    datasets = [
        TrainingDatasetSplit(
            name=name,
            train_uri=train_uri,
            eval_uri=eval_sets.get(name),
        )
        for name, train_uri in train_sets.items()
    ]
    hardware = TrainingHardware(
        machine_type=_as_str(opts, "machine"),
        accelerator_type=_as_str(opts, "gpu"),
        accelerator_count=int(_as_str(opts, "gpu_count")),
        boot_disk_gb=int(_as_str(opts, "boot_disk_gb")),
    )
    try:
        plan = build_training_sweep_plan(
            sweep_id=_as_str(opts, "sweep_id"),
            datasets=datasets,
            base_models=models,
            epochs=[int(value) for value in epochs],
            batch_sizes=[int(value) for value in batch_sizes],
            learning_rate=float(_as_str(opts, "learning_rate"))
            if opts["learning_rate"]
            else None,
            output_prefix=_as_str(opts, "output_prefix"),
            allow_experimental_model=bool(opts["allow_experimental_model"]),
            display_prefix=_as_str(opts, "display_prefix"),
        )
        specs = plan.to_specs(
            project=_as_str(opts, "project"),
            region=_as_str(opts, "region"),
            container_image_uri=_as_str(opts, "image"),
            hardware=hardware,
            timeout_minutes=int(_as_str(opts, "timeout_minutes")),
            caller=_as_str(opts, "caller"),
            allow_experimental_model=bool(opts["allow_experimental_model"]),
            service_account=_as_optional_str(opts, "service_account"),
            network=_as_optional_str(opts, "network"),
        )
    except Exception as exc:
        print(f"Error: sweep planning failed: {exc}")
        sys.exit(1)
    limit = int(_as_str(opts, "limit"))
    if limit and len(specs) > limit:
        print(f"Error: sweep has {len(specs)} jobs, above --limit {limit}")
        sys.exit(1)

    dry_run = not opts["execute"]
    submissions = []
    for spec in specs:
        try:
            submission = submit_training_job(
                spec,
                backend="vertex",
                dry_run=bool(dry_run),
            )
        except Exception as exc:
            print(f"Error: sweep job submission failed for {spec.display_name}: {exc}")
            sys.exit(1)
        submissions.append(json.loads(submission_to_json(submission)))

    print(
        json.dumps(
            {
                "dry_run": dry_run,
                "plan": plan.to_dict(),
                "submissions": submissions,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _parse_sweep_args(args: list[str]) -> dict[str, object]:
    opts: dict[str, object] = {
        "sweep_id": "managed-sweep",
        "display_prefix": "director-ai-managed-sweep",
        "caller": "internal",
        "train_sets": {},
        "eval_sets": {},
        "models": [],
        "epochs": [],
        "batch_sizes": [],
        "learning_rate": None,
        "output_prefix": "",
        "project": "",
        "region": "europe-west4",
        "image": "",
        "timeout_minutes": "90",
        "service_account": None,
        "network": None,
        "machine": "n1-standard-8",
        "gpu": "NVIDIA_T4",
        "gpu_count": "1",
        "boot_disk_gb": "100",
        "allow_experimental_model": False,
        "execute": False,
        "limit": "0",
    }
    train_sets: dict[str, str] = {}
    eval_sets: dict[str, str] = {}
    models: list[str] = []
    epochs: list[str] = []
    batch_sizes: list[str] = []
    value_options = {
        "--sweep-id": "sweep_id",
        "--display-prefix": "display_prefix",
        "--caller": "caller",
        "--lr": "learning_rate",
        "--output-prefix": "output_prefix",
        "--project": "project",
        "--region": "region",
        "--image": "image",
        "--timeout-min": "timeout_minutes",
        "--service-account": "service_account",
        "--network": "network",
        "--machine": "machine",
        "--gpu": "gpu",
        "--gpu-count": "gpu_count",
        "--boot-disk-gb": "boot_disk_gb",
        "--limit": "limit",
    }

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--execute":
            opts["execute"] = True
            i += 1
            continue
        if arg == "--dry-run":
            opts["execute"] = False
            i += 1
            continue
        if arg == "--allow-experimental-model":
            opts["allow_experimental_model"] = True
            i += 1
            continue
        if arg in ("--train-set", "--eval-set", "--model", "--epochs", "--batch-size"):
            if i + 1 >= len(args):
                print(f"Unknown or incomplete option: {arg}")
                sys.exit(1)
            value = args[i + 1]
            if arg == "--train-set":
                name, uri = _split_named_uri("--train-set", value)
                train_sets[name] = uri
            elif arg == "--eval-set":
                name, uri = _split_named_uri("--eval-set", value)
                eval_sets[name] = uri
            elif arg == "--model":
                models.append(value)
            elif arg == "--epochs":
                epochs.append(value)
            else:
                batch_sizes.append(value)
            i += 2
            continue
        if arg in value_options and i + 1 < len(args):
            opts[value_options[arg]] = args[i + 1]
            i += 2
            continue
        print(f"Unknown train sweep option: {arg}")
        sys.exit(1)

    if not train_sets:
        print("Error: at least one --train-set name=uri is required")
        sys.exit(1)
    if not models:
        print("Error: at least one --model is required")
        sys.exit(1)
    if not epochs:
        print("Error: at least one --epochs value is required")
        sys.exit(1)
    if not batch_sizes:
        batch_sizes.append("1")
    if not opts["output_prefix"]:
        print("Error: --output-prefix is required")
        sys.exit(1)
    if not opts["project"]:
        print("Error: --project is required")
        sys.exit(1)
    if not opts["image"]:
        print("Error: --image is required")
        sys.exit(1)

    opts["train_sets"] = train_sets
    opts["eval_sets"] = eval_sets
    opts["models"] = models
    opts["epochs"] = epochs
    opts["batch_sizes"] = batch_sizes
    return opts


def _split_named_uri(option: str, value: str) -> tuple[str, str]:
    if "=" not in value:
        print(f"Error: {option} must use name=uri")
        sys.exit(1)
    name, uri = value.split("=", 1)
    if not name or not uri:
        print(f"Error: {option} must use name=uri")
        sys.exit(1)
    return name, uri


def _cmd_train_models(args: list[str]) -> None:
    """Print the fine-tune model registry as JSON."""

    import json

    from director_ai.core.training.model_registry import (
        finetune_model_registry_to_dict,
    )

    include_experimental = "--include-experimental" in args
    unknown = [arg for arg in args if arg != "--include-experimental"]
    if unknown:
        print(f"Unknown train models option: {unknown[0]}")
        sys.exit(1)
    print(
        json.dumps(
            {
                "models": finetune_model_registry_to_dict(
                    include_experimental=include_experimental,
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _cmd_train_benchmark_models(args: list[str]) -> None:
    """Benchmark trained model artifacts with identical inputs."""

    import json

    from director_ai.core.training.finetune_benchmark import (
        benchmark_model_candidates,
    )

    opts = _parse_benchmark_models_args(args)
    models = cast(dict[str, str], opts["models"])
    general_path = cast(str | Path | None, opts["general_path"])
    eval_path = cast(str | Path | None, opts["eval_path"])
    batch_size = cast(str | None, opts["batch_size"])
    report = benchmark_model_candidates(
        models,
        general_path=general_path,
        eval_path=eval_path,
        batch_size=int(batch_size) if batch_size else None,
        allow_experimental=bool(opts["allow_experimental_model"]),
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


def _cmd_train_harvest(args: list[str]) -> None:
    """Collect managed training result artifacts from a sweep prefix."""

    import json

    from director_ai.core.training.results import harvest_training_results

    opts = _parse_harvest_args(args)
    try:
        report = harvest_training_results(_as_str(opts, "prefix_uri"))
    except Exception as exc:
        print(f"Error: training result harvest failed: {exc}")
        sys.exit(1)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


def _parse_harvest_args(args: list[str]) -> dict[str, object]:
    opts: dict[str, object] = {"prefix_uri": ""}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--prefix-uri", "--sweep-prefix", "--output-prefix"):
            if i + 1 >= len(args):
                print(f"Unknown or incomplete option: {arg}")
                sys.exit(1)
            opts["prefix_uri"] = args[i + 1]
            i += 2
            continue
        print(f"Unknown train harvest option: {arg}")
        sys.exit(1)
    if not opts["prefix_uri"]:
        print("Error: --prefix-uri is required")
        sys.exit(1)
    return opts


def _parse_benchmark_models_args(args: list[str]) -> dict[str, object]:
    opts: dict[str, object] = {
        "models": {},
        "general_path": None,
        "eval_path": None,
        "batch_size": None,
        "allow_experimental_model": False,
    }
    models: dict[str, str] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--allow-experimental-model":
            opts["allow_experimental_model"] = True
            i += 1
            continue
        if arg in ("--model", "--general-uri", "--eval-uri", "--batch-size"):
            if i + 1 >= len(args):
                print(f"Unknown or incomplete option: {arg}")
                sys.exit(1)
            value = args[i + 1]
            if arg == "--model":
                if "=" not in value:
                    print("Error: --model must use alias=artifact_path")
                    sys.exit(1)
                alias, model_path = value.split("=", 1)
                if not alias or not model_path:
                    print("Error: --model must use alias=artifact_path")
                    sys.exit(1)
                models[alias] = model_path
            elif arg == "--general-uri":
                opts["general_path"] = value
            elif arg == "--eval-uri":
                opts["eval_path"] = value
            else:
                opts["batch_size"] = value
            i += 2
            continue
        print(f"Unknown train benchmark-models option: {arg}")
        sys.exit(1)

    if not models:
        print("Error: at least one --model alias=artifact_path is required")
        sys.exit(1)
    opts["models"] = models
    return opts


def _as_str(opts: Mapping[str, object], key: str) -> str:
    value = opts[key]
    if not isinstance(value, str):
        print(f"Error: {key} must be a string")
        sys.exit(1)
    return value


def _as_optional_str(opts: Mapping[str, object], key: str) -> str | None:
    value = opts[key]
    if value is None or isinstance(value, str):
        return value
    print(f"Error: {key} must be a string")
    sys.exit(1)


def _print_train_help() -> None:
    print(
        "Usage: director-ai train <subcommand> [options]\n"
        "\n"
        "Subcommands:\n"
        "  submit   Submit or dry-run a managed training job\n"
        "  models   List selectable fine-tune base models\n"
        "  benchmark-models   Benchmark trained artifacts for model choice\n"
        "  sweep    Plan or submit a managed training scenario matrix\n"
        "  harvest  Collect training_result.json artifacts from a sweep prefix\n"
        "\n"
        "Submit options:\n"
        "  --backend local|portable|vertex  Backend (default: vertex)\n"
        "  --dataset-uri URI            Training data URI or local path\n"
        "  --output-uri URI             Artifact output URI or local directory\n"
        "  --project ID                 Cloud project for vertex backend\n"
        "  --image URI                  Training container image\n"
        "  --suite NAME                 Internal pytest suite instead of fine-tune\n"
        "  --model ALIAS                Registry alias or explicit model id\n"
        "  --allow-experimental-model   Permit experimental/custom model ids\n"
        "  --execute                    Provision the job; default is dry-run\n",
    )
