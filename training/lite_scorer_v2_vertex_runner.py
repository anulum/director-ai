#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 Vertex runner
"""Vertex AI entrypoint for Lite Scorer v2 distillation.

Large datasets and model checkpoints are materialised from GCS at runtime. This
keeps the Cloud Build context small and leaves a durable GCS output prefix for
the export and evaluation gates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass
class LiteScorerV2VertexConfig:
    data_uri: str
    teacher_uri: str
    student_uri: str
    output_uri: str
    work_dir: Path
    epochs: int
    batch_size: int
    max_length: int
    temperature: float
    alpha: float
    learning_rate: float
    seed: int
    eval_limit: int
    num_workers: int
    device: str
    summ_target: int
    general_target: int


def _is_gcs_uri(uri: str) -> bool:
    parsed = urlparse(uri)
    return (
        parsed.scheme == "gs" and bool(parsed.netloc) and bool(parsed.path.strip("/"))
    )


def _gcs_prefix_overlaps(left_uri: str, right_uri: str) -> bool:
    left_bucket, left_prefix = _split_gcs_uri(left_uri)
    right_bucket, right_prefix = _split_gcs_uri(right_uri)
    if left_bucket != right_bucket:
        return False
    return (
        left_prefix == right_prefix
        or left_prefix.startswith(f"{right_prefix}/")
        or right_prefix.startswith(f"{left_prefix}/")
    )


def validate_config(config: LiteScorerV2VertexConfig) -> list[str]:
    errors: list[str] = []
    for field in ("data_uri", "teacher_uri", "student_uri", "output_uri"):
        if not _is_gcs_uri(str(getattr(config, field))):
            errors.append(f"{field} must be a gs:// URI")
    if not any(error.endswith("must be a gs:// URI") for error in errors):
        for field in ("data_uri", "teacher_uri", "student_uri"):
            if _gcs_prefix_overlaps(config.output_uri, str(getattr(config, field))):
                errors.append(f"output_uri must not overlap {field}")
    if config.epochs <= 0:
        errors.append("epochs must be positive")
    if config.batch_size <= 0:
        errors.append("batch_size must be positive")
    if config.max_length <= 0:
        errors.append("max_length must be positive")
    if config.temperature <= 0.0:
        errors.append("temperature must be positive")
    if config.alpha <= 0.0 or config.alpha > 1.0:
        errors.append("alpha must be in (0, 1]")
    if config.learning_rate <= 0.0:
        errors.append("lr must be positive")
    if config.seed < 0:
        errors.append("seed must be non-negative")
    if config.eval_limit <= 0:
        errors.append("eval_limit must be positive")
    if config.num_workers < 0:
        errors.append("num_workers must be non-negative")
    if config.summ_target <= 0:
        errors.append("summ_target must be positive")
    if config.general_target <= 0:
        errors.append("general_target must be positive")
    if config.device not in {"auto", "cpu", "cuda"}:
        errors.append("device must be one of auto, cpu, or cuda")
    return errors


def _split_gcs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"not a GCS URI: {uri}")
    prefix = parsed.path.lstrip("/")
    if not prefix:
        raise ValueError(f"GCS URI must include an object prefix: {uri}")
    return parsed.netloc, prefix.rstrip("/")


def _relative_blob_path(prefix: str, blob_name: str) -> Path | None:
    normalized_prefix = prefix.rstrip("/")
    if not blob_name.startswith(f"{normalized_prefix}/"):
        return None
    relative = blob_name.removeprefix(f"{normalized_prefix}/")
    if not relative:
        return None
    return Path(relative)


def download_prefix(uri: str, destination: Path) -> None:
    from google.cloud import storage

    bucket_name, prefix = _split_gcs_uri(uri)
    client = storage.Client()
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    files = [
        (blob, relative)
        for blob in blobs
        if not blob.name.endswith("/")
        for relative in [_relative_blob_path(prefix, blob.name)]
        if relative is not None
    ]
    if not files:
        raise FileNotFoundError(f"no files found under {uri}")

    destination.mkdir(parents=True, exist_ok=True)
    for blob, relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(target)


def upload_prefix(source: Path, uri: str) -> None:
    from google.cloud import storage

    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {source}")
    bucket_name, prefix = _split_gcs_uri(uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        name = f"{prefix}/{path.relative_to(source).as_posix()}"
        bucket.blob(name).upload_from_filename(path)


def build_train_argv(
    config: LiteScorerV2VertexConfig,
    *,
    data_dir: Path,
    teacher_dir: Path,
    student_dir: Path,
    output_dir: Path,
) -> list[str]:
    _ = data_dir
    return [
        "--teacher",
        teacher_dir.as_posix(),
        "--student",
        student_dir.as_posix(),
        "--epochs",
        str(config.epochs),
        "--batch-size",
        str(config.batch_size),
        "--max-length",
        str(config.max_length),
        "--temperature",
        str(config.temperature),
        "--alpha",
        str(config.alpha),
        "--lr",
        str(config.learning_rate),
        "--seed",
        str(config.seed),
        "--eval-limit",
        str(config.eval_limit),
        "--num-workers",
        str(config.num_workers),
        "--device",
        config.device,
        "--summ-target",
        str(config.summ_target),
        "--general-target",
        str(config.general_target),
        "--output-dir",
        output_dir.as_posix(),
    ]


def run_vertex_training(
    config: LiteScorerV2VertexConfig,
    *,
    download_prefix: Callable[[str, Path], None] = download_prefix,
    upload_prefix: Callable[[Path, str], None] = upload_prefix,
    train_main: Callable[[list[str]], int] | None = None,
) -> dict[str, object]:
    errors = validate_config(config)
    if errors:
        raise ValueError("; ".join(errors))

    data_dir = config.work_dir / "data"
    teacher_dir = config.work_dir / "teacher"
    student_dir = config.work_dir / "student-base"
    output_dir = config.work_dir / "student-output"
    config.work_dir.mkdir(parents=True, exist_ok=True)

    download_prefix(config.data_uri, data_dir)
    download_prefix(config.teacher_uri, teacher_dir)
    download_prefix(config.student_uri, student_dir)

    os.environ["DIRECTOR_DATA_DIR"] = data_dir.as_posix()
    selected_train_main: Callable[[list[str]], int]
    if train_main is None:
        from training.train_distillation import main as distillation_main

        def run_distillation(argv: list[str]) -> int:
            return distillation_main(argv)

        selected_train_main = run_distillation
    else:
        selected_train_main = train_main

    argv = build_train_argv(
        config,
        data_dir=data_dir,
        teacher_dir=teacher_dir,
        student_dir=student_dir,
        output_dir=output_dir,
    )
    result_code = selected_train_main(argv)
    if result_code != 0:
        raise RuntimeError(f"training failed with exit code {result_code}")
    if not (output_dir / "model.safetensors").exists():
        raise FileNotFoundError(f"missing trained model artifact in {output_dir}")

    result: dict[str, object] = {
        "status": "recorded",
        "public_score_claim": False,
        "official_score_claim": False,
        "benchmark_eligible": False,
        "claim_boundary": (
            "Vertex distillation handoff only; not an official Lite Scorer v2 "
            "benchmark result and not public performance evidence."
        ),
        "output_uri": config.output_uri,
        "local_output_dir": output_dir.as_posix(),
        "config": {
            key: value.as_posix() if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }
    (output_dir / "vertex_run_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    upload_prefix(output_dir, config.output_uri)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-uri", required=True)
    parser.add_argument("--teacher-uri", required=True)
    parser.add_argument("--student-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/lite-scorer-v2"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--eval-limit", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--summ-target", type=int, default=15000)
    parser.add_argument("--general-target", type=int, default=15000)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> LiteScorerV2VertexConfig:
    return LiteScorerV2VertexConfig(
        data_uri=args.data_uri,
        teacher_uri=args.teacher_uri,
        student_uri=args.student_uri,
        output_uri=args.output_uri,
        work_dir=args.work_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        temperature=args.temperature,
        alpha=args.alpha,
        learning_rate=args.lr,
        seed=args.seed,
        eval_limit=args.eval_limit,
        num_workers=args.num_workers,
        device=args.device,
        summ_target=args.summ_target,
        general_target=args.general_target,
    )


def main(argv: list[str] | None = None) -> int:
    config = config_from_args(parse_args(argv))
    try:
        result = run_vertex_training(config)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
