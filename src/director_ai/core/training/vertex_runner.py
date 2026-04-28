# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vertex Fine-tune Runner

"""Container entry point for managed Vertex fine-tuning jobs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .finetune import FinetuneConfig, finetune_nli


def main(argv: list[str] | None = None) -> None:
    """Run a managed fine-tune job inside a Vertex custom job container."""

    args = _parse_args(argv)
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    train_path = _materialise_uri(args.train_uri, work_dir / "train.jsonl")
    eval_path = (
        _materialise_uri(args.eval_uri, work_dir / "eval.jsonl")
        if args.eval_uri
        else None
    )
    model_dir = work_dir / "model"
    result_path = work_dir / "training_result.json"

    result = finetune_nli(
        train_path,
        eval_path=eval_path,
        config=FinetuneConfig(
            base_model=args.base_model,
            output_dir=str(model_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        ),
    )
    result_path.write_text(
        json.dumps(result.__dict__, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _publish_dir(model_dir, args.output_uri)
    _publish_file(result_path, f"{args.output_uri.rstrip('/')}/training_result.json")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-uri", required=True)
    parser.add_argument("--eval-uri")
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=2e-5)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "director-ai-managed-training",
    )
    return parser.parse_args(argv)


def _materialise_uri(uri: str, destination: Path) -> Path:
    """Return a local path for either an existing local URI or a GCS object."""

    if _is_gcs_uri(uri):
        bucket_name, blob_name = _split_gcs_uri(uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _storage_client().bucket(bucket_name).blob(blob_name).download_to_filename(
            str(destination),
        )
        return destination
    return Path(uri)


def _publish_file(source: Path, destination_uri: str) -> None:
    """Publish one file to a local path or a `gs://` object URI."""

    if _is_gcs_uri(destination_uri):
        bucket_name, blob_name = _split_gcs_uri(destination_uri)
        _storage_client().bucket(bucket_name).blob(blob_name).upload_from_filename(
            str(source),
        )
        return

    destination = Path(destination_uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _publish_dir(source_dir: Path, destination_uri: str) -> None:
    """Publish a model directory to a local directory or GCS prefix."""

    if _is_gcs_uri(destination_uri):
        base = destination_uri.rstrip("/")
        for path in source_dir.rglob("*"):
            if path.is_file():
                rel = path.relative_to(source_dir).as_posix()
                _publish_file(path, f"{base}/{rel}")
        return

    destination = Path(destination_uri)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source_dir, destination)


def _is_gcs_uri(uri: str) -> bool:
    return uri.startswith("gs://")


def _split_gcs_uri(uri: str) -> tuple[str, str]:
    """Split `gs://bucket/object` into a bucket name and object path."""

    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"invalid GCS URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _storage_client():
    try:
        from google.cloud.storage import Client
    except ImportError as exc:
        raise ImportError("google-cloud-storage is required for gs:// URIs") from exc

    return Client()


if __name__ == "__main__":
    main()
