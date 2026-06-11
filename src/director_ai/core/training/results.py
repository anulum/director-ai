# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Managed Training Result Harvesting

"""Collect managed training result artifacts from local or GCS prefixes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_RESULT_NAME = "training_result.json"


@dataclass(frozen=True)
class TrainingResultRecord:
    """One collected managed training result."""

    scenario: str
    result_uri: str
    artifact_uri: str
    best_balanced_accuracy: float = 0.0
    final_loss: float = 0.0
    epochs_completed: int = 0
    train_samples: int = 0
    eval_samples: int = 0
    eval_metrics: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "result_uri": self.result_uri,
            "artifact_uri": self.artifact_uri,
            "best_balanced_accuracy": self.best_balanced_accuracy,
            "final_loss": self.final_loss,
            "epochs_completed": self.epochs_completed,
            "train_samples": self.train_samples,
            "eval_samples": self.eval_samples,
            "eval_metrics": dict(self.eval_metrics),
        }


@dataclass(frozen=True)
class TrainingHarvestReport:
    """Summary for a harvested managed training sweep."""

    prefix_uri: str
    result_count: int
    best: TrainingResultRecord | None
    results: list[TrainingResultRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefix_uri": self.prefix_uri,
            "result_count": self.result_count,
            "best": self.best.to_dict() if self.best else None,
            "results": [record.to_dict() for record in self.results],
        }


def harvest_training_results(prefix_uri: str) -> TrainingHarvestReport:
    """Collect `training_result.json` artifacts under *prefix_uri*.

    `prefix_uri` can be a local directory or a `gs://bucket/prefix` URI.
    Results are sorted by balanced accuracy descending, then scenario name.
    """

    records = (
        _harvest_gcs_results(prefix_uri)
        if _is_gcs_uri(prefix_uri)
        else _harvest_local_results(Path(prefix_uri))
    )
    records.sort(key=lambda r: (-r.best_balanced_accuracy, r.scenario))
    return TrainingHarvestReport(
        prefix_uri=prefix_uri,
        result_count=len(records),
        best=records[0] if records else None,
        results=records,
    )


def _harvest_local_results(prefix: Path) -> list[TrainingResultRecord]:
    if not prefix.exists():
        raise FileNotFoundError(f"training result prefix not found: {prefix}")
    if not prefix.is_dir():
        raise ValueError(f"training result prefix must be a directory: {prefix}")
    records = []
    for path in sorted(prefix.rglob(_RESULT_NAME)):
        data = _load_result_json(path.read_text(encoding="utf-8"), str(path))
        scenario = _scenario_from_result_path(path, prefix)
        artifact_uri = str(path.parent)
        records.append(_record_from_data(scenario, str(path), artifact_uri, data))
    return records


def _harvest_gcs_results(prefix_uri: str) -> list[TrainingResultRecord]:
    bucket_name, prefix = _split_gcs_uri(prefix_uri)
    client = _storage_client()
    bucket = client.bucket(bucket_name)
    records = []
    for blob in client.list_blobs(bucket, prefix=prefix.rstrip("/") + "/"):
        blob_name = getattr(blob, "name", "")
        if not blob_name.endswith("/" + _RESULT_NAME):
            continue
        result_uri = f"gs://{bucket_name}/{blob_name}"
        data = _load_result_json(blob.download_as_text(), result_uri)
        artifact_blob = blob_name[: -len("/" + _RESULT_NAME)]
        artifact_uri = f"gs://{bucket_name}/{artifact_blob}"
        scenario = _scenario_from_gcs_blob(blob_name, prefix)
        records.append(_record_from_data(scenario, result_uri, artifact_uri, data))
    return records


def _record_from_data(
    scenario: str,
    result_uri: str,
    artifact_uri: str,
    data: dict[str, Any],
) -> TrainingResultRecord:
    return TrainingResultRecord(
        scenario=scenario,
        result_uri=result_uri,
        artifact_uri=artifact_uri,
        best_balanced_accuracy=float(data.get("best_balanced_accuracy") or 0.0),
        final_loss=float(data.get("final_loss") or 0.0),
        epochs_completed=int(data.get("epochs_completed") or 0),
        train_samples=int(data.get("train_samples") or 0),
        eval_samples=int(data.get("eval_samples") or 0),
        eval_metrics=dict(data.get("eval_metrics") or {}),
        raw=data,
    )


def _load_result_json(raw: str, source: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid training result JSON at {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"training result JSON must be an object: {source}")
    return data


def _scenario_from_result_path(path: Path, prefix: Path) -> str:
    rel_parent = path.parent.relative_to(prefix)
    scenario = rel_parent.as_posix()
    return scenario if scenario != "." else path.parent.name


def _scenario_from_gcs_blob(blob_name: str, prefix: str) -> str:
    clean_prefix = prefix.strip("/")
    result_parent = blob_name[: -len("/" + _RESULT_NAME)].strip("/")
    if clean_prefix and result_parent.startswith(clean_prefix + "/"):
        result_parent = result_parent[len(clean_prefix) + 1 :]
    return result_parent or result_parent.rsplit("/", 1)[-1]


def _is_gcs_uri(uri: str) -> bool:
    return uri.startswith("gs://")


def _split_gcs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"invalid GCS URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _storage_client():
    try:
        from google.cloud.storage import Client
    except ImportError as exc:
        raise ImportError("google-cloud-storage is required for gs:// URIs") from exc
    return Client()
