# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Sentinel-Judge schema helpers
"""Schema types and JSON loading for Sentinel-Judge analyser inputs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast


class JudgeRecord(TypedDict):
    """Normalised judge result packet."""

    name: str
    model: str
    preds: list[int]
    scores: list[float] | None
    labels: list[int]
    datasets: list[str]


class DatasetMetrics(TypedDict):
    """Per-dataset metric payload."""

    samples: int
    balanced_accuracy: float


class EnsembleMetrics(TypedDict):
    """Common ensemble metric payload."""

    global_balanced_accuracy: float
    per_dataset: dict[str, DatasetMetrics]


class RoutedMetrics(EnsembleMetrics):
    """Routed ensemble metric payload."""

    routing_table: dict[str, str]


class LrFusionMetrics(EnsembleMetrics):
    """Logistic-regression fusion metric payload."""

    method: str


class SentinelReport(TypedDict):
    """Complete Sentinel-Judge report schema."""

    judges: list[str]
    samples: int
    individual: dict[str, EnsembleMetrics]
    voting: EnsembleMetrics
    routed: RoutedMetrics
    lr_fusion: LrFusionMetrics | None
    oracle_upper_bound: EnsembleMetrics


def _read_json_object(path: Path) -> Mapping[str, object]:
    """Read a JSON object from ``path``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return cast(Mapping[str, object], data)


def _list_field(data: Mapping[str, object], field: str, path: Path) -> list[object]:
    """Return a JSON list field or raise a schema error."""
    value = data.get(field, [])
    if not isinstance(value, list):
        raise ValueError(f"{path}: field {field!r} must be a list")
    return value


def _coerce_int_list(values: Sequence[object], field: str, path: Path) -> list[int]:
    """Coerce a JSON sequence to integers."""
    out: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool):
            raise ValueError(f"{path}: field {field!r} item {index} is not an integer")
        if isinstance(value, int):
            out.append(value)
        elif isinstance(value, float) and value.is_integer():
            out.append(int(value))
        elif isinstance(value, str):
            try:
                out.append(int(value))
            except ValueError as exc:
                raise ValueError(
                    f"{path}: field {field!r} item {index} is not an integer",
                ) from exc
        else:
            raise ValueError(f"{path}: field {field!r} item {index} is not an integer")
    return out


def _coerce_str_list(values: Sequence[object], field: str, path: Path) -> list[str]:
    """Coerce a JSON sequence to strings."""
    out: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise ValueError(f"{path}: field {field!r} item {index} is not a string")
        out.append(value)
    return out


def _score_from_legacy_mapping(value: object, path: Path, index: int) -> float:
    """Read one legacy ``scores`` mapping entry."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: legacy score item {index} must be an object")
    return float(value.get("score", -1.0))


def _normalise_scores(
    data: Mapping[str, object],
    *,
    expected_len: int,
    path: Path,
) -> list[float] | None:
    """Return normalised score values or ``None`` when unavailable."""
    raw = data.get("scores")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError(f"{path}: field 'scores' must be a list when provided")
    if not raw:
        return None

    if isinstance(raw[0], Mapping):
        scores = [
            _score_from_legacy_mapping(value, path=path, index=index)
            for index, value in enumerate(raw)
        ]
    else:
        try:
            scores = [float(value) for value in raw]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: scores must be numeric") from exc

    if len(scores) != expected_len:
        raise ValueError(
            f"{path}: inconsistent score length "
            f"scores={len(scores)} labels={expected_len}",
        )
    return scores


def load_judge(path: str) -> JudgeRecord:
    """Load a judge JSON file and normalise its schema."""
    source = Path(path)
    data = _read_json_object(source)
    model = str(data.get("model", path))
    preds = _coerce_int_list(
        _list_field(data, "predictions", source),
        "predictions",
        source,
    )
    labels = _coerce_int_list(_list_field(data, "labels", source), "labels", source)
    datasets = _coerce_str_list(
        _list_field(data, "datasets_per_sample", source),
        "datasets_per_sample",
        source,
    )
    if not (len(preds) == len(labels) == len(datasets)):
        raise ValueError(
            f"{path}: inconsistent list lengths "
            f"preds={len(preds)} labels={len(labels)} datasets={len(datasets)}",
        )
    return {
        "name": source.stem,
        "model": model,
        "preds": preds,
        "scores": _normalise_scores(data, expected_len=len(labels), path=source),
        "labels": labels,
        "datasets": datasets,
    }


def align_judges(
    judges: Sequence[JudgeRecord],
) -> tuple[list[int], list[str], list[list[int]], list[list[float]]]:
    """Return aligned labels, datasets, predictions, and score matrices."""
    if not judges:
        raise ValueError("at least one judge file is required")
    base_labels = judges[0]["labels"]
    base_datasets = judges[0]["datasets"]
    for judge in judges[1:]:
        if judge["labels"] != base_labels:
            raise ValueError(
                f"label mismatch between {judges[0]['name']} and {judge['name']}",
            )
        if judge["datasets"] != base_datasets:
            raise ValueError(
                f"dataset mismatch between {judges[0]['name']} and {judge['name']}",
            )
    return (
        base_labels,
        base_datasets,
        [judge["preds"] for judge in judges],
        [
            judge["scores"]
            if judge["scores"] is not None
            else [-1.0] * len(base_labels)
            for judge in judges
        ],
    )
