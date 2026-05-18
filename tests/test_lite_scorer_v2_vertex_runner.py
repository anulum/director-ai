# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 Vertex runner tests

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, ROOT.as_posix())

MODULE: Any = importlib.import_module("training.lite_scorer_v2_vertex_runner")

LiteScorerV2VertexConfig = MODULE.LiteScorerV2VertexConfig
build_train_argv = MODULE.build_train_argv
parse_args = MODULE.parse_args
run_vertex_training = MODULE.run_vertex_training
validate_config = MODULE.validate_config


def _config(tmp_path: Path) -> Any:
    return LiteScorerV2VertexConfig(
        data_uri="gs://bucket/lite/data",
        teacher_uri="gs://bucket/lite/teacher",
        student_uri="gs://bucket/lite/student-base",
        output_uri="gs://bucket/lite/output",
        work_dir=tmp_path / "work",
        epochs=5,
        batch_size=32,
        max_length=256,
        temperature=3.0,
        alpha=0.5,
        learning_rate=5e-5,
        seed=20260518,
        eval_limit=5000,
        num_workers=2,
        device="auto",
        summ_target=15000,
        general_target=15000,
    )


def test_vertex_runner_keeps_download_paths_inside_requested_prefix() -> None:
    relative_blob_path = MODULE._relative_blob_path

    assert relative_blob_path("lite/data", "lite/data/train.jsonl") == Path(
        "train.jsonl"
    )
    assert relative_blob_path("lite/data", "lite/data/nested/eval.jsonl") == Path(
        "nested/eval.jsonl"
    )
    assert relative_blob_path("lite/data", "lite/data-old/train.jsonl") is None
    assert relative_blob_path("lite/data", "lite/data") is None


def test_vertex_runner_validates_gcs_inputs(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert validate_config(config) == []

    invalid = LiteScorerV2VertexConfig(
        **{**config.__dict__, "teacher_uri": "/local/teacher", "epochs": 0}
    )
    assert validate_config(invalid) == [
        "teacher_uri must be a gs:// URI",
        "epochs must be positive",
    ]


def test_vertex_runner_rejects_overlapping_output_prefixes(tmp_path: Path) -> None:
    config = _config(tmp_path)

    nested_under_data = LiteScorerV2VertexConfig(
        **{**config.__dict__, "output_uri": "gs://bucket/lite/data/output"}
    )
    assert validate_config(nested_under_data) == [
        "output_uri must not overlap data_uri"
    ]

    parent_of_teacher = LiteScorerV2VertexConfig(
        **{**config.__dict__, "output_uri": "gs://bucket/lite"}
    )
    assert validate_config(parent_of_teacher) == [
        "output_uri must not overlap data_uri",
        "output_uri must not overlap teacher_uri",
        "output_uri must not overlap student_uri",
    ]


def test_vertex_runner_builds_train_argv_from_materialised_paths(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    argv = build_train_argv(
        config,
        data_dir=tmp_path / "work" / "data",
        teacher_dir=tmp_path / "work" / "teacher",
        student_dir=tmp_path / "work" / "student-base",
        output_dir=tmp_path / "work" / "student-output",
    )

    assert argv[:2] == ["--teacher", (tmp_path / "work" / "teacher").as_posix()]
    assert "--student" in argv
    assert (tmp_path / "work" / "student-base").as_posix() in argv
    assert argv[argv.index("--device") + 1] == "auto"
    assert argv[argv.index("--summ-target") + 1] == "15000"
    assert argv[argv.index("--general-target") + 1] == "15000"


def test_vertex_runner_downloads_trains_and_uploads(
    tmp_path: Path, monkeypatch: Any
) -> None:
    config = _config(tmp_path)
    downloads: list[tuple[str, Path]] = []
    uploads: list[tuple[Path, str]] = []
    train_calls: list[list[str]] = []

    def fake_download(uri: str, destination: Path) -> None:
        downloads.append((uri, destination))
        destination.mkdir(parents=True)

    def fake_upload(source: Path, uri: str) -> None:
        uploads.append((source, uri))

    def fake_train(argv: list[str]) -> int:
        train_calls.append(argv)
        output = Path(argv[argv.index("--output-dir") + 1])
        output.mkdir(parents=True)
        (output / "model.safetensors").write_bytes(b"student")
        return 0

    result = run_vertex_training(
        config,
        download_prefix=fake_download,
        upload_prefix=fake_upload,
        train_main=fake_train,
    )

    assert downloads == [
        ("gs://bucket/lite/data", tmp_path / "work" / "data"),
        ("gs://bucket/lite/teacher", tmp_path / "work" / "teacher"),
        ("gs://bucket/lite/student-base", tmp_path / "work" / "student-base"),
    ]
    assert uploads == [
        (tmp_path / "work" / "student-output", "gs://bucket/lite/output")
    ]
    assert train_calls
    assert result["status"] == "recorded"
    assert result["public_score_claim"] is False
    assert result["official_score_claim"] is False
    assert result["benchmark_eligible"] is False
    assert (
        result["claim_boundary"]
        == "Vertex distillation handoff only; not an official Lite Scorer v2 "
        "benchmark result and not public performance evidence."
    )
    assert result["output_uri"] == "gs://bucket/lite/output"

    result_file = tmp_path / "work" / "student-output" / "vertex_run_result.json"
    recorded = MODULE.json.loads(result_file.read_text(encoding="utf-8"))
    assert recorded["official_score_claim"] is False
    assert recorded["benchmark_eligible"] is False


def test_vertex_runner_parse_args_uses_t4_safe_defaults(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--data-uri",
            "gs://bucket/data",
            "--teacher-uri",
            "gs://bucket/teacher",
            "--student-uri",
            "gs://bucket/student",
            "--output-uri",
            "gs://bucket/output",
            "--work-dir",
            str(tmp_path),
        ]
    )

    assert args.batch_size == 32
    assert args.device == "auto"
    assert args.num_workers == 2
