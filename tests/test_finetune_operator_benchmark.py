# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — operator-held general benchmark for API fine-tuning jobs
"""The operator's general benchmark, from configuration to the job's report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import director_ai._finetune_worker as worker_module
import director_ai.core.training.finetune as finetune_module
import director_ai.finetune_api as finetune_api_module
from director_ai._finetune_operator_benchmark import (
    operator_benchmark,
    snapshot_operator_benchmark,
)
from director_ai.core.config import DirectorConfig
from director_ai.core.training.finetune import FinetuneConfig
from director_ai.finetune_jobs import FinetuneJob, _JobStore

ROWS = [
    {"premise": f"General passage {i}.", "hypothesis": f"Claim {i}.", "label": i % 2}
    for i in range(6)
]


def benchmark_file(tmp_path: Path) -> Path:
    """Write a small labelled general benchmark.

    Parameters
    ----------
    tmp_path : Path
        Directory for the file.

    Returns
    -------
    Path
        The benchmark.
    """
    path = tmp_path / "operator-general.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in ROWS), encoding="utf-8")
    return path


def test_an_unset_benchmark_is_none_and_a_set_one_is_checked(tmp_path: Path) -> None:
    """Return None when unset; refuse a missing, non-file or unlabelled path."""
    assert operator_benchmark(None) is None
    assert operator_benchmark("") is None
    with pytest.raises(FileNotFoundError):
        operator_benchmark(tmp_path / "absent.jsonl")
    with pytest.raises(FileNotFoundError, match="not a file"):
        operator_benchmark(tmp_path)
    unlabelled = tmp_path / "unlabelled.jsonl"
    unlabelled.write_text('{"premise": "p"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no labelled rows"):
        operator_benchmark(unlabelled)
    path = benchmark_file(tmp_path)
    assert operator_benchmark(str(path)) == path.resolve()


def test_a_snapshot_copies_the_bytes_and_names_their_digest(tmp_path: Path) -> None:
    """The job's copy has the operator file's bytes and their SHA-256."""
    source = benchmark_file(tmp_path)
    copy = tmp_path / "copy.jsonl"
    digest = snapshot_operator_benchmark(source, copy)
    assert copy.read_bytes() == source.read_bytes()
    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()


def run_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    general: Path | None,
    *,
    auto_benchmark: bool = True,
) -> tuple[FinetuneJob, dict[str, object]]:
    """Run the real worker with the training call replaced by a recorder.

    Parameters
    ----------
    tmp_path : Path
        Directory for the upload and the models.
    monkeypatch : pytest.MonkeyPatch
        Replaces the training call.
    general : Path | None
        The operator's benchmark.
    auto_benchmark : bool
        The job's benchmark request.

    Returns
    -------
    tuple[FinetuneJob, dict[str, object]]
        The finished job and what the training call saw.
    """
    data_path = tmp_path / "upload.jsonl"
    data_path.write_text(
        "".join(
            json.dumps(
                {"premise": f"Domain {i}.", "hypothesis": f"Claim {i}.", "label": i % 2}
            )
            + "\n"
            for i in range(20)
        ),
        encoding="utf-8",
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    seen: dict[str, object] = {}

    def record(
        train_path: str, *, eval_path: str, config: FinetuneConfig
    ) -> SimpleNamespace:
        """Record the benchmark path the job trains against, and its bytes."""
        seen["path"] = config.general_benchmark_path
        if config.general_benchmark_path is not None:
            seen["bytes"] = Path(config.general_benchmark_path).read_bytes()
        return SimpleNamespace(
            output_dir=config.output_dir,
            eval_metrics={},
            regression_report={"recommendation": "deploy"},
            best_balanced_accuracy=0.0,
        )

    monkeypatch.setattr(finetune_module, "finetune_nli", record)
    job = FinetuneJob(
        job_id="operator-benchmark",
        config={"epochs": 1, "batch_size": 4, "auto_benchmark": auto_benchmark},
    )
    worker_module._run_training_worker(job, data_path, models_dir, _JobStore(), general)
    return job, seen


def test_a_job_benchmarks_against_its_own_copy_and_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Train against a private copy; report its digest; remove it afterwards."""
    general = benchmark_file(tmp_path)
    job, seen = run_worker(tmp_path, monkeypatch, general)
    assert job.state == "completed", job.error
    copy = Path(str(seen["path"]))
    assert copy != general and copy.name == "operator-benchmark_general.jsonl"
    assert seen["bytes"] == general.read_bytes()
    assert not copy.exists()
    assert job.regression_report == {
        "recommendation": "deploy",
        "general_benchmark_sha256": hashlib.sha256(general.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize("auto_benchmark", [True, False])
def test_no_copy_without_a_benchmark_or_its_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, auto_benchmark: bool
) -> None:
    """Without an operator benchmark, or when not requested, nothing is copied."""
    general = None if auto_benchmark else benchmark_file(tmp_path)
    job, seen = run_worker(
        tmp_path, monkeypatch, general, auto_benchmark=auto_benchmark
    )
    assert job.state == "completed", job.error
    assert seen["path"] is None
    assert job.regression_report == {"recommendation": "deploy"}
    assert not (tmp_path / "operator-benchmark_general.jsonl").exists()


def test_the_server_passes_the_configured_benchmark_to_each_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The configuration reaches the job's thread through the real start route."""
    from fastapi.testclient import TestClient

    from director_ai.server import create_app

    general = benchmark_file(tmp_path)
    app = create_app(
        DirectorConfig(
            use_nli=False,
            finetune_models_dir=str(tmp_path / "models"),
            finetune_general_benchmark_path=str(general),
            api_keys=["operator-test-key"],
        )
    )
    upload = "".join(
        json.dumps(
            {"premise": f"Fact {i}.", "hypothesis": f"Claim {i}.", "label": i % 2}
        )
        + "\n"
        for i in range(600)
    ).encode()
    started: list[tuple[object, ...]] = []

    def record_start(*args: object) -> None:
        """Record the arguments the start route gives the job's thread."""
        started.append(args)

    monkeypatch.setattr(finetune_api_module, "_run_training_worker", record_start)
    with TestClient(app) as client:
        response = client.post(
            "/v1/finetune/start",
            headers={"X-API-Key": "operator-test-key"},
            files={"file": ("train.jsonl", upload, "application/jsonl")},
        )
    assert response.status_code == 200, response.text
    assert len(started) == 1
    assert started[0][-1] == general.resolve()


def test_a_misconfigured_benchmark_stops_the_server(tmp_path: Path) -> None:
    """Refuse to start with a benchmark that does not exist."""
    from director_ai.server import create_app

    with pytest.raises(FileNotFoundError):
        create_app(
            DirectorConfig(
                use_nli=False,
                finetune_models_dir=str(tmp_path / "models"),
                finetune_general_benchmark_path=str(tmp_path / "absent.jsonl"),
            )
        )


def test_the_benchmark_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configure the benchmark with its environment variable."""
    monkeypatch.setenv("DIRECTOR_FINETUNE_GENERAL_BENCHMARK_PATH", "/srv/general.jsonl")
    assert (
        DirectorConfig.from_env().finetune_general_benchmark_path
        == "/srv/general.jsonl"
    )
