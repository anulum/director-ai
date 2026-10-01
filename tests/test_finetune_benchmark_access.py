# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Managed benchmark HTTP confinement tests

"""Exercise artifact confinement, operator authentication and local inference."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from director_ai.core.config import DirectorConfig
from director_ai.finetune_jobs import FinetuneJob
from director_ai.server import create_app

_ALIAS = "factcg-deberta-v3-large"
_JOB = "a" * 32
_ROWS = b'{"premise":"Evidence confirms approval.","hypothesis":"Approval exists.","label":1}\n'


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Start the real server with separate ordinary and operator API keys."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("DIRECTOR_FORCE_CPU", "1")
    config = DirectorConfig(
        use_nli=False,
        scorer_backend="lite",
        finetune_models_dir=str(tmp_path / "models"),
        api_keys=["ordinary-test-key", "operator-test-key"],
        finetune_operator_api_keys=["operator-test-key"],
    )
    with TestClient(
        create_app(config), headers={"X-API-Key": "operator-test-key"}
    ) as http:
        yield http


def _job(tmp_path: Path, model: Path, state: str = "completed") -> None:
    """Persist the real job ledger format consumed by the server worker."""
    record = FinetuneJob(
        job_id=_JOB,
        state=state,
        config={"base_model": _ALIAS},
        model_path=str(model),
    )
    with sqlite3.connect(tmp_path / "models" / "finetune_jobs.sqlite3") as db:
        db.execute(
            "INSERT OR REPLACE INTO finetune_jobs VALUES (?, ?, ?)",
            (record.job_id, record.state, json.dumps(asdict(record))),
        )


def _dataset(client: TestClient) -> str:
    """Upload an actual labelled dataset through the public HTTP endpoint."""
    response = client.post(
        "/v1/finetune/managed/datasets", files={"file": ("evaluation.jsonl", _ROWS)}
    )
    assert response.status_code == 200, response.text
    return str(response.json()["dataset_id"])


def _body(dataset_id: str, job_id: str = _JOB) -> dict[str, object]:
    """Build a managed benchmark request using opaque resource identifiers."""
    return {"model_jobs": {_ALIAS: job_id}, "general_dataset_id": dataset_id}


@pytest.mark.parametrize(
    "path",
    ["jobs", "models", "submit", "status", "cancel", "datasets", "benchmark-models"],
)
def test_every_managed_route_requires_operator(client: TestClient, path: str) -> None:
    """Ordinary valid API keys cannot invoke any managed operation."""
    url = f"/v1/finetune/managed/{path}"
    response = client.request(
        "GET" if path in ("jobs", "models") else "POST",
        url,
        headers={"X-API-Key": "ordinary-test-key"},
        json={},
    )
    assert response.status_code == 403
    assert response.json() == {
        "detail": "Managed training requires an operator API key"
    }


@pytest.mark.parametrize(
    "job_id", ["vendor/model", "/outside/model", "~/.cache/model", "b" * 32]
)
def test_foreign_model_ids_are_refused(client: TestClient, job_id: str) -> None:
    """Paths, Hub names and unknown job IDs never reach model inference."""
    response = client.post(
        "/v1/finetune/managed/benchmark-models", json=_body(_dataset(client), job_id)
    )
    assert response.status_code == 422
    assert response.json() == {
        "detail": "Model must identify a completed local training job"
    }
    assert job_id not in response.text


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"not json",
        b"[]\n",
        b"\xff",
        b'{"premise":5,"hypothesis":"claim","label":1}\n',
        b'{"premise":"evidence","hypothesis":"claim","label":true}\n',
        b'{"premise":"evidence","hypothesis":"claim","label":2}\n',
    ],
)
def test_malformed_upload_is_fixed(client: TestClient, content: bytes) -> None:
    """Malformed dataset rows fail before any dataset file is registered."""
    response = client.post(
        "/v1/finetune/managed/datasets", files={"file": ("data.jsonl", content)}
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid benchmark JSONL dataset"}


def test_legacy_paths_are_refused_without_echo(client: TestClient) -> None:
    """Removed free-path fields fail closed instead of silently being ignored."""
    response = client.post(
        "/v1/finetune/managed/benchmark-models",
        json={
            "model_artifacts": {_ALIAS: "/ordinary/outside/model"},
            "general_path": "/ordinary/outside/data",
        },
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid managed training request"}
    assert "/ordinary" not in response.text


def test_benchmark_model_count_is_bounded(client: TestClient) -> None:
    """Reject oversized candidate batches before resolving any artifact."""
    response = client.post(
        "/v1/finetune/managed/benchmark-models",
        json={
            "model_jobs": {f"candidate-{i}": _JOB for i in range(9)},
            "general_dataset_id": "b" * 32,
        },
    )
    assert response.status_code == 422
    assert response.json() == {
        "detail": "Benchmark requires between one and eight local jobs"
    }


def test_completed_artifact_must_stay_under_root(
    client: TestClient, tmp_path: Path
) -> None:
    """A foreign directory in a stored job cannot be loaded as a model."""
    outside = tmp_path / "outside"
    outside.mkdir()
    _job(tmp_path, outside)
    response = client.post(
        "/v1/finetune/managed/benchmark-models", json=_body(_dataset(client))
    )
    assert response.status_code == 422
    assert response.json() == {
        "detail": "Model artifact is outside the training model root"
    }
    assert str(tmp_path) not in response.text


def test_dataset_symlink_is_confined(client: TestClient, tmp_path: Path) -> None:
    """An uploaded dataset ID cannot redirect reads outside the model root."""
    model = tmp_path / "models" / "candidate"
    model.mkdir()
    _job(tmp_path, model)
    dataset_id = _dataset(client)
    dataset = tmp_path / "models" / "_benchmark_datasets" / f"{dataset_id}.jsonl"
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(_ROWS)
    dataset.unlink()
    dataset.symlink_to(outside)
    response = client.post(
        "/v1/finetune/managed/benchmark-models", json=_body(dataset_id)
    )
    assert response.status_code == 422
    assert response.json() == {
        "detail": "Benchmark dataset is outside the training model root"
    }
    assert str(tmp_path) not in response.text


@pytest.mark.parametrize("condition", ["failed", "wrong-alias", "missing", "file"])
def test_unusable_job_artifacts_fail_before_inference(
    client: TestClient, tmp_path: Path, condition: str
) -> None:
    """Incomplete, mismatched and missing artifacts cannot become benchmarks."""
    model = tmp_path / "models" / "candidate"
    if condition == "file":
        model.write_text("ordinary file", encoding="utf-8")
    elif condition != "missing":
        model.mkdir()
    _job(tmp_path, model, state="failed" if condition == "failed" else "completed")
    body = _body(_dataset(client))
    if condition == "wrong-alias":
        body["model_jobs"] = {"modernbert-large": _JOB}
    response = client.post("/v1/finetune/managed/benchmark-models", json=body)
    assert response.status_code == 422
    assert str(model) not in response.text


@pytest.mark.parametrize("dataset_id", ["b" * 32, "/outside/data", "~/data"])
def test_unknown_or_path_dataset_ids_are_refused(
    client: TestClient, tmp_path: Path, dataset_id: str
) -> None:
    """Dataset resolution never accepts paths or silently selects another file."""
    model = tmp_path / "models" / "candidate"
    model.mkdir()
    _job(tmp_path, model)
    response = client.post(
        "/v1/finetune/managed/benchmark-models", json=_body(dataset_id)
    )
    assert response.status_code == (404 if dataset_id == "b" * 32 else 422)
    assert dataset_id not in response.text


def test_dataset_directory_refuses_upload_outside_root(
    client: TestClient, tmp_path: Path
) -> None:
    """Dataset registration also confines writes when the data root is linked."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "models" / "_benchmark_datasets").symlink_to(
        outside, target_is_directory=True
    )
    response = client.post(
        "/v1/finetune/managed/datasets", files={"file": ("data.jsonl", _ROWS)}
    )
    assert response.status_code == 422
    assert list(outside.iterdir()) == []


def test_upload_size_is_bounded(client: TestClient) -> None:
    """Oversized uploads stop at the fixed dataset limit before registration."""
    response = client.post(
        "/v1/finetune/managed/datasets",
        files={"file": ("large.jsonl", b" " * (10 * 1024 * 1024 + 1))},
    )
    assert response.status_code == 413
    assert response.json() == {"detail": "Benchmark dataset exceeds 10 MiB"}


def test_upload_storage_fault_is_fixed(client: TestClient, tmp_path: Path) -> None:
    """An actual invalid storage layout yields a fixed failure response."""
    (tmp_path / "models" / "_benchmark_datasets").write_text(
        "occupied", encoding="utf-8"
    )
    response = client.post(
        "/v1/finetune/managed/datasets", files={"file": ("data.jsonl", _ROWS)}
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "Managed training request failed"}
    assert str(tmp_path) not in response.text


def test_operator_env_keys_are_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    """JSON environment keys parse correctly without appearing in config output."""
    monkeypatch.setenv("DIRECTOR_FINETUNE_OPERATOR_API_KEYS", '["operator-test-key"]')
    config = DirectorConfig.from_env()
    assert config.finetune_operator_api_keys == ["operator-test-key"]
    assert config.to_dict()["finetune_operator_api_keys"] == "***"


def test_unconfigured_operator_access_is_disabled(tmp_path: Path) -> None:
    """A standalone development router does not grant anonymous managed access."""
    from fastapi import FastAPI

    from director_ai.finetune_api import create_finetune_router

    app = FastAPI()
    app.include_router(
        create_finetune_router(tmp_path / "models"), prefix="/v1/finetune"
    )
    with TestClient(app) as http:
        response = http.get("/v1/finetune/managed/jobs")
    assert response.status_code == 403


@pytest.fixture
def local_inference_state() -> Iterator[None]:
    """Keep native checkpoint generation isolated from other tests' Torch state."""
    import torch

    previous_threads = torch.get_num_threads()
    previous_rng = torch.get_rng_state()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    try:
        yield
    finally:
        torch.set_rng_state(previous_rng)
        torch.set_num_threads(previous_threads)


def test_completed_local_checkpoint_runs_real_inference(
    client: TestClient, tmp_path: Path, local_inference_state: None
) -> None:
    """Actual Transformers inference uses the stored local checkpoint offline."""
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

    model = tmp_path / "models" / "candidate"
    model.mkdir()
    vocabulary = {
        word: index
        for index, word in enumerate(
            [
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "evidence",
                "approval",
                "exists",
                ".",
            ]
        )
    }
    tokenizer = BertTokenizer(vocab=vocabulary)
    tokenizer.save_pretrained(model)
    configuration = BertConfig(
        vocab_size=len(vocabulary),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
    )
    configuration.num_labels = 2
    checkpoint = BertForSequenceClassification.from_pretrained(
        None,
        config=configuration,
        state_dict={},
    )
    checkpoint.save_pretrained(model)
    _job(tmp_path, model)
    dataset_id = _dataset(client)
    response = client.post(
        "/v1/finetune/managed/benchmark-models",
        json={**_body(dataset_id), "eval_dataset_id": dataset_id, "batch_size": 1},
    )
    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert result["error"] == ""
    assert result["details"]["general_samples"] == 1
    assert result["details"]["domain_samples"] == 1
    assert result["model_path"] == _JOB
    assert response.json()["general_path"] == dataset_id
    assert str(tmp_path) not in response.text


def test_non_ascii_invalid_key_returns_auth_refusal(client: TestClient) -> None:
    """Malformed API keys yield a fixed auth refusal rather than a native fault."""
    response = client.get(
        "/v1/finetune/managed/jobs",
        headers=[(b"X-API-Key", b"invalid-\xff")],
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}
