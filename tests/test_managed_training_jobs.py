# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Managed Training Job Tests

"""Multi-angle tests for managed training job specifications and callers."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from director_ai.cli import main
from director_ai.core.training.jobs import (
    LocalTrainingBackend,
    TrainingHardware,
    TrainingJobSpec,
    VertexTrainingBackend,
    build_internal_suite_spec,
    build_vertex_custom_job_request,
    get_training_backend,
    submit_training_job,
)


def _vertex_spec(**overrides) -> TrainingJobSpec:
    values = {
        "display_name": "tenant-training",
        "caller": "product",
        "dataset_uri": "gs://director-data/train.jsonl",
        "output_uri": "gs://director-artifacts/jobs/job-1",
        "eval_uri": "gs://director-data/eval.jsonl",
        "project": "director-project",
        "container_image_uri": "us-docker.pkg.dev/director/train:latest",
        "hardware": TrainingHardware(),
    }
    values.update(overrides)
    return TrainingJobSpec(**values)


class TestTrainingJobSpec:
    def test_vertex_spec_requires_cloud_uris(self):
        spec = _vertex_spec(dataset_uri="/tmp/train.jsonl")
        with pytest.raises(ValueError, match="dataset_uri must be a gs:// URI"):
            spec.validate("vertex")

    def test_vertex_spec_requires_project(self):
        spec = _vertex_spec(project=None)
        with pytest.raises(ValueError, match="project is required"):
            spec.validate("vertex")

    def test_vertex_spec_rejects_placeholder_image(self):
        spec = _vertex_spec(container_image_uri="python:3.12-slim")
        with pytest.raises(ValueError, match="training image"):
            spec.validate("vertex")

    def test_local_spec_accepts_paths(self, tmp_path):
        spec = TrainingJobSpec(
            display_name="local",
            dataset_uri=str(tmp_path / "train.jsonl"),
            output_uri=str(tmp_path / "out"),
        )
        spec.validate("local")

    def test_env_redaction(self):
        spec = _vertex_spec(env={"API_TOKEN": "secret", "MODE": "test"})
        redacted = spec.to_redacted_dict()
        assert redacted["env"]["API_TOKEN"] == "<redacted>"
        assert redacted["env"]["MODE"] == "test"

    def test_hashes_are_stable(self):
        spec = _vertex_spec()
        assert spec.dataset_hash == _vertex_spec().dataset_hash
        assert spec.config_hash == _vertex_spec().config_hash

    def test_default_model_resolves_to_registry_profile(self):
        spec = _vertex_spec()
        profile = spec.resolved_model_profile()
        assert profile.alias == "factcg-deberta-v3-large"
        assert profile.model_id == "yaxili96/FactCG-DeBERTa-v3-Large"

    def test_experimental_model_requires_explicit_flag(self):
        spec = _vertex_spec(base_model="roberta-large-mnli")
        with pytest.raises(ValueError, match="experimental"):
            spec.validate("vertex")

    def test_custom_model_requires_explicit_flag(self):
        spec = _vertex_spec(base_model="org/custom-model")
        with pytest.raises(ValueError, match="stable fine-tune registry"):
            spec.validate("vertex")


class TestVertexRequest:
    def test_builds_worker_pool_spec(self):
        request = build_vertex_custom_job_request(_vertex_spec())
        pool = request["job_spec"]["worker_pool_specs"][0]
        assert request["display_name"] == "tenant-training"
        assert pool["machine_spec"]["machine_type"] == "g2-standard-8"
        assert pool["machine_spec"]["accelerator_type"] == "NVIDIA_L4"
        assert pool["container_spec"]["image_uri"].endswith("train:latest")
        assert "--epochs" in pool["container_spec"]["args"]
        assert "yaxili96/FactCG-DeBERTa-v3-Large" in pool["container_spec"]["args"]

    def test_timeout_converted_to_seconds(self):
        request = build_vertex_custom_job_request(_vertex_spec(timeout_minutes=7))
        assert request["job_spec"]["scheduling"]["timeout"] == "420s"

    def test_internal_suite_uses_same_vertex_request_shape(self):
        spec = build_internal_suite_spec(
            suite="test_finetune_gpu",
            dataset_uri="gs://director-data/internal.jsonl",
            output_uri="gs://director-artifacts/internal",
            project="director-project",
            container_image_uri="us-docker.pkg.dev/director/train:latest",
        )
        request = build_vertex_custom_job_request(spec)
        args = request["job_spec"]["worker_pool_specs"][0]["container_spec"]["args"]
        assert spec.caller == "internal"
        assert spec.task_type == "suite"
        assert "pytest" in args
        assert request["labels"]["suite"] == "test-finetune-gpu"

    def test_experimental_model_request_when_allowed(self):
        spec = _vertex_spec(
            base_model="roberta-large-mnli",
            allow_experimental_model=True,
        )
        request = build_vertex_custom_job_request(spec)
        args = request["job_spec"]["worker_pool_specs"][0]["container_spec"]["args"]
        assert "roberta-large-mnli" in args
        assert request["labels"]["director-ai-model"] == "roberta-large-mnli"


class TestBackends:
    def test_get_backend(self):
        assert isinstance(get_training_backend("local"), LocalTrainingBackend)
        assert isinstance(get_training_backend("vertex"), VertexTrainingBackend)
        with pytest.raises(ValueError):
            get_training_backend("missing")

    def test_vertex_dry_run_does_not_import_cloud_sdk(self):
        with patch("importlib.import_module") as mock_import:
            result = submit_training_job(_vertex_spec(), backend="vertex", dry_run=True)
        mock_import.assert_not_called()
        assert result.state == "dry_run"
        assert result.job_id.startswith("projects/director-project/")

    def test_local_dry_run_returns_command(self, tmp_path):
        spec = TrainingJobSpec(
            display_name="local",
            dataset_uri=str(tmp_path / "train.jsonl"),
            output_uri=str(tmp_path / "model"),
            epochs=2,
        )
        result = submit_training_job(spec, backend="local", dry_run=True)
        assert result.request["command"][:2] == ["director-ai", "finetune"]
        assert "--epochs" in result.request["command"]

    def test_local_execute_runs_finetune_api(self, tmp_path):
        spec = TrainingJobSpec(
            display_name="local",
            dataset_uri=str(tmp_path / "train.jsonl"),
            output_uri=str(tmp_path / "model"),
        )
        with patch("director_ai.core.training.finetune.finetune_nli") as mock_finetune:
            result = submit_training_job(spec, backend="local", dry_run=False)
        mock_finetune.assert_called_once()
        assert result.state == "completed"


class TestManagedTrainingCLI:
    def test_train_help_mentions_submit(self, capsys):
        main(["train"])
        captured = capsys.readouterr()
        assert "submit" in captured.out
        assert "models" in captured.out

    def test_cli_models_lists_stable_registry(self, capsys):
        main(["train", "models"])
        captured = capsys.readouterr()
        body = json.loads(captured.out)
        assert body["models"][0]["alias"] == "factcg-deberta-v3-large"
        assert all(model["status"] == "stable" for model in body["models"])

    def test_cli_models_can_include_experimental(self, capsys):
        main(["train", "models", "--include-experimental"])
        captured = capsys.readouterr()
        body = json.loads(captured.out)
        assert any(model["status"] == "experimental" for model in body["models"])

    def test_cli_vertex_dry_run_outputs_json(self, capsys):
        main(
            [
                "train",
                "submit",
                "--backend",
                "vertex",
                "--dataset-uri",
                "gs://director-data/train.jsonl",
                "--output-uri",
                "gs://director-artifacts/jobs/job-1",
                "--project",
                "director-project",
                "--image",
                "us-docker.pkg.dev/director/train:latest",
            ]
        )
        captured = capsys.readouterr()
        body = json.loads(captured.out)
        assert body["backend"] == "vertex"
        assert body["dry_run"] is True
        assert body["request"]["job_spec"]["worker_pool_specs"]

    def test_cli_submit_rejects_experimental_without_flag(self, tmp_path):
        with pytest.raises(ValueError, match="experimental"):
            main(
                [
                    "train",
                    "submit",
                    "--backend",
                    "local",
                    "--dataset-uri",
                    str(tmp_path / "train.jsonl"),
                    "--output-uri",
                    str(tmp_path / "out"),
                    "--model",
                    "roberta-large-mnli",
                ]
            )

    @patch("director_ai.core.training.finetune_benchmark._evaluate_model")
    def test_cli_benchmark_models_outputs_report(self, mock_eval, capsys, tmp_path):
        general = tmp_path / "general.jsonl"
        general.write_text(
            json.dumps({"premise": "a", "hypothesis": "b", "label": 1}) + "\n",
            encoding="utf-8",
        )
        mock_eval.return_value = {"balanced_accuracy": 0.80, "f1": 0.78}
        main(
            [
                "train",
                "benchmark-models",
                "--model",
                f"factcg-deberta-v3-large={tmp_path / 'model'}",
                "--general-uri",
                str(general),
            ]
        )
        captured = capsys.readouterr()
        body = json.loads(captured.out)
        assert body["best_model_alias"] == "factcg-deberta-v3-large"

    def test_cli_local_suite_dry_run_prints_command(self, capsys, tmp_path):
        main(
            [
                "train",
                "submit",
                "--backend",
                "local",
                "--suite",
                "test_finetune_gpu",
                "--dataset-uri",
                str(tmp_path / "train.jsonl"),
                "--output-uri",
                str(tmp_path / "out"),
            ]
        )
        captured = capsys.readouterr()
        assert "pytest" in captured.out
        assert "Command:" in captured.out


class TestManagedTrainingAPI:
    @pytest.fixture
    def client(self, tmp_path):
        pytest.importorskip("fastapi")
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from director_ai.finetune_api import create_finetune_router

        app = FastAPI()
        app.include_router(
            create_finetune_router(models_dir=tmp_path / "models"),
            prefix="/v1/finetune",
        )
        return TestClient(app)

    def test_managed_submit_vertex_dry_run(self, client):
        response = client.post(
            "/v1/finetune/managed/submit",
            json={
                "backend": "vertex",
                "dry_run": True,
                "dataset_uri": "gs://director-data/train.jsonl",
                "output_uri": "gs://director-artifacts/jobs/job-1",
                "project": "director-project",
                "container_image_uri": "us-docker.pkg.dev/director/train:latest",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["backend"] == "vertex"
        assert body["state"] == "dry_run"
        args = body["request"]["job_spec"]["worker_pool_specs"][0]["container_spec"][
            "args"
        ]
        assert "yaxili96/FactCG-DeBERTa-v3-Large" in args

    def test_managed_submit_rejects_invalid_vertex_spec(self, client):
        response = client.post(
            "/v1/finetune/managed/submit",
            json={
                "backend": "vertex",
                "dataset_uri": "/tmp/train.jsonl",
                "output_uri": "gs://director-artifacts/jobs/job-1",
                "project": "director-project",
                "container_image_uri": "us-docker.pkg.dev/director/train:latest",
            },
        )
        assert response.status_code == 422

    def test_managed_models_endpoint(self, client):
        response = client.get("/v1/finetune/managed/models")
        assert response.status_code == 200
        body = response.json()
        assert body["models"][0]["alias"] == "factcg-deberta-v3-large"

    @patch("director_ai.core.training.finetune_benchmark._evaluate_model")
    def test_managed_benchmark_models_endpoint(self, mock_eval, client, tmp_path):
        general = tmp_path / "general.jsonl"
        general.write_text(
            json.dumps({"premise": "a", "hypothesis": "b", "label": 1}) + "\n",
            encoding="utf-8",
        )
        mock_eval.return_value = {"balanced_accuracy": 0.80, "f1": 0.78}
        response = client.post(
            "/v1/finetune/managed/benchmark-models",
            json={
                "model_artifacts": {
                    "factcg-deberta-v3-large": str(tmp_path / "model"),
                },
                "general_path": str(general),
            },
        )
        assert response.status_code == 200
        assert response.json()["best_model_alias"] == "factcg-deberta-v3-large"
