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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from director_ai.cli import main
from director_ai.core.training.jobs import (
    LocalTrainingBackend,
    TrainingHardware,
    TrainingJobSpec,
    TrainingJobStatus,
    TrainingJobSubmission,
    VertexTrainingBackend,
    build_internal_suite_spec,
    build_vertex_custom_job_request,
    get_training_backend,
    submit_training_job,
)
from director_ai.core.training.results import harvest_training_results
from director_ai.core.training.sweeps import (
    TrainingDatasetSplit,
    build_training_sweep_plan,
)
from director_ai.core.training.vertex_runner import _split_gcs_uri
from director_ai.core.training.vertex_runner import main as vertex_runner_main


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
        assert pool["container_spec"]["command"] == ["python"]
        assert pool["container_spec"]["args"][:3] == [
            "-m",
            "director_ai.core.training.vertex_runner",
            "--train-uri",
        ]
        assert "--epochs" in pool["container_spec"]["args"]
        assert "yaxili96/FactCG-DeBERTa-v3-Large" in pool["container_spec"]["args"]

    def test_timeout_converted_to_seconds(self):
        request = build_vertex_custom_job_request(_vertex_spec(timeout_minutes=7))
        assert request["job_spec"]["scheduling"]["timeout"] == "420s"

    def test_gpu_quota_alias_is_normalised_for_vertex_sdk(self):
        spec = _vertex_spec(
            hardware=TrainingHardware(
                machine_type="n1-standard-8",
                accelerator_type="NVIDIA_T4",
                accelerator_count=1,
            ),
        )
        request = build_vertex_custom_job_request(spec)
        pool = request["job_spec"]["worker_pool_specs"][0]
        assert pool["machine_spec"]["accelerator_type"] == "NVIDIA_TESLA_T4"

    def test_internal_suite_uses_same_vertex_request_shape(self):
        spec = build_internal_suite_spec(
            suite="test_finetune_gpu",
            dataset_uri="gs://director-data/internal.jsonl",
            output_uri="gs://director-artifacts/internal",
            project="director-project",
            container_image_uri="us-docker.pkg.dev/director/train:latest",
        )
        request = build_vertex_custom_job_request(spec)
        container = request["job_spec"]["worker_pool_specs"][0]["container_spec"]
        assert spec.caller == "internal"
        assert spec.task_type == "suite"
        assert container["command"] == ["python"]
        assert container["args"][:2] == ["-m", "pytest"]
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

    def test_vertex_execute_uses_current_sdk_submit_signature(self):
        class FakeJob:
            resource_name = "projects/director-project/locations/us/customJobs/123"
            gca_resource = "console-resource"

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                submitted_jobs.append(self)

            def submit(self, **kwargs):
                self.submit_kwargs = kwargs

        submitted_jobs = []
        fake_module = SimpleNamespace(
            init=lambda **kwargs: None,
            CustomJob=FakeJob,
        )

        with patch("importlib.import_module", return_value=fake_module):
            result = submit_training_job(
                _vertex_spec(timeout_minutes=7),
                backend="vertex",
                dry_run=False,
            )

        assert result.state == "submitted"
        assert result.job_id.endswith("/123")
        assert result.console_uri.endswith("/training/123?project=director-project")
        assert submitted_jobs[0].submit_kwargs["timeout"] == 420
        assert "sync" not in submitted_jobs[0].submit_kwargs

    def test_cli_submit_reports_submission_failure(self, capsys):
        with (
            patch(
                "director_ai.core.training.jobs.submit_training_job",
                side_effect=RuntimeError("quota unavailable"),
            ),
            pytest.raises(SystemExit) as excinfo,
        ):
            main(
                [
                    "train",
                    "submit",
                    "--dataset-uri",
                    "gs://director-data/train.jsonl",
                    "--output-uri",
                    "gs://director-artifacts/out",
                    "--project",
                    "director-project",
                    "--image",
                    "us-docker.pkg.dev/director/train:latest",
                    "--execute",
                ]
            )

        assert excinfo.value.code == 1
        assert "training job submission failed" in capsys.readouterr().out

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


class TestVertexRunner:
    def test_rejects_malformed_gcs_uri(self):
        with pytest.raises(ValueError, match="invalid GCS URI"):
            _split_gcs_uri("gs://bucket")

    def test_local_smoke_runner_publishes_result(self, tmp_path):
        train = tmp_path / "train.jsonl"
        train.write_text(
            '{"premise":"p","hypothesis":"h","label":1}\n',
            encoding="utf-8",
        )
        output = tmp_path / "published"

        def fake_finetune(train_path, *, eval_path, config):
            assert train_path == train
            assert eval_path is None
            model_dir = tmp_path / "work" / "model"
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            assert config.output_dir == str(model_dir)
            return SimpleNamespace(
                output_dir=str(model_dir), best_balanced_accuracy=1.0
            )

        with patch(
            "director_ai.core.training.vertex_runner.finetune_nli",
            side_effect=fake_finetune,
        ):
            vertex_runner_main(
                [
                    "--train-uri",
                    str(train),
                    "--output-uri",
                    str(output),
                    "--base-model",
                    "model",
                    "--work-dir",
                    str(tmp_path / "work"),
                ],
            )

        assert (output / "config.json").exists()
        assert (output / "training_result.json").exists()


class TestManagedTrainingSweeps:
    def test_builds_cross_product_with_output_paths(self):
        plan = build_training_sweep_plan(
            sweep_id="sweep-1",
            datasets=[
                TrainingDatasetSplit(
                    name="smoke-500",
                    train_uri="gs://director-data/smoke-train.jsonl",
                    eval_uri="gs://director-data/smoke-eval.jsonl",
                )
            ],
            base_models=["factcg-deberta-v3-large", "roberta-large-mnli"],
            epochs=[1, 3],
            batch_sizes=[1],
            learning_rate=1e-5,
            output_prefix="gs://director-artifacts/sweeps/sweep-1",
            allow_experimental_model=True,
        )

        assert len(plan.scenarios) == 4
        assert (
            plan.scenarios[0].scenario_id == "smoke-500-factcg-deberta-v3-large-e1-b1"
        )
        assert plan.scenarios[0].output_uri.endswith(
            "/smoke-500-factcg-deberta-v3-large-e1-b1"
        )
        assert plan.scenarios[-1].labels["model"] == "roberta-large-mnli"

    def test_sweep_plan_converts_to_vertex_specs(self):
        plan = build_training_sweep_plan(
            sweep_id="sweep-1",
            datasets=[
                TrainingDatasetSplit(
                    name="smoke",
                    train_uri="gs://director-data/train.jsonl",
                    eval_uri="gs://director-data/eval.jsonl",
                )
            ],
            base_models=["factcg-deberta-v3-large"],
            epochs=[1],
            batch_sizes=[1],
            output_prefix="gs://director-artifacts/sweeps/sweep-1",
        )
        specs = plan.to_specs(
            project="director-project",
            region="europe-west4",
            container_image_uri="us-docker.pkg.dev/director/train:latest",
            hardware=TrainingHardware(
                machine_type="n1-standard-8",
                accelerator_type="NVIDIA_T4",
            ),
            timeout_minutes=45,
        )

        assert len(specs) == 1
        assert specs[0].display_name.startswith("director-ai-managed-sweep-")
        assert specs[0].labels["sweep"] == "sweep-1"
        assert specs[0].eval_uri == "gs://director-data/eval.jsonl"


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

    def test_cli_submit_rejects_experimental_without_flag(self, capsys, tmp_path):
        with pytest.raises(SystemExit) as excinfo:
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
        assert excinfo.value.code == 1
        assert "experimental" in capsys.readouterr().out

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

    def test_cli_sweep_dry_run_outputs_all_scenarios(self, capsys):
        main(
            [
                "train",
                "sweep",
                "--sweep-id",
                "smoke-sweep",
                "--project",
                "director-project",
                "--image",
                "us-docker.pkg.dev/director/train:latest",
                "--output-prefix",
                "gs://director-artifacts/sweeps/smoke-sweep",
                "--train-set",
                "smoke=gs://director-data/train.jsonl",
                "--eval-set",
                "smoke=gs://director-data/eval.jsonl",
                "--model",
                "factcg-deberta-v3-large",
                "--epochs",
                "1",
                "--epochs",
                "3",
                "--batch-size",
                "1",
            ]
        )
        body = json.loads(capsys.readouterr().out)
        assert body["dry_run"] is True
        assert body["plan"]["scenario_count"] == 2
        assert len(body["submissions"]) == 2
        args = body["submissions"][0]["request"]["job_spec"]["worker_pool_specs"][0][
            "container_spec"
        ]["args"]
        assert "director_ai.core.training.vertex_runner" in args

    def test_cli_sweep_limit_blocks_accidental_large_batch(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(
                [
                    "train",
                    "sweep",
                    "--sweep-id",
                    "too-large",
                    "--project",
                    "director-project",
                    "--image",
                    "us-docker.pkg.dev/director/train:latest",
                    "--output-prefix",
                    "gs://director-artifacts/sweeps/too-large",
                    "--train-set",
                    "smoke=gs://director-data/train.jsonl",
                    "--model",
                    "factcg-deberta-v3-large",
                    "--epochs",
                    "1",
                    "--epochs",
                    "3",
                    "--limit",
                    "1",
                ]
            )

        assert excinfo.value.code == 1
        assert "above --limit" in capsys.readouterr().out

    def test_cli_sweep_rejects_experimental_without_flag(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(
                [
                    "train",
                    "sweep",
                    "--sweep-id",
                    "experimental",
                    "--project",
                    "director-project",
                    "--image",
                    "us-docker.pkg.dev/director/train:latest",
                    "--output-prefix",
                    "gs://director-artifacts/sweeps/experimental",
                    "--train-set",
                    "smoke=gs://director-data/train.jsonl",
                    "--model",
                    "roberta-large-mnli",
                    "--epochs",
                    "1",
                ]
            )

        assert excinfo.value.code == 1
        assert "experimental" in capsys.readouterr().out

    def test_cli_harvest_prints_report(self, tmp_path, capsys):
        scenario = tmp_path / "sweep" / "scenario-a"
        scenario.mkdir(parents=True)
        (scenario / "training_result.json").write_text(
            json.dumps({"best_balanced_accuracy": 0.82}),
            encoding="utf-8",
        )

        main(["train", "harvest", "--prefix-uri", str(tmp_path / "sweep")])

        data = json.loads(capsys.readouterr().out)
        assert data["result_count"] == 1
        assert data["best"]["scenario"] == "scenario-a"
        assert data["best"]["best_balanced_accuracy"] == 0.82


class TestManagedTrainingResults:
    def test_harvest_local_results_sorted_by_balanced_accuracy(self, tmp_path):
        sweep = tmp_path / "sweep"
        first = sweep / "natural100" / "model-a"
        second = sweep / "natural500" / "model-b"
        first.mkdir(parents=True)
        second.mkdir(parents=True)
        (first / "training_result.json").write_text(
            json.dumps(
                {
                    "best_balanced_accuracy": 0.61,
                    "epochs_completed": 1,
                    "train_samples": 100,
                    "eval_samples": 50,
                    "eval_metrics": {"eval_balanced_accuracy": 0.61},
                }
            ),
            encoding="utf-8",
        )
        (second / "training_result.json").write_text(
            json.dumps(
                {
                    "best_balanced_accuracy": 0.73,
                    "epochs_completed": 3,
                    "train_samples": 500,
                    "eval_samples": 50,
                    "final_loss": 0.2,
                }
            ),
            encoding="utf-8",
        )

        report = harvest_training_results(str(sweep))

        assert report.result_count == 2
        assert report.best is not None
        assert report.best.scenario == "natural500/model-b"
        assert report.results[0].best_balanced_accuracy == 0.73
        assert report.results[1].artifact_uri == str(first)


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
            headers={"X-Tenant-ID": "tenant-a"},
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
        assert body["tenant_id"] == "tenant-a"
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

    def test_managed_jobs_are_listed_by_tenant(self, client):
        submit = client.post(
            "/v1/finetune/managed/submit",
            headers={"X-Tenant-ID": "tenant-a"},
            json={
                "backend": "vertex",
                "dry_run": True,
                "dataset_uri": "gs://director-data/train.jsonl",
                "output_uri": "gs://director-artifacts/jobs/job-1",
                "project": "director-project",
                "container_image_uri": "us-docker.pkg.dev/director/train:latest",
            },
        )
        assert submit.status_code == 200

        own = client.get(
            "/v1/finetune/managed/jobs",
            headers={"X-Tenant-ID": "tenant-a"},
        )
        other = client.get(
            "/v1/finetune/managed/jobs",
            headers={"X-Tenant-ID": "tenant-b"},
        )

        assert own.status_code == 200
        assert own.json()["count"] == 1
        assert own.json()["jobs"][0]["job_id"] == submit.json()["job_id"]
        assert other.status_code == 200
        assert other.json()["count"] == 0

    def test_managed_status_returns_dry_run_record_without_backend_call(self, client):
        submit = client.post(
            "/v1/finetune/managed/submit",
            headers={"X-Tenant-ID": "tenant-a"},
            json={
                "backend": "vertex",
                "dry_run": True,
                "dataset_uri": "gs://director-data/train.jsonl",
                "output_uri": "gs://director-artifacts/jobs/job-1",
                "project": "director-project",
                "container_image_uri": "us-docker.pkg.dev/director/train:latest",
            },
        )
        assert submit.status_code == 200
        job_id = submit.json()["job_id"]

        with patch("director_ai.core.training.jobs.get_training_backend") as backend:
            status = client.post(
                "/v1/finetune/managed/status",
                headers={"X-Tenant-ID": "tenant-a"},
                json={"backend": "vertex", "job_id": job_id},
            )

        backend.assert_not_called()
        assert status.status_code == 200
        assert status.json()["state"] == "dry_run"

    def test_managed_status_rejects_cross_tenant_lookup(self, client):
        submit = client.post(
            "/v1/finetune/managed/submit",
            headers={"X-Tenant-ID": "tenant-a"},
            json={
                "backend": "vertex",
                "dry_run": True,
                "dataset_uri": "gs://director-data/train.jsonl",
                "output_uri": "gs://director-artifacts/jobs/job-1",
                "project": "director-project",
                "container_image_uri": "us-docker.pkg.dev/director/train:latest",
            },
        )

        status = client.post(
            "/v1/finetune/managed/status",
            headers={"X-Tenant-ID": "tenant-b"},
            json={"backend": "vertex", "job_id": submit.json()["job_id"]},
        )

        assert status.status_code == 404

    def test_managed_cancel_rejects_dry_run(self, client):
        submit = client.post(
            "/v1/finetune/managed/submit",
            headers={"X-Tenant-ID": "tenant-a"},
            json={
                "backend": "vertex",
                "dry_run": True,
                "dataset_uri": "gs://director-data/train.jsonl",
                "output_uri": "gs://director-artifacts/jobs/job-1",
                "project": "director-project",
                "container_image_uri": "us-docker.pkg.dev/director/train:latest",
            },
        )

        cancel = client.post(
            "/v1/finetune/managed/cancel",
            headers={"X-Tenant-ID": "tenant-a"},
            json={"backend": "vertex", "job_id": submit.json()["job_id"]},
        )

        assert cancel.status_code == 409

    def test_managed_status_and_cancel_call_backend_for_live_job(self, client):
        submitted = TrainingJobSubmission(
            backend="vertex",
            job_id="projects/p/locations/r/customJobs/123",
            state="submitted",
            dry_run=False,
            request={"job_spec": {}},
            submitted_at=1.0,
            console_uri="https://example.invalid/job/123",
        )
        fake_backend = MagicMock()
        fake_backend.status.return_value = TrainingJobStatus(
            backend="vertex",
            job_id=submitted.job_id,
            state="JOB_STATE_RUNNING",
        )
        fake_backend.cancel.return_value = TrainingJobStatus(
            backend="vertex",
            job_id=submitted.job_id,
            state="cancelled",
        )

        with patch(
            "director_ai.core.training.jobs.submit_training_job",
            return_value=submitted,
        ):
            submit = client.post(
                "/v1/finetune/managed/submit",
                headers={"X-Tenant-ID": "tenant-a"},
                json={
                    "backend": "vertex",
                    "dry_run": False,
                    "dataset_uri": "gs://director-data/train.jsonl",
                    "output_uri": "gs://director-artifacts/jobs/job-1",
                    "project": "director-project",
                    "container_image_uri": "us-docker.pkg.dev/director/train:latest",
                },
            )
        assert submit.status_code == 200

        with patch(
            "director_ai.core.training.jobs.get_training_backend",
            return_value=fake_backend,
        ):
            status = client.post(
                "/v1/finetune/managed/status",
                headers={"X-Tenant-ID": "tenant-a"},
                json={"backend": "vertex", "job_id": submitted.job_id},
            )
            cancel = client.post(
                "/v1/finetune/managed/cancel",
                headers={"X-Tenant-ID": "tenant-a"},
                json={"backend": "vertex", "job_id": submitted.job_id},
            )

        assert status.status_code == 200
        assert status.json()["state"] == "JOB_STATE_RUNNING"
        assert cancel.status_code == 200
        assert cancel.json()["state"] == "cancelled"
        fake_backend.status.assert_called_once_with(submitted.job_id)
        fake_backend.cancel.assert_called_once_with(submitted.job_id)

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
