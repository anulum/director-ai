# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Managed Training Job Tests

"""Multi-angle tests for managed training job specifications and callers."""

from __future__ import annotations

import builtins
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import director_ai._cli_train as cli_train
import director_ai.core.training.results as training_results
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
    shell_join,
    submit_training_job,
)
from director_ai.core.training.results import harvest_training_results
from director_ai.core.training.sweeps import (
    TrainingDatasetSplit,
    _scenario_id,
    _slug,
    build_training_sweep_plan,
)
from director_ai.core.training.vertex_runner import (
    _materialise_uri,
    _publish_dir,
    _publish_file,
    _split_gcs_uri,
    _storage_client,
)
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
    @pytest.mark.parametrize(
        ("hardware", "message"),
        [
            (TrainingHardware(machine_type=""), "machine_type is required"),
            (
                TrainingHardware(accelerator_count=-1),
                "accelerator_count must be >= 0",
            ),
            (
                TrainingHardware(accelerator_type="", accelerator_count=1),
                "accelerator_type is required",
            ),
            (TrainingHardware(boot_disk_gb=49), "boot_disk_gb must be at least 50"),
        ],
    )
    def test_hardware_validation_errors_are_actionable(self, hardware, message):
        with pytest.raises(ValueError, match=message):
            hardware.validate()

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"caller": "partner"}, "caller must be one of"),
            ({"task_type": "pretrain"}, "task_type must be one of"),
            ({"display_name": ""}, "display_name is required"),
            ({"dataset_uri": ""}, "dataset_uri is required"),
            ({"output_uri": ""}, "output_uri is required"),
            ({"epochs": 0}, "epochs must be >= 1"),
            ({"batch_size": 0}, "batch_size must be >= 1"),
            ({"learning_rate": 0.0}, "learning_rate must be > 0"),
            ({"timeout_minutes": 0}, "timeout_minutes must be between"),
            ({"timeout_minutes": 24 * 60 + 1}, "timeout_minutes must be between"),
        ],
    )
    def test_spec_validation_errors_are_actionable(self, overrides, message):
        spec = _vertex_spec(**overrides)
        with pytest.raises(ValueError, match=message):
            spec.validate("vertex")

    def test_spec_validation_rejects_unknown_backend(self):
        with pytest.raises(ValueError, match="backend must be one of"):
            _vertex_spec().validate("kubernetes")

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

    def test_suite_spec_redaction_skips_model_resolution(self):
        spec = TrainingJobSpec(
            display_name="suite",
            task_type="suite",
            caller="internal",
            dataset_uri="/tmp/input",
            output_uri="/tmp/out",
            base_model="not-in-registry",
            env={"PASSWORD": "secret"},
        )

        redacted = spec.to_redacted_dict()

        assert redacted["resolved_base_model"] == "not-in-registry"
        assert redacted["model_profile"] == {}
        assert redacted["env"]["PASSWORD"] == "<redacted>"


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

    def test_cpu_only_worker_pool_omits_accelerator_fields(self):
        spec = _vertex_spec(
            hardware=TrainingHardware(
                machine_type="n1-standard-4",
                accelerator_type="",
                accelerator_count=0,
            ),
            eval_uri=None,
            env={"MODE": "smoke"},
            labels={"TEAM_NAME": "Safety Ops", "": "ignored", "EMPTY": ""},
        )
        request = build_vertex_custom_job_request(spec)
        pool = request["job_spec"]["worker_pool_specs"][0]

        assert pool["machine_spec"] == {"machine_type": "n1-standard-4"}
        assert pool["container_spec"]["env"] == [{"name": "MODE", "value": "smoke"}]
        assert request["labels"]["team-name"] == "safety ops"
        assert "" not in request["labels"]


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

    def test_local_status_and_cancel_are_explicitly_synchronous(self):
        backend = LocalTrainingBackend()

        status = backend.status("local-123")
        cancelled = backend.cancel("local-123")

        assert status == TrainingJobStatus(
            backend="local",
            job_id="local-123",
            state="unknown",
        )
        assert cancelled.backend == "local"
        assert cancelled.job_id == "local-123"
        assert cancelled.state == "unsupported"
        assert "synchronously" in cancelled.error

    def test_local_dry_run_includes_eval_argument_and_shell_display(self, tmp_path):
        spec = TrainingJobSpec(
            display_name="local",
            dataset_uri=str(tmp_path / "train.jsonl"),
            output_uri=str(tmp_path / "model"),
            eval_uri=str(tmp_path / "eval.jsonl"),
        )

        result = submit_training_job(spec, backend="local", dry_run=True)

        assert "--eval" in result.request["command"]
        assert shell_join(["director-ai", "finetune", "path with spaces"]) == (
            "director-ai finetune 'path with spaces'"
        )

    def test_local_suite_execute_runs_pytest_args_and_reports_failure(self):
        spec = build_internal_suite_spec(
            suite="test_smoke",
            dataset_uri="/tmp/input",
            output_uri="/tmp/out",
            container_image_uri="python:3.12-slim",
        )

        with patch("pytest.main", return_value=0) as mock_pytest:
            result = submit_training_job(spec, backend="local", dry_run=False)
        mock_pytest.assert_called_once_with(["tests/test_smoke.py", "-q"])
        assert result.state == "completed"

        with (
            patch("pytest.main", return_value=2),
            pytest.raises(
                RuntimeError, match="local suite job failed with exit code 2"
            ),
        ):
            submit_training_job(spec, backend="local", dry_run=False)

    def test_local_suite_execute_accepts_plain_args(self):
        spec = TrainingJobSpec(
            display_name="suite",
            task_type="suite",
            caller="internal",
            dataset_uri="/tmp/input",
            output_uri="/tmp/out",
            args=["tests/test_smoke.py", "-q"],
        )

        with patch("pytest.main", return_value=0) as mock_pytest:
            submit_training_job(spec, backend="local", dry_run=False)

        mock_pytest.assert_called_once_with(["tests/test_smoke.py", "-q"])

    def test_vertex_status_and_cancel_use_sdk_job_lookup(self):
        class FakeJob:
            state = "JOB_STATE_RUNNING"

            def __init__(self):
                self.cancelled = False

            def cancel(self):
                self.cancelled = True
                cancelled_jobs.append(self)

        fake_job = FakeJob()
        cancelled_jobs = []
        fake_module = SimpleNamespace(
            CustomJob=SimpleNamespace(get=MagicMock(return_value=fake_job))
        )

        with patch("importlib.import_module", return_value=fake_module):
            status = VertexTrainingBackend().status("projects/p/locations/r/jobs/1")
            cancelled = VertexTrainingBackend().cancel("projects/p/locations/r/jobs/1")

        assert status == TrainingJobStatus(
            backend="vertex",
            job_id="projects/p/locations/r/jobs/1",
            state="JOB_STATE_RUNNING",
        )
        assert cancelled.state == "cancelled"
        assert cancelled_jobs == [fake_job]
        assert fake_module.CustomJob.get.call_count == 2

    def test_vertex_status_defaults_unknown_when_state_missing(self):
        fake_module = SimpleNamespace(
            CustomJob=SimpleNamespace(get=MagicMock(return_value=SimpleNamespace()))
        )

        with patch("importlib.import_module", return_value=fake_module):
            status = VertexTrainingBackend().status("job-without-state")

        assert status.state == "unknown"

    def test_internal_suite_requires_name(self):
        with pytest.raises(ValueError, match="suite is required"):
            build_internal_suite_spec(
                suite="",
                dataset_uri="/tmp/input",
                output_uri="/tmp/out",
            )

    def test_local_finetune_execution_passes_eval_and_config(self, tmp_path):
        spec = TrainingJobSpec(
            display_name="local",
            dataset_uri=str(tmp_path / "train.jsonl"),
            output_uri=str(tmp_path / "model"),
            eval_uri=str(tmp_path / "eval.jsonl"),
            epochs=4,
            batch_size=3,
            learning_rate=1e-5,
        )

        with patch("director_ai.core.training.finetune.finetune_nli") as mock_finetune:
            submit_training_job(spec, backend="local", dry_run=False)

        _, kwargs = mock_finetune.call_args
        assert kwargs["eval_path"] == str(tmp_path / "eval.jsonl")
        assert kwargs["config"].output_dir == str(tmp_path / "model")
        assert kwargs["config"].epochs == 4
        assert kwargs["config"].batch_size == 3
        assert kwargs["config"].learning_rate == 1e-5


class TestVertexRunner:
    def test_rejects_malformed_gcs_uri(self):
        with pytest.raises(ValueError, match="invalid GCS URI"):
            _split_gcs_uri("gs://bucket")

    def test_rejects_non_gcs_uri_in_gcs_splitter(self):
        with pytest.raises(ValueError, match="invalid GCS URI"):
            _split_gcs_uri("https://storage.local/bucket/object")

    def test_materialise_gcs_downloads_to_destination(self, tmp_path, monkeypatch):
        calls = []

        class _Blob:
            def __init__(self, bucket: str, name: str) -> None:
                self.bucket = bucket
                self.name = name

            def download_to_filename(self, filename: str) -> None:
                calls.append(("download", self.bucket, self.name, filename))

        class _Bucket:
            def __init__(self, name: str) -> None:
                self.name = name

            def blob(self, name: str) -> _Blob:
                return _Blob(self.name, name)

        fake_client = SimpleNamespace(bucket=lambda name: _Bucket(name))
        monkeypatch.setattr(
            "director_ai.core.training.vertex_runner._storage_client",
            lambda: fake_client,
        )

        destination = tmp_path / "nested" / "train.jsonl"

        assert _materialise_uri("gs://training-data/path/train.jsonl", destination) == (
            destination
        )
        assert calls == [
            ("download", "training-data", "path/train.jsonl", str(destination))
        ]
        assert destination.parent.exists()

    def test_publish_file_uploads_gcs_object(self, tmp_path, monkeypatch):
        calls = []
        source = tmp_path / "training_result.json"
        source.write_text('{"ok": true}', encoding="utf-8")

        class _Blob:
            def __init__(self, bucket: str, name: str) -> None:
                self.bucket = bucket
                self.name = name

            def upload_from_filename(self, filename: str) -> None:
                calls.append(("upload", self.bucket, self.name, filename))

        class _Bucket:
            def __init__(self, name: str) -> None:
                self.name = name

            def blob(self, name: str) -> _Blob:
                return _Blob(self.name, name)

        monkeypatch.setattr(
            "director_ai.core.training.vertex_runner._storage_client",
            lambda: SimpleNamespace(bucket=lambda name: _Bucket(name)),
        )

        _publish_file(source, "gs://training-output/job/training_result.json")

        assert calls == [
            (
                "upload",
                "training-output",
                "job/training_result.json",
                str(source),
            )
        ]

    def test_publish_dir_uploads_only_files_to_gcs_prefix(self, tmp_path, monkeypatch):
        calls = []
        model_dir = tmp_path / "model"
        (model_dir / "nested").mkdir(parents=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "nested" / "weights.bin").write_bytes(b"weights")

        class _Blob:
            def __init__(self, name: str) -> None:
                self.name = name

            def upload_from_filename(self, filename: str) -> None:
                calls.append((self.name, filename))

        class _Bucket:
            def blob(self, name: str) -> _Blob:
                return _Blob(name)

        monkeypatch.setattr(
            "director_ai.core.training.vertex_runner._storage_client",
            lambda: SimpleNamespace(bucket=lambda name: _Bucket()),
        )

        _publish_dir(model_dir, "gs://training-output/job/model/")

        assert calls == [
            ("job/model/config.json", str(model_dir / "config.json")),
            ("job/model/nested/weights.bin", str(model_dir / "nested" / "weights.bin")),
        ]

    def test_publish_dir_replaces_existing_local_destination(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("new", encoding="utf-8")
        destination = tmp_path / "published"
        destination.mkdir()
        (destination / "stale.txt").write_text("old", encoding="utf-8")

        _publish_dir(model_dir, str(destination))

        assert not (destination / "stale.txt").exists()
        assert (destination / "config.json").read_text(encoding="utf-8") == "new"

    def test_storage_client_uses_google_storage_client(self, monkeypatch):
        class _Client:
            pass

        storage_module = types.ModuleType("google.cloud.storage")
        storage_module.Client = _Client
        cloud_module = types.ModuleType("google.cloud")
        cloud_module.storage = storage_module
        google_module = types.ModuleType("google")
        google_module.cloud = cloud_module
        monkeypatch.setitem(sys.modules, "google", google_module)
        monkeypatch.setitem(sys.modules, "google.cloud", cloud_module)
        monkeypatch.setitem(sys.modules, "google.cloud.storage", storage_module)

        assert isinstance(_storage_client(), _Client)

    def test_storage_client_import_error_mentions_required_package(self, monkeypatch):
        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "google.cloud.storage":
                raise ImportError("missing storage client")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        with pytest.raises(ImportError, match="google-cloud-storage"):
            _storage_client()

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
    def test_dataset_split_validation_and_serialisation(self):
        split = TrainingDatasetSplit(
            name="legal-smoke",
            train_uri="gs://director-data/legal-train.jsonl",
            eval_uri=None,
        )

        split.validate()

        assert split.to_dict() == {
            "name": "legal-smoke",
            "train_uri": "gs://director-data/legal-train.jsonl",
            "eval_uri": None,
        }
        with pytest.raises(ValueError, match="dataset split name"):
            TrainingDatasetSplit(name="", train_uri="gs://x/train.jsonl").validate()
        with pytest.raises(ValueError, match="dataset split train_uri"):
            TrainingDatasetSplit(name="legal", train_uri="").validate()

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

    def test_sweep_plan_defaults_to_registry_batch_size_and_learning_rate(self):
        plan = build_training_sweep_plan(
            sweep_id="sweep-defaults",
            datasets=[
                TrainingDatasetSplit(name="QA Smoke", train_uri="gs://data/train")
            ],
            base_models=["factcg-deberta-v3-large"],
            epochs=[2],
            output_prefix="gs://director-artifacts/sweeps/defaults/",
        )
        payload = plan.to_dict()
        scenario = plan.scenarios[0]

        assert scenario.batch_size == 16
        assert scenario.learning_rate == pytest.approx(2e-5)
        assert scenario.output_uri == (
            "gs://director-artifacts/sweeps/defaults/"
            "qa-smoke-factcg-deberta-v3-large-e2-b16"
        )
        assert payload["scenario_count"] == 1
        assert payload["scenarios"][0]["dataset"]["name"] == "QA Smoke"

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

    def test_sweep_spec_conversion_carries_security_and_network_controls(self):
        plan = build_training_sweep_plan(
            sweep_id="sweep-secure",
            datasets=[TrainingDatasetSplit(name="secure", train_uri="gs://data/train")],
            base_models=["factcg-deberta-v3-large"],
            epochs=[1],
            output_prefix="gs://out",
        )

        spec = plan.to_specs(
            project="director-project",
            region="europe-west4",
            container_image_uri="us-docker.pkg.dev/director/train:latest",
            hardware=TrainingHardware(machine_type="n1-standard-8"),
            timeout_minutes=30,
            caller="pilot",
            allow_experimental_model=True,
            service_account="trainer@director-project.iam.gserviceaccount.com",
            network="projects/director/global/networks/private",
        )[0]

        assert spec.caller == "pilot"
        assert spec.allow_experimental_model is True
        assert (
            spec.service_account == "trainer@director-project.iam.gserviceaccount.com"
        )
        assert spec.network == "projects/director/global/networks/private"

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"sweep_id": ""}, "sweep_id"),
            ({"datasets": []}, "at least one dataset"),
            ({"base_models": []}, "at least one base model"),
            ({"epochs": []}, "at least one epoch"),
            ({"output_prefix": ""}, "output_prefix"),
            ({"epochs": [0]}, "epochs values"),
            ({"batch_sizes": [0]}, "batch_sizes values"),
        ],
    )
    def test_sweep_plan_rejects_incomplete_or_unsafe_matrices(
        self,
        overrides,
        message,
    ):
        values = {
            "sweep_id": "sweep-invalid",
            "datasets": [
                TrainingDatasetSplit(name="smoke", train_uri="gs://data/train")
            ],
            "base_models": ["factcg-deberta-v3-large"],
            "epochs": [1],
            "batch_sizes": [1],
            "output_prefix": "gs://out",
        }
        values.update(overrides)

        with pytest.raises(ValueError, match=message):
            build_training_sweep_plan(**values)

    def test_scenario_id_slugging_is_stable_for_storage_paths_and_labels(self):
        assert _slug("___") == "unnamed"
        assert _slug("A_Long Dataset/Name With Symbols!" * 3) == (
            "a-long-dataset-name-with-symbols-a-long-dataset-"
        )
        assert (
            _scenario_id(
                dataset_name="Legal QA",
                model_alias="FactCG_DeBERTa",
                epochs=3,
                batch_size=8,
            )
            == "legal-qa-factcg-deberta-e3-b8"
        )


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

    def test_unknown_train_subcommand_exits_with_help(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["train", "teleport"])

        assert excinfo.value.code == 1
        out = capsys.readouterr().out
        assert "Unknown train subcommand: teleport" in out
        assert "benchmark-models" in out

    def test_submit_parser_flags_aliases_and_required_errors(self, capsys):
        opts = cli_train._parse_submit_args(
            [
                "--train-uri",
                "gs://data/train.jsonl",
                "--output-uri",
                "gs://out/job",
                "--dry-run",
                "--allow-experimental-model",
                "--service-account",
                "trainer@example.iam.gserviceaccount.com",
            ]
        )
        assert opts["dataset_uri"] == "gs://data/train.jsonl"
        assert opts["execute"] is False
        assert opts["allow_experimental_model"] is True
        assert opts["service_account"] == "trainer@example.iam.gserviceaccount.com"

        with pytest.raises(SystemExit) as excinfo:
            cli_train._parse_submit_args(["--dataset-uri", "gs://data/train.jsonl"])
        assert excinfo.value.code == 1
        assert "--output-uri is required" in capsys.readouterr().out

        with pytest.raises(SystemExit) as excinfo:
            cli_train._parse_submit_args(["--output-uri", "gs://out/job"])
        assert excinfo.value.code == 1
        assert "--dataset-uri is required" in capsys.readouterr().out

        with pytest.raises(SystemExit) as excinfo:
            cli_train._parse_submit_args(["--dataset-uri"])
        assert excinfo.value.code == 1
        assert "Unknown or incomplete option: --dataset-uri" in capsys.readouterr().out

    def test_sweep_parser_covers_flags_defaults_and_validation_errors(self, capsys):
        opts = cli_train._parse_sweep_args(
            [
                "--execute",
                "--dry-run",
                "--allow-experimental-model",
                "--train-set",
                "legal=gs://data/legal-train.jsonl",
                "--eval-set",
                "legal=gs://data/legal-eval.jsonl",
                "--model",
                "custom-model",
                "--epochs",
                "2",
                "--output-prefix",
                "gs://out/sweeps/legal",
                "--project",
                "director-project",
                "--image",
                "us-docker.pkg.dev/director/train:latest",
                "--limit",
                "4",
            ]
        )
        assert opts["execute"] is False
        assert opts["allow_experimental_model"] is True
        assert opts["batch_sizes"] == ["1"]
        assert opts["train_sets"] == {"legal": "gs://data/legal-train.jsonl"}
        assert opts["eval_sets"] == {"legal": "gs://data/legal-eval.jsonl"}

        required_cases = [
            ([], "at least one --train-set"),
            (["--train-set", "a=gs://train"], "at least one --model"),
            (
                ["--train-set", "a=gs://train", "--model", "m"],
                "at least one --epochs",
            ),
            (
                ["--train-set", "a=gs://train", "--model", "m", "--epochs", "1"],
                "--output-prefix is required",
            ),
            (
                [
                    "--train-set",
                    "a=gs://train",
                    "--model",
                    "m",
                    "--epochs",
                    "1",
                    "--output-prefix",
                    "gs://out",
                ],
                "--project is required",
            ),
            (
                [
                    "--train-set",
                    "a=gs://train",
                    "--model",
                    "m",
                    "--epochs",
                    "1",
                    "--output-prefix",
                    "gs://out",
                    "--project",
                    "director-project",
                ],
                "--image is required",
            ),
        ]
        for args, message in required_cases:
            with pytest.raises(SystemExit):
                cli_train._parse_sweep_args(args)
            assert message in capsys.readouterr().out

        with pytest.raises(SystemExit):
            cli_train._parse_sweep_args(["--train-set"])
        assert "Unknown or incomplete option: --train-set" in capsys.readouterr().out

        with pytest.raises(SystemExit):
            cli_train._parse_sweep_args(["--unknown"])
        assert "Unknown train sweep option: --unknown" in capsys.readouterr().out

    @pytest.mark.parametrize("value", ["missing-equals", "=gs://train", "name="])
    def test_split_named_uri_rejects_malformed_values(self, value, capsys):
        with pytest.raises(SystemExit) as excinfo:
            cli_train._split_named_uri("--train-set", value)

        assert excinfo.value.code == 1
        assert "--train-set must use name=uri" in capsys.readouterr().out

    def test_models_benchmark_and_harvest_parsers_reject_bad_inputs(self, capsys):
        with pytest.raises(SystemExit):
            cli_train._cmd_train_models(["--json"])
        assert "Unknown train models option: --json" in capsys.readouterr().out

        with pytest.raises(SystemExit):
            cli_train._parse_harvest_args(["--prefix-uri"])
        assert "Unknown or incomplete option: --prefix-uri" in capsys.readouterr().out

        with pytest.raises(SystemExit):
            cli_train._parse_harvest_args(["--bad"])
        assert "Unknown train harvest option: --bad" in capsys.readouterr().out

        with pytest.raises(SystemExit):
            cli_train._parse_harvest_args([])
        assert "--prefix-uri is required" in capsys.readouterr().out

        with pytest.raises(SystemExit):
            cli_train._parse_benchmark_models_args(["--model"])
        assert "Unknown or incomplete option: --model" in capsys.readouterr().out

        with pytest.raises(SystemExit):
            cli_train._parse_benchmark_models_args(["--model", "missing-equals"])
        assert "--model must use alias=artifact_path" in capsys.readouterr().out

        with pytest.raises(SystemExit):
            cli_train._parse_benchmark_models_args(["--model", "=/tmp/model"])
        assert "--model must use alias=artifact_path" in capsys.readouterr().out

        with pytest.raises(SystemExit):
            cli_train._parse_benchmark_models_args(["--unexpected"])
        assert (
            "Unknown train benchmark-models option: --unexpected"
            in capsys.readouterr().out
        )

        with pytest.raises(SystemExit):
            cli_train._parse_benchmark_models_args([])
        assert "at least one --model" in capsys.readouterr().out

    def test_benchmark_parser_accepts_optional_paths_batch_and_experimental_flag(self):
        opts = cli_train._parse_benchmark_models_args(
            [
                "--allow-experimental-model",
                "--model",
                "candidate=/models/candidate",
                "--general-uri",
                "/data/general.jsonl",
                "--eval-uri",
                "/data/eval.jsonl",
                "--batch-size",
                "8",
            ]
        )

        assert opts == {
            "models": {"candidate": "/models/candidate"},
            "general_path": "/data/general.jsonl",
            "eval_path": "/data/eval.jsonl",
            "batch_size": "8",
            "allow_experimental_model": True,
        }

    def test_cli_sweep_submission_failure_names_failed_spec(self, monkeypatch, capsys):
        import director_ai.core.training.jobs as jobs_mod

        def fail_submission(spec, *, backend, dry_run):
            assert backend == "vertex"
            assert dry_run is True
            raise RuntimeError(f"quota denied for {spec.display_name}")

        monkeypatch.setattr(jobs_mod, "submit_training_job", fail_submission)

        with pytest.raises(SystemExit) as excinfo:
            main(
                [
                    "train",
                    "sweep",
                    "--project",
                    "director-project",
                    "--image",
                    "us-docker.pkg.dev/director/train:latest",
                    "--output-prefix",
                    "gs://director-artifacts/sweeps/failing",
                    "--train-set",
                    "smoke=gs://director-data/train.jsonl",
                    "--model",
                    "factcg-deberta-v3-large",
                    "--epochs",
                    "1",
                ]
            )

        assert excinfo.value.code == 1
        out = capsys.readouterr().out
        assert "sweep job submission failed" in out
        assert "director-ai-managed-sweep" in out

    def test_cli_harvest_failure_is_reported(self, monkeypatch, capsys):
        monkeypatch.setattr(
            training_results,
            "harvest_training_results",
            lambda prefix: (_ for _ in ()).throw(RuntimeError(f"denied {prefix}")),
        )

        with pytest.raises(SystemExit) as excinfo:
            main(["train", "harvest", "--prefix-uri", "gs://private/results"])

        assert excinfo.value.code == 1
        assert "training result harvest failed" in capsys.readouterr().out

    def test_cli_type_guards_reject_non_string_values(self, capsys):
        with pytest.raises(SystemExit):
            cli_train._as_str({"project": 7}, "project")
        assert "project must be a string" in capsys.readouterr().out

        with pytest.raises(SystemExit):
            cli_train._as_optional_str({"network": object()}, "network")
        assert "network must be a string" in capsys.readouterr().out


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

    def test_harvest_empty_local_prefix_returns_empty_report(self, tmp_path):
        empty_sweep = tmp_path / "empty-sweep"
        empty_sweep.mkdir()

        report = harvest_training_results(str(empty_sweep))

        assert report.to_dict() == {
            "prefix_uri": str(empty_sweep),
            "result_count": 0,
            "best": None,
            "results": [],
        }

    def test_harvest_local_rejects_missing_prefix_and_file_prefix(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="training result prefix"):
            harvest_training_results(str(tmp_path / "missing"))

        file_prefix = tmp_path / "not-a-directory"
        file_prefix.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a directory"):
            harvest_training_results(str(file_prefix))

    def test_harvest_local_rejects_invalid_and_non_object_result_json(self, tmp_path):
        invalid = tmp_path / "invalid" / "scenario"
        invalid.mkdir(parents=True)
        (invalid / "training_result.json").write_text("{not-json", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid training result JSON"):
            harvest_training_results(str(tmp_path / "invalid"))

        non_object = tmp_path / "non-object" / "scenario"
        non_object.mkdir(parents=True)
        (non_object / "training_result.json").write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(ValueError, match="must be an object"):
            harvest_training_results(str(tmp_path / "non-object"))

    def test_root_level_local_result_uses_parent_name_as_scenario(self, tmp_path):
        sweep = tmp_path / "sweep"
        sweep.mkdir()
        (sweep / "training_result.json").write_text(
            json.dumps(
                {
                    "best_balanced_accuracy": "0.65",
                    "final_loss": "0.4",
                    "epochs_completed": "2",
                    "train_samples": "30",
                    "eval_samples": "10",
                    "eval_metrics": {"balanced_accuracy": 0.65},
                }
            ),
            encoding="utf-8",
        )

        report = harvest_training_results(str(sweep))

        assert report.result_count == 1
        assert report.best is not None
        assert report.best.scenario == "sweep"
        assert report.best.artifact_uri == str(sweep)
        assert report.best.raw["best_balanced_accuracy"] == "0.65"
        assert report.best.to_dict()["eval_metrics"] == {"balanced_accuracy": 0.65}

    def test_harvest_gcs_results_filters_blobs_and_sorts_records(self, monkeypatch):
        class FakeBlob:
            def __init__(self, name, payload):
                self.name = name
                self._payload = payload

            def download_as_text(self):
                return json.dumps(self._payload)

        class FakeClient:
            def __init__(self):
                self.bucket_names: list[str] = []
                self.list_prefixes: list[str] = []

            def bucket(self, name):
                self.bucket_names.append(name)
                return f"bucket:{name}"

            def list_blobs(self, bucket, *, prefix):
                self.list_prefixes.append(prefix)
                assert bucket == "bucket:director-artifacts"
                return [
                    FakeBlob(
                        "runs/sweep-a/training_result.json",
                        {"best_balanced_accuracy": 0.7},
                    ),
                    FakeBlob(
                        "runs/sweep-b/training_result.json",
                        {"best_balanced_accuracy": 0.9},
                    ),
                    FakeBlob("runs/sweep-b/metrics.json", {"ignored": True}),
                ]

        fake_client = FakeClient()
        monkeypatch.setattr(training_results, "_storage_client", lambda: fake_client)

        report = harvest_training_results("gs://director-artifacts/runs")

        assert fake_client.bucket_names == ["director-artifacts"]
        assert fake_client.list_prefixes == ["runs/"]
        assert report.prefix_uri == "gs://director-artifacts/runs"
        assert [record.scenario for record in report.results] == ["sweep-b", "sweep-a"]
        assert report.best is not None
        assert (
            report.best.result_uri
            == "gs://director-artifacts/runs/sweep-b/training_result.json"
        )
        assert report.best.artifact_uri == "gs://director-artifacts/runs/sweep-b"

    def test_gcs_uri_helpers_validate_scheme_and_scenario_fallback(self):
        assert training_results._is_gcs_uri("gs://bucket/path")
        assert not training_results._is_gcs_uri("/tmp/path")
        assert training_results._split_gcs_uri("gs://bucket/path/to/results") == (
            "bucket",
            "path/to/results",
        )
        with pytest.raises(ValueError, match="invalid GCS URI"):
            training_results._split_gcs_uri("https://bucket/path")

        assert (
            training_results._scenario_from_gcs_blob(
                "scenario-alone/training_result.json", ""
            )
            == "scenario-alone"
        )

    def test_storage_client_imports_google_storage_client(self, monkeypatch):
        calls: list[str] = []

        class FakeClient:
            def __init__(self):
                calls.append("constructed")

        storage_module = types.ModuleType("google.cloud.storage")
        storage_module.Client = FakeClient
        cloud_module = types.ModuleType("google.cloud")
        cloud_module.storage = storage_module
        google_module = types.ModuleType("google")
        google_module.cloud = cloud_module
        monkeypatch.setitem(sys.modules, "google", google_module)
        monkeypatch.setitem(sys.modules, "google.cloud", cloud_module)
        monkeypatch.setitem(sys.modules, "google.cloud.storage", storage_module)

        client = training_results._storage_client()

        assert isinstance(client, FakeClient)
        assert calls == ["constructed"]


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
