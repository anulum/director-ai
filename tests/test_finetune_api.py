# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Fine-tuning API Tests
"""Multi-angle tests for fine-tuning API pipeline.

Covers: finetune config, dataset loading, training, evaluation, checkpoint
management, LoRA config, pipeline integration with judge training, and
performance documentation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="fastapi required for finetune API tests")

from fastapi import APIRouter
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import director_ai.finetune_api as finetune_api_module
from director_ai.core.training.jobs import (
    TrainingJobSpec,
    TrainingJobStatus,
    TrainingJobSubmission,
)
from director_ai.finetune_api import (
    _MAX_CONCURRENT_JOBS,
    FinetuneJob,
    ManagedTrainingRecord,
    _JobStore,
    _managed_record_to_dict,
    _ManagedJobStore,
    _run_training_worker,
    create_finetune_router,
)

_Store = TypeVar("_Store", _JobStore, _ManagedJobStore)


def _closure_store(router: APIRouter, store_type: type[_Store]) -> _Store:
    """Return a router-local store captured by endpoint closures."""
    routes = list(router.routes)
    while routes:
        route = routes.pop()
        child = getattr(route, "original_router", None)
        if isinstance(child, APIRouter):
            routes.extend(child.routes)
            continue
        endpoint = getattr(route, "endpoint", None)
        closure = getattr(endpoint, "__closure__", None)
        if not closure:
            continue
        for cell in closure:
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if isinstance(value, store_type):
                return value
    raise AssertionError(f"{store_type.__name__} not found in router closures")


class TestJobStore:
    def test_create_and_get(self) -> None:
        store = _JobStore()
        assert store is not None
        job = store.create({"epochs": 3})
        assert job.state == "pending"
        assert job.job_id
        fetched = store.get(job.job_id)
        assert fetched is job

    def test_get_nonexistent(self) -> None:
        store = _JobStore()
        assert store.get("does-not-exist") is None

    def test_list_all(self) -> None:
        store = _JobStore()
        store.create({"epochs": 1})
        store.create({"epochs": 2})
        jobs = store.list_all()
        assert len(jobs) == 2

    def test_delete(self) -> None:
        store = _JobStore()
        assert store is not None
        job = store.create({"epochs": 3})
        assert store.delete(job.job_id)
        assert store.get(job.job_id) is None

    def test_delete_nonexistent(self) -> None:
        store = _JobStore()
        assert not store.delete("nope")

    def test_concurrent_job_limit(self) -> None:
        store = _JobStore()
        for _i in range(_MAX_CONCURRENT_JOBS):
            assert store is not None
            job = store.create({"epochs": 1})
            job.state = "training"
        with pytest.raises(ValueError, match="Too many"):
            store.create({"epochs": 1})


class TestManagedJobStore:
    def test_records_are_tenant_scoped_and_serializable(self) -> None:
        store = _ManagedJobStore()
        record = ManagedTrainingRecord(
            job_id="job-1",
            backend="portable",
            state="submitted",
            tenant_id="tenant-a",
            dry_run=False,
            submitted_at=123.0,
            display_name="training job",
            output_uri="gs://bucket/out",
            console_uri="https://console.example/job",
        )

        store.add(record)

        assert store.get("tenant-a", "job-1") is record
        assert store.get("tenant-b", "job-1") is None
        assert store.get("tenant-a", "missing") is None
        assert store.list_for_tenant("tenant-a") == [record]
        assert store.list_for_tenant("tenant-b") == []

        payload = _managed_record_to_dict(record)
        assert payload["job_id"] == "job-1"
        assert payload["tenant_id"] == "tenant-a"
        assert payload["console_uri"] == "https://console.example/job"

    def test_update_state_respects_tenant_ownership(self) -> None:
        store = _ManagedJobStore()
        record = ManagedTrainingRecord(
            job_id="job-1",
            backend="portable",
            state="submitted",
            tenant_id="tenant-a",
            dry_run=False,
            submitted_at=123.0,
            display_name="training job",
            output_uri="gs://bucket/out",
        )
        store.add(record)

        assert store.update_state("tenant-b", "job-1", "failed") is None
        assert store.update_state("tenant-a", "missing", "failed") is None
        updated = store.update_state(
            "tenant-a",
            "job-1",
            "failed",
            error="backend down",
        )

        assert updated is record
        assert record.state == "failed"
        assert record.error == "backend down"


class TestFinetuneJob:
    def test_defaults(self) -> None:
        job = FinetuneJob(job_id="test-123")
        assert job.state == "pending"
        assert job.progress == 0.0
        assert job.activated is False
        assert job.error == ""

    def test_state_transitions(self) -> None:
        job = FinetuneJob(job_id="test-456")
        job.state = "training"
        job.progress = 0.5
        job.state = "completed"
        job.progress = 1.0
        assert job.state == "completed"
        assert job.progress == 1.0


class TestCreateRouter:
    def test_router_creates(self, tmp_path: Path) -> None:
        router = create_finetune_router(models_dir=tmp_path / "models")
        assert router is not None
        routes = [r.path for r in router.routes if isinstance(r, APIRoute)]
        assert "/validate" in routes
        assert "/start" in routes
        assert "/{job_id}" in routes
        assert "/{job_id}/result" in routes
        assert "/{job_id}/activate" in routes
        assert "/{job_id}/rollback" in routes
        assert "/" in routes

    def test_models_dir_created(self, tmp_path: Path) -> None:
        models_dir = tmp_path / "new_models"
        create_finetune_router(models_dir=models_dir)
        assert models_dir.exists()
        assert (models_dir / "_uploads").exists()

    def test_router_uses_default_models_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        default_dir = tmp_path / "default_models"
        monkeypatch.setattr(finetune_api_module, "_DEFAULT_MODELS_DIR", default_dir)

        create_finetune_router()

        assert default_dir.exists()
        assert (default_dir / "_uploads").exists()

    def test_router_reports_missing_fastapi(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(finetune_api_module, "_FASTAPI_AVAILABLE", False)

        with pytest.raises(ImportError, match="director-ai\\[server\\]"):
            create_finetune_router()

    def test_module_import_without_fastapi_marks_router_unavailable(self) -> None:
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(
            "director_ai._finetune_api_no_fastapi",
            finetune_api_module.__file__,
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"fastapi": None}):
            spec.loader.exec_module(module)

        assert module._FASTAPI_AVAILABLE is False
        with pytest.raises(ImportError, match="director-ai\\[server\\]"):
            module.create_finetune_router()

    def test_router_warns_when_models_dir_cannot_be_created(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        target = (tmp_path / "readonly_models").resolve()
        original_mkdir = Path.mkdir
        calls = {"count": 0}

        def guarded_mkdir(
            path: Path,
            mode: int = 0o777,
            parents: bool = False,
            exist_ok: bool = False,
        ) -> None:
            if path == target and calls["count"] == 0:
                calls["count"] += 1
                raise PermissionError("read-only")
            return original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

        monkeypatch.setattr(Path, "mkdir", guarded_mkdir)

        router = create_finetune_router(models_dir=target)

        assert router is not None
        assert "read-only filesystem" in caplog.text
        assert (target / "_uploads").exists()


class TestRouterEndpoints:
    """Integration tests using FastAPI TestClient."""

    @pytest.fixture
    def client(self, tmp_path: Path) -> TestClient:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        router = create_finetune_router(
            models_dir=tmp_path / "models",
            operator_api_keys=("operator-test-key",),
        )
        app.include_router(router, prefix="/v1/finetune")
        return TestClient(app, headers={"X-API-Key": "operator-test-key"})

    def _make_jsonl_bytes(self, n_pos: int = 300, n_neg: int = 300) -> bytes:
        rows = []
        for i in range(n_pos):
            rows.append(
                {"premise": f"Fact {i}.", "hypothesis": f"Claim {i}.", "label": 1},
            )
        for i in range(n_neg):
            rows.append(
                {"premise": f"Source {i}.", "hypothesis": f"Wrong {i}.", "label": 0},
            )
        return ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")

    def test_validate_valid_data(self, client: TestClient) -> None:
        data = self._make_jsonl_bytes()
        resp = client.post(
            "/v1/finetune/validate",
            files={"file": ("train.jsonl", data, "application/jsonl")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_valid"]
        assert body["total_samples"] == 600
        assert body["estimated_cost_usd"] > 0

    def test_validate_invalid_data(self, client: TestClient) -> None:
        data = b'{"premise": "a"}\n{"hypothesis": "b"}\n'
        resp = client.post(
            "/v1/finetune/validate",
            files={"file": ("bad.jsonl", data, "application/jsonl")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert not body["is_valid"]
        assert len(body["errors"]) > 0

    def test_start_rejects_bad_data(self, client: TestClient) -> None:
        data = b"not json\n"
        resp = client.post(
            "/v1/finetune/start",
            files={"file": ("bad.jsonl", data, "application/jsonl")},
        )
        assert resp.status_code == 422

    def test_list_models_empty(self, client: TestClient) -> None:
        resp = client.get("/v1/finetune/")
        assert resp.status_code == 200
        assert resp.json()["models"] == []

    def test_get_nonexistent_job(self, client: TestClient) -> None:
        resp = client.get("/v1/finetune/nonexistent")
        assert resp.status_code == 404

    def test_activate_nonexistent_job(self, client: TestClient) -> None:
        resp = client.post("/v1/finetune/nonexistent/activate")
        assert resp.status_code == 404

    def test_rollback_nonexistent_job(self, client: TestClient) -> None:
        resp = client.post("/v1/finetune/nonexistent/rollback")
        assert resp.status_code == 404

    def test_delete_nonexistent_job(self, client: TestClient) -> None:
        resp = client.delete("/v1/finetune/nonexistent")
        assert resp.status_code == 404

    def test_result_nonexistent_job(self, client: TestClient) -> None:
        resp = client.get("/v1/finetune/nonexistent/result")
        assert resp.status_code == 404

    def test_managed_jobs_reject_invalid_tenant_header(
        self, client: TestClient
    ) -> None:
        resp = client.get(
            "/v1/finetune/managed/jobs",
            headers={"X-Tenant-ID": "../bad"},
        )
        assert resp.status_code == 400


class TestManagedTrainingEndpoints:
    """Managed-training endpoint contracts with backend calls stubbed."""

    @pytest.fixture
    def client_and_router(self, tmp_path: Path) -> tuple[TestClient, APIRouter]:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        router = create_finetune_router(
            models_dir=tmp_path / "models",
            operator_api_keys=("operator-test-key",),
        )
        app.include_router(router, prefix="/v1/finetune")
        store = _closure_store(router, _JobStore)
        job = store.create({"base_model": "factcg-deberta-v3-large"})
        job.state = "completed"
        model = tmp_path / "models" / "candidate"
        model.mkdir()
        job.model_path = str(model)
        store.save(job)
        datasets = tmp_path / "models" / "_benchmark_datasets"
        datasets.mkdir()
        (datasets / ("b" * 32 + ".jsonl")).write_text(
            '{"premise":"Evidence","hypothesis":"Claim","label":1}\n',
            encoding="utf-8",
        )
        return TestClient(app, headers={"X-API-Key": "operator-test-key"}), router

    def _submit_payload(self, **overrides: object) -> dict[str, object]:
        payload = {
            "backend": "portable",
            "dry_run": True,
            "display_name": "tenant-a-train",
            "dataset_uri": "gs://tenant-a/data/train.jsonl",
            "output_uri": "gs://tenant-a/models/out",
            "container_image_uri": "registry.example/director/train:latest",
            "epochs": 1,
            "batch_size": 4,
            "accelerator_count": 0,
            "boot_disk_gb": 100,
        }
        payload.update(overrides)
        return payload

    def test_submit_and_list_managed_training_jobs(
        self,
        client_and_router: tuple[TestClient, APIRouter],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import director_ai.core.training.jobs as jobs_module
        from director_ai.core.training.jobs import TrainingJobSubmission

        client, _router = client_and_router

        def fake_submit(
            spec: TrainingJobSpec, *, backend: str, dry_run: bool
        ) -> TrainingJobSubmission:
            assert spec.display_name == "tenant-a-train"
            assert backend == "portable"
            assert dry_run is True
            return TrainingJobSubmission(
                backend=backend,
                job_id="portable-1",
                state="dry_run",
                dry_run=True,
                request={"display_name": spec.display_name},
                submitted_at=321.0,
                console_uri="",
            )

        monkeypatch.setattr(jobs_module, "submit_training_job", fake_submit)

        resp = client.post(
            "/v1/finetune/managed/submit",
            json=self._submit_payload(),
            headers={"X-Tenant-ID": "tenant-a"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == "portable-1"
        assert body["tenant_id"] == "tenant-a"
        assert body["request"] == {"display_name": "tenant-a-train"}

        same_tenant = client.get(
            "/v1/finetune/managed/jobs",
            headers={"X-Tenant-ID": "tenant-a"},
        )
        other_tenant = client.get(
            "/v1/finetune/managed/jobs",
            headers={"X-Tenant-ID": "tenant-b"},
        )

        assert same_tenant.json()["count"] == 1
        assert same_tenant.json()["jobs"][0]["job_id"] == "portable-1"
        assert other_tenant.json()["count"] == 0

    def test_invalid_tenant_refuses_before_backend_submission(
        self,
        client_and_router: tuple[TestClient, APIRouter],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reject invalid ownership before submitting any training operation."""
        import director_ai.core.training.jobs as jobs_module

        invoked = False

        def refuse_submission(
            spec: TrainingJobSpec, *, backend: str, dry_run: bool
        ) -> TrainingJobSubmission:
            nonlocal invoked
            invoked = True
            raise AssertionError("Invalid tenant must not reach the backend")

        monkeypatch.setattr(jobs_module, "submit_training_job", refuse_submission)
        client, _router = client_and_router
        response = client.post(
            "/v1/finetune/managed/submit",
            json=self._submit_payload(dry_run=False),
            headers={"X-Tenant-ID": "../bad"},
        )
        assert response.status_code == 400
        assert not invoked

    def test_submit_managed_training_suite_builds_internal_spec(
        self,
        client_and_router: tuple[TestClient, APIRouter],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import director_ai.core.training.jobs as jobs_module
        from director_ai.core.training.jobs import TrainingJobSubmission

        client, _router = client_and_router

        def fake_submit(
            spec: TrainingJobSpec, *, backend: str, dry_run: bool
        ) -> TrainingJobSubmission:
            assert spec.task_type == "suite"
            assert spec.caller == "internal"
            assert spec.display_name == "director-ai-ragtruth-calibration"
            assert spec.labels["suite"] == "ragtruth-calibration"
            return TrainingJobSubmission(
                backend=backend,
                job_id="suite-1",
                state="dry_run",
                dry_run=dry_run,
                request={"task_type": spec.task_type, "labels": spec.labels},
                submitted_at=111.0,
            )

        monkeypatch.setattr(jobs_module, "submit_training_job", fake_submit)

        resp = client.post(
            "/v1/finetune/managed/submit",
            json=self._submit_payload(suite="ragtruth-calibration"),
        )

        assert resp.status_code == 200
        assert resp.json()["job_id"] == "suite-1"
        assert resp.json()["request"]["task_type"] == "suite"

    def test_submit_managed_training_maps_validation_errors(
        self,
        client_and_router: tuple[TestClient, APIRouter],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import director_ai.core.training.jobs as jobs_module

        client, _router = client_and_router
        monkeypatch.setattr(
            jobs_module,
            "submit_training_job",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ValueError("dataset_uri is required")
            ),
        )

        resp = client.post(
            "/v1/finetune/managed/submit",
            json=self._submit_payload(),
        )

        assert resp.status_code == 422
        assert resp.json() == {"detail": "Invalid managed training request"}
        assert "dataset_uri" not in resp.text

    def test_managed_status_dry_run_and_backend_conflict(
        self, client_and_router: tuple[TestClient, APIRouter]
    ) -> None:
        client, router = client_and_router
        managed_store = _closure_store(router, _ManagedJobStore)
        managed_store.add(
            ManagedTrainingRecord(
                job_id="dry-1",
                backend="portable",
                state="dry_run",
                tenant_id="tenant-a",
                dry_run=True,
                submitted_at=1.0,
                display_name="dry run",
                output_uri="gs://out",
                error="",
            )
        )

        status = client.post(
            "/v1/finetune/managed/status",
            json={"backend": "portable", "job_id": "dry-1"},
            headers={"X-Tenant-ID": "tenant-a"},
        )
        conflict = client.post(
            "/v1/finetune/managed/status",
            json={"backend": "vertex", "job_id": "dry-1"},
            headers={"X-Tenant-ID": "tenant-a"},
        )
        missing = client.post(
            "/v1/finetune/managed/status",
            json={"backend": "portable", "job_id": "dry-1"},
            headers={"X-Tenant-ID": "tenant-b"},
        )

        assert status.status_code == 200
        assert status.json()["state"] == "dry_run"
        assert status.json()["metrics"] == {}
        assert conflict.status_code == 409
        assert missing.status_code == 404

    def test_managed_status_updates_from_backend_and_maps_errors(
        self,
        client_and_router: tuple[TestClient, APIRouter],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import director_ai.core.training.jobs as jobs_module
        from director_ai.core.training.jobs import TrainingJobStatus

        client, router = client_and_router
        managed_store = _closure_store(router, _ManagedJobStore)
        managed_store.add(
            ManagedTrainingRecord(
                job_id="live-1",
                backend="portable",
                state="submitted",
                tenant_id="tenant-a",
                dry_run=False,
                submitted_at=1.0,
                display_name="live run",
                output_uri="gs://out",
            )
        )

        class Backend:
            def status(self, job_id: str) -> TrainingJobStatus:
                return TrainingJobStatus(
                    backend="portable",
                    job_id=job_id,
                    state="completed",
                    metrics={"balanced_accuracy": 0.8},
                    artifact_uri="gs://out/model",
                )

            def cancel(self, job_id: str) -> TrainingJobStatus:
                raise AssertionError("not used")

        monkeypatch.setattr(
            jobs_module, "get_training_backend", lambda _backend: Backend()
        )

        status = client.post(
            "/v1/finetune/managed/status",
            json={"backend": "portable", "job_id": "live-1"},
            headers={"X-Tenant-ID": "tenant-a"},
        )

        assert status.status_code == 200
        assert status.json()["state"] == "completed"
        assert status.json()["metrics"] == {"balanced_accuracy": 0.8}
        restored = managed_store.get("tenant-a", "live-1")
        assert restored is not None
        assert restored.state == "completed"

        monkeypatch.setattr(
            jobs_module,
            "get_training_backend",
            lambda _backend: (_ for _ in ()).throw(ValueError("bad backend")),
        )
        failed = client.post(
            "/v1/finetune/managed/status",
            json={"backend": "portable", "job_id": "live-1"},
            headers={"X-Tenant-ID": "tenant-a"},
        )
        assert failed.status_code == 422

        class BrokenStatusBackend:
            def status(self, job_id: str) -> TrainingJobStatus:
                raise RuntimeError("backend offline")

            def cancel(self, job_id: str) -> TrainingJobStatus:
                raise AssertionError("not used")

        monkeypatch.setattr(
            jobs_module,
            "get_training_backend",
            lambda _backend: BrokenStatusBackend(),
        )
        backend_error = client.post(
            "/v1/finetune/managed/status",
            json={"backend": "portable", "job_id": "live-1"},
            headers={"X-Tenant-ID": "tenant-a"},
        )
        assert backend_error.status_code == 502

    def test_managed_cancel_dry_run_and_backend_paths(
        self,
        client_and_router: tuple[TestClient, APIRouter],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import director_ai.core.training.jobs as jobs_module
        from director_ai.core.training.jobs import TrainingJobStatus

        client, router = client_and_router
        managed_store = _closure_store(router, _ManagedJobStore)
        managed_store.add(
            ManagedTrainingRecord(
                job_id="dry-1",
                backend="portable",
                state="dry_run",
                tenant_id="tenant-a",
                dry_run=True,
                submitted_at=1.0,
                display_name="dry run",
                output_uri="gs://out",
            )
        )
        managed_store.add(
            ManagedTrainingRecord(
                job_id="live-1",
                backend="portable",
                state="running",
                tenant_id="tenant-a",
                dry_run=False,
                submitted_at=2.0,
                display_name="live run",
                output_uri="gs://out/live",
            )
        )

        dry_cancel = client.post(
            "/v1/finetune/managed/cancel",
            json={"backend": "portable", "job_id": "dry-1"},
            headers={"X-Tenant-ID": "tenant-a"},
        )
        missing = client.post(
            "/v1/finetune/managed/cancel",
            json={"backend": "portable", "job_id": "live-1"},
            headers={"X-Tenant-ID": "tenant-b"},
        )
        conflict = client.post(
            "/v1/finetune/managed/cancel",
            json={"backend": "vertex", "job_id": "live-1"},
            headers={"X-Tenant-ID": "tenant-a"},
        )

        class Backend:
            def status(self, job_id: str) -> TrainingJobStatus:
                raise AssertionError("not used")

            def cancel(self, job_id: str) -> TrainingJobStatus:
                return TrainingJobStatus(
                    backend="portable",
                    job_id=job_id,
                    state="cancelled",
                    error="stopped",
                )

        monkeypatch.setattr(
            jobs_module, "get_training_backend", lambda _backend: Backend()
        )
        cancelled = client.post(
            "/v1/finetune/managed/cancel",
            json={"backend": "portable", "job_id": "live-1"},
            headers={"X-Tenant-ID": "tenant-a"},
        )

        assert dry_cancel.status_code == 409
        assert missing.status_code == 404
        assert conflict.status_code == 409
        assert cancelled.status_code == 200
        assert cancelled.json()["state"] == "cancelled"
        restored = managed_store.get("tenant-a", "live-1")
        assert restored is not None
        assert restored.state == "cancelled"

        monkeypatch.setattr(
            jobs_module,
            "get_training_backend",
            lambda _backend: (_ for _ in ()).throw(ValueError("bad backend")),
        )
        bad_backend = client.post(
            "/v1/finetune/managed/cancel",
            json={"backend": "portable", "job_id": "live-1"},
            headers={"X-Tenant-ID": "tenant-a"},
        )
        assert bad_backend.status_code == 422

        class BrokenCancelBackend:
            def status(self, job_id: str) -> TrainingJobStatus:
                raise AssertionError("not used")

            def cancel(self, job_id: str) -> TrainingJobStatus:
                raise RuntimeError("cancel offline")

        monkeypatch.setattr(
            jobs_module,
            "get_training_backend",
            lambda _backend: BrokenCancelBackend(),
        )
        backend_error = client.post(
            "/v1/finetune/managed/cancel",
            json={"backend": "portable", "job_id": "live-1"},
            headers={"X-Tenant-ID": "tenant-a"},
        )
        assert backend_error.status_code == 502

    def test_managed_model_registry_and_benchmark_endpoint(
        self,
        client_and_router: tuple[TestClient, APIRouter],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import director_ai.core.training.finetune_benchmark as benchmark_module
        import director_ai.core.training.model_registry as registry_module

        client, _router = client_and_router
        monkeypatch.setattr(
            registry_module,
            "finetune_model_registry_to_dict",
            lambda *, include_experimental: [
                {"alias": "stable", "experimental": include_experimental}
            ],
        )

        models = client.get("/v1/finetune/managed/models?include_experimental=true")

        class Report:
            def to_dict(self) -> dict[str, object]:
                return {"winner": "model-a", "score": 0.91, "results": []}

        monkeypatch.setattr(
            benchmark_module,
            "benchmark_model_candidates",
            lambda *args, **kwargs: Report(),
        )
        benchmark = client.post(
            "/v1/finetune/managed/benchmark-models",
            json={
                "model_jobs": {
                    "factcg-deberta-v3-large": _closure_store(_router, _JobStore)
                    .list_all()[0]
                    .job_id
                },
                "general_dataset_id": "b" * 32,
                "batch_size": 4,
            },
        )
        monkeypatch.setattr(
            benchmark_module,
            "benchmark_model_candidates",
            lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad artifact")),
        )
        invalid = client.post(
            "/v1/finetune/managed/benchmark-models",
            json={"model_jobs": {"bad": ""}, "general_dataset_id": "b" * 32},
        )

        assert models.status_code == 200
        assert models.json()["models"] == [{"alias": "stable", "experimental": True}]
        assert benchmark.status_code == 200
        assert benchmark.json() == {
            "winner": "model-a",
            "score": 0.91,
            "results": [],
            "general_path": "b" * 32,
            "eval_path": "",
        }
        assert invalid.status_code == 422

    def test_managed_benchmark_dependency_failure_is_fixed(
        self,
        client_and_router: tuple[TestClient, APIRouter],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unexpected benchmark dependencies produce a fixed upstream status."""
        exception = OSError("ordinary benchmark fixture failure")
        import director_ai.core.training.finetune_benchmark as benchmark_module

        def fail(*args: object, **kwargs: object) -> None:
            """Simulate an unavailable benchmark dependency at its boundary."""
            raise exception

        monkeypatch.setattr(benchmark_module, "benchmark_model_candidates", fail)
        client, _router = client_and_router
        response = client.post(
            "/v1/finetune/managed/benchmark-models",
            json={
                "model_jobs": {
                    "factcg-deberta-v3-large": _closure_store(_router, _JobStore)
                    .list_all()[0]
                    .job_id
                },
                "general_dataset_id": "b" * 32,
            },
        )
        assert response.status_code == 502
        assert response.json() == {"detail": "Model benchmark failed"}
        assert str(exception) not in response.text


class TestRouterSuccessPaths:
    """Test activate/rollback/delete on real (mocked-completed) jobs."""

    @pytest.fixture
    def client_with_job(self, tmp_path: Path) -> TestClient:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        router = create_finetune_router(models_dir=tmp_path / "models")
        app.include_router(router, prefix="/v1/finetune")
        return TestClient(app)

    def test_activate_completed_job(self, tmp_path: Path) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        router = create_finetune_router(models_dir=tmp_path / "models")
        app.include_router(router, prefix="/v1/finetune")
        client = TestClient(app)

        # Manually create a completed job via the internal store

        from director_ai.finetune_api import _JobStore

        # Get the store from the router closure
        for route in router.routes:
            if hasattr(route, "endpoint"):
                fn = route.endpoint
                if hasattr(fn, "__closure__") and fn.__closure__:
                    for cell in fn.__closure__:
                        try:
                            obj = cell.cell_contents
                            if isinstance(obj, _JobStore):
                                store = obj
                                break
                        except ValueError:
                            pass

        assert store is not None
        job = store.create({"epochs": 1})
        job.state = "completed"
        job.model_path = str(tmp_path / "models" / job.job_id)

        resp = client.post(f"/v1/finetune/{job.job_id}/activate")
        assert resp.status_code == 200
        body = resp.json()
        assert body["activated"] is True
        assert body["model_path"] == job.model_path
        # The endpoint is honest about not hot-swapping the live scorer and
        # points to the real way to serve the model.
        assert "nli_model" in body["detail"]
        assert "restart" in body["detail"]

    def test_rollback_activated_job(self, tmp_path: Path) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from director_ai.finetune_api import _JobStore

        app = FastAPI()
        router = create_finetune_router(models_dir=tmp_path / "models")
        app.include_router(router, prefix="/v1/finetune")
        client = TestClient(app)

        store = None
        for route in router.routes:
            if (
                hasattr(route, "endpoint")
                and hasattr(route.endpoint, "__closure__")
                and route.endpoint.__closure__
            ):
                for cell in route.endpoint.__closure__:
                    try:
                        obj = cell.cell_contents
                        if isinstance(obj, _JobStore):
                            store = obj
                            break
                    except ValueError:
                        pass
                if store:
                    break

        assert store is not None
        job = store.create({"epochs": 1})
        job.state = "completed"
        job.activated = True

        resp = client.post(f"/v1/finetune/{job.job_id}/rollback")
        assert resp.status_code == 200
        assert resp.json()["activated"] is False

    def test_delete_completed_job(self, tmp_path: Path) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from director_ai.finetune_api import _JobStore

        app = FastAPI()
        router = create_finetune_router(models_dir=tmp_path / "models")
        app.include_router(router, prefix="/v1/finetune")
        client = TestClient(app)

        store = None
        for route in router.routes:
            if (
                hasattr(route, "endpoint")
                and hasattr(route.endpoint, "__closure__")
                and route.endpoint.__closure__
            ):
                for cell in route.endpoint.__closure__:
                    try:
                        obj = cell.cell_contents
                        if isinstance(obj, _JobStore):
                            store = obj
                            break
                    except ValueError:
                        pass
                if store:
                    break

        assert store is not None
        job = store.create({"epochs": 1})
        job.state = "completed"

        resp = client.delete(f"/v1/finetune/{job.job_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

        resp2 = client.get(f"/v1/finetune/{job.job_id}")
        assert resp2.status_code == 404

    def test_delete_activated_blocked(self, tmp_path: Path) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from director_ai.finetune_api import _JobStore

        app = FastAPI()
        router = create_finetune_router(models_dir=tmp_path / "models")
        app.include_router(router, prefix="/v1/finetune")
        client = TestClient(app)

        store = None
        for route in router.routes:
            if (
                hasattr(route, "endpoint")
                and hasattr(route.endpoint, "__closure__")
                and route.endpoint.__closure__
            ):
                for cell in route.endpoint.__closure__:
                    try:
                        obj = cell.cell_contents
                        if isinstance(obj, _JobStore):
                            store = obj
                            break
                    except ValueError:
                        pass
                if store:
                    break

        assert store is not None
        job = store.create({"epochs": 1})
        job.state = "completed"
        job.activated = True

        resp = client.delete(f"/v1/finetune/{job.job_id}")
        assert resp.status_code == 409

    def test_activate_training_returns_409(self, tmp_path: Path) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from director_ai.finetune_api import _JobStore

        app = FastAPI()
        router = create_finetune_router(models_dir=tmp_path / "models")
        app.include_router(router, prefix="/v1/finetune")
        client = TestClient(app)

        store = None
        for route in router.routes:
            if (
                hasattr(route, "endpoint")
                and hasattr(route.endpoint, "__closure__")
                and route.endpoint.__closure__
            ):
                for cell in route.endpoint.__closure__:
                    try:
                        obj = cell.cell_contents
                        if isinstance(obj, _JobStore):
                            store = obj
                            break
                    except ValueError:
                        pass
                if store:
                    break

        assert store is not None
        job = store.create({"epochs": 1})
        job.state = "training"

        resp = client.post(f"/v1/finetune/{job.job_id}/activate")
        assert resp.status_code == 409

    def test_delete_cleans_model_directory(self, tmp_path: Path) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from director_ai.finetune_api import _JobStore

        models_dir = tmp_path / "models"
        app = FastAPI()
        router = create_finetune_router(models_dir=models_dir)
        app.include_router(router, prefix="/v1/finetune")
        client = TestClient(app)

        store = None
        for route in router.routes:
            if (
                hasattr(route, "endpoint")
                and hasattr(route.endpoint, "__closure__")
                and route.endpoint.__closure__
            ):
                for cell in route.endpoint.__closure__:
                    try:
                        obj = cell.cell_contents
                        if isinstance(obj, _JobStore):
                            store = obj
                            break
                    except ValueError:
                        pass
                if store:
                    break

        model_dir = models_dir / "test-model-dir"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")

        assert store is not None
        job = store.create({"epochs": 1})
        job.state = "completed"
        job.model_path = str(model_dir)

        resp = client.delete(f"/v1/finetune/{job.job_id}")
        assert resp.status_code == 200
        assert not model_dir.exists()

    def test_delete_does_not_remove_model_path_outside_models_dir(
        self, tmp_path: Path
    ) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        router = create_finetune_router(models_dir=tmp_path / "models")
        app.include_router(router, prefix="/v1/finetune")
        client = TestClient(app)
        store = _closure_store(router, _JobStore)

        outside_dir = tmp_path / "outside-model"
        outside_dir.mkdir()
        (outside_dir / "config.json").write_text("{}", encoding="utf-8")
        assert store is not None
        job = store.create({"epochs": 1})
        job.state = "completed"
        job.model_path = str(outside_dir)

        resp = client.delete(f"/v1/finetune/{job.job_id}")

        assert resp.status_code == 200
        assert outside_dir.exists()
        assert store.get(job.job_id) is None

    def test_result_completed_job(self, tmp_path: Path) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from director_ai.finetune_api import _JobStore

        app = FastAPI()
        router = create_finetune_router(models_dir=tmp_path / "models")
        app.include_router(router, prefix="/v1/finetune")
        client = TestClient(app)

        store = None
        for route in router.routes:
            if (
                hasattr(route, "endpoint")
                and hasattr(route.endpoint, "__closure__")
                and route.endpoint.__closure__
            ):
                for cell in route.endpoint.__closure__:
                    try:
                        obj = cell.cell_contents
                        if isinstance(obj, _JobStore):
                            store = obj
                            break
                    except ValueError:
                        pass
                if store:
                    break

        assert store is not None
        job = store.create({"epochs": 1})
        job.state = "completed"
        job.metrics = {"eval_balanced_accuracy": 0.85}
        job.regression_report = {"recommendation": "deploy"}

        resp = client.get(f"/v1/finetune/{job.job_id}/result")
        assert resp.status_code == 200
        body = resp.json()
        assert body["state"] == "completed"
        assert body["metrics"]["eval_balanced_accuracy"] == 0.85
        assert body["regression_report"]["recommendation"] == "deploy"


class TestRouterIsolation:
    def test_separate_routers_have_separate_stores(self, tmp_path: Path) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        r1 = create_finetune_router(models_dir=tmp_path / "m1")
        r2 = create_finetune_router(models_dir=tmp_path / "m2")
        app.include_router(r1, prefix="/v1/ft1")
        app.include_router(r2, prefix="/v1/ft2")
        client = TestClient(app)

        resp1 = client.get("/v1/ft1/")
        resp2 = client.get("/v1/ft2/")
        assert resp1.json()["models"] == []
        assert resp2.json()["models"] == []


class TestRouterStartEndpoint:
    """Test the /start endpoint with mocked training worker."""

    @pytest.fixture
    def client(self, tmp_path: Path) -> TestClient:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        router = create_finetune_router(
            models_dir=tmp_path / "models",
            operator_api_keys=("operator-test-key",),
        )
        app.include_router(router, prefix="/v1/finetune")
        return TestClient(app, headers={"X-API-Key": "operator-test-key"})

    def _make_jsonl_bytes(self, n_pos: int = 300, n_neg: int = 300) -> bytes:
        rows = []
        for i in range(n_pos):
            rows.append(
                {"premise": f"Fact {i}.", "hypothesis": f"Claim {i}.", "label": 1},
            )
        for i in range(n_neg):
            rows.append(
                {"premise": f"Source {i}.", "hypothesis": f"Wrong {i}.", "label": 0},
            )
        return ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8")

    @patch("director_ai.finetune_api._run_training_worker")
    def test_start_valid_data(self, mock_worker: MagicMock, client: TestClient) -> None:
        data = self._make_jsonl_bytes()
        resp = client.post(
            "/v1/finetune/start",
            files={"file": ("train.jsonl", data, "application/jsonl")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "job_id" in body
        assert body["total_samples"] == 600
        assert body["estimated_time_min"] > 0
        assert mock_worker.called

    def test_validate_rejects_oversized_upload(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(finetune_api_module, "_MAX_UPLOAD_BYTES", 8)

        resp = client.post(
            "/v1/finetune/validate",
            files={"file": ("too-large.jsonl", b"x" * 16, "application/jsonl")},
        )

        assert resp.status_code == 413

    @patch("director_ai.finetune_api._run_training_worker")
    def test_start_rejects_unknown_base_model(
        self,
        mock_worker: MagicMock,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import director_ai.core.training.model_registry as registry_module

        monkeypatch.setattr(
            registry_module,
            "resolve_finetune_model",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ValueError("unknown scorer_model")
            ),
        )

        resp = client.post(
            "/v1/finetune/start",
            data={"base_model": "unknown-model"},
            files={
                "file": ("train.jsonl", self._make_jsonl_bytes(), "application/jsonl")
            },
        )

        assert resp.status_code == 422
        assert resp.json() == {"detail": "Invalid fine-tune model request"}
        assert "unknown scorer_model" not in resp.text
        mock_worker.assert_not_called()

    @pytest.mark.parametrize(
        ("boundary", "exception", "status", "detail"),
        [
            (
                "model",
                OSError("ordinary registry fixture failure"),
                500,
                "Fine-tune model resolution failed",
            ),
            (
                "store",
                ValueError("ordinary job fixture failure"),
                422,
                "Invalid fine-tune job request",
            ),
            (
                "store",
                OSError("ordinary store dependency fixture failure"),
                500,
                "Fine-tune job creation failed",
            ),
        ],
    )
    def test_start_dependency_failure_cleans_upload(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        boundary: str,
        exception: Exception,
        status: int,
        detail: str,
    ) -> None:
        """Dependency failures keep fixed statuses and remove uploaded data."""
        import director_ai.core.training.model_registry as registry_module

        def fail(*args: object, **kwargs: object) -> None:
            """Simulate a failure at the selected local training dependency."""
            raise exception

        if boundary == "model":
            monkeypatch.setattr(registry_module, "resolve_finetune_model", fail)
        else:
            monkeypatch.setattr(_JobStore, "create", fail)
        response = client.post(
            "/v1/finetune/start",
            files={
                "file": ("train.jsonl", self._make_jsonl_bytes(), "application/jsonl")
            },
        )
        assert response.status_code == status
        assert response.json() == {"detail": detail}
        assert str(exception) not in response.text
        assert list((tmp_path / "models" / "_uploads").iterdir()) == []

    @patch("director_ai.finetune_api._run_training_worker")
    def test_start_returns_409_on_result_while_training(
        self, mock_worker: MagicMock, client: TestClient
    ) -> None:
        data = self._make_jsonl_bytes()
        resp = client.post(
            "/v1/finetune/start",
            files={"file": ("train.jsonl", data, "application/jsonl")},
        )
        job_id = resp.json()["job_id"]
        result_resp = client.get(f"/v1/finetune/{job_id}/result")
        assert result_resp.status_code == 409

    @patch("director_ai.finetune_api._run_training_worker")
    def test_start_get_status(self, mock_worker: MagicMock, client: TestClient) -> None:
        data = self._make_jsonl_bytes()
        resp = client.post(
            "/v1/finetune/start",
            files={"file": ("train.jsonl", data, "application/jsonl")},
        )
        job_id = resp.json()["job_id"]
        status = client.get(f"/v1/finetune/{job_id}")
        assert status.status_code == 200
        assert status.json()["job_id"] == job_id

    @patch("director_ai.finetune_api._run_training_worker")
    def test_start_shows_in_list(
        self, mock_worker: MagicMock, client: TestClient
    ) -> None:
        data = self._make_jsonl_bytes()
        client.post(
            "/v1/finetune/start",
            files={"file": ("train.jsonl", data, "application/jsonl")},
        )
        listing = client.get("/v1/finetune/")
        assert len(listing.json()["models"]) == 1

    @patch("director_ai.finetune_api._run_training_worker")
    def test_start_concurrent_limit_429(
        self, mock_worker: MagicMock, tmp_path: Path
    ) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        router = create_finetune_router(models_dir=tmp_path / "models")
        app.include_router(router, prefix="/v1/finetune")
        client = TestClient(app)

        # Fill up the concurrent limit
        store = None
        for route in router.routes:
            if (
                hasattr(route, "endpoint")
                and hasattr(route.endpoint, "__closure__")
                and route.endpoint.__closure__
            ):
                for cell in route.endpoint.__closure__:
                    try:
                        obj = cell.cell_contents
                        if isinstance(obj, _JobStore):
                            store = obj
                            break
                    except ValueError:
                        pass
                if store:
                    break

        for _i in range(_MAX_CONCURRENT_JOBS):
            assert store is not None
            job = store.create({"epochs": 1})
            job.state = "training"
            store.save(job)  # write-through, as the training worker does

        data = self._make_jsonl_bytes()
        resp = client.post(
            "/v1/finetune/start",
            files={"file": ("train.jsonl", data, "application/jsonl")},
        )
        assert resp.status_code == 429


class TestBenchmarkJsonlRobust:
    def test_malformed_json_skipped(self, tmp_path: Path) -> None:
        from director_ai.core.training.finetune_benchmark import _load_benchmark_jsonl

        f = tmp_path / "bench.jsonl"
        f.write_text(
            '{"premise":"a","hypothesis":"b","label":1}\n'
            "not json at all\n"
            '{"premise":"c","hypothesis":"d","label":0}\n',
            encoding="utf-8",
        )
        rows = _load_benchmark_jsonl(f)
        assert len(rows) == 2


def _make_jsonl_file(path: Path, n_pos: int = 60, n_neg: int = 60) -> None:
    rows = []
    for i in range(n_pos):
        rows.append({"premise": f"Fact {i}.", "hypothesis": f"Claim {i}.", "label": 1})
    for i in range(n_neg):
        rows.append(
            {"premise": f"Source {i}.", "hypothesis": f"Wrong {i}.", "label": 0},
        )
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


class TestTrainingWorkerDirect:
    """Test _run_training_worker synchronously with mocked finetune_nli."""

    @patch("director_ai.core.finetune.finetune_nli")
    def test_worker_completes(self, mock_ft: MagicMock, tmp_path: Path) -> None:
        from director_ai.core.training.finetune import FinetuneResult

        mock_ft.return_value = FinetuneResult(
            output_dir=str(tmp_path / "models" / "test-ok"),
            epochs_completed=1,
            train_samples=108,
            eval_samples=12,
            best_balanced_accuracy=0.85,
            final_loss=0.3,
            eval_metrics={"eval_balanced_accuracy": 0.85},
        )

        data_path = tmp_path / "upload.jsonl"
        _make_jsonl_file(data_path)

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        job = FinetuneJob(
            job_id="test-ok",
            config={
                "epochs": 1,
                "batch_size": 4,
                "auto_benchmark": False,
                "auto_onnx_export": False,
            },
        )

        _run_training_worker(job, data_path, models_dir, _JobStore())

        assert job.state == "completed"
        assert job.progress == 1.0
        assert job.model_path == str(tmp_path / "models" / "test-ok")
        assert job.completed_at > 0
        assert not data_path.exists()

    @patch(
        "director_ai.core.finetune.finetune_nli",
        side_effect=ValueError("No valid samples"),
    )
    def test_worker_handles_training_error(
        self, mock_ft: MagicMock, tmp_path: Path
    ) -> None:
        data_path = tmp_path / "data.jsonl"
        _make_jsonl_file(data_path, 10, 10)

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        job = FinetuneJob(
            job_id="test-fail",
            config={"epochs": 1, "batch_size": 4},
        )

        _run_training_worker(job, data_path, models_dir, _JobStore())

        assert job.state == "failed"
        assert "No valid samples" in job.error
        assert not data_path.exists()

    @patch("director_ai.core.finetune.finetune_nli")
    def test_worker_splits_data_90_10(self, mock_ft: MagicMock, tmp_path: Path) -> None:
        from director_ai.core.training.finetune import FinetuneResult

        mock_ft.return_value = FinetuneResult(output_dir=str(tmp_path / "m" / "j1"))

        data_path = tmp_path / "data.jsonl"
        _make_jsonl_file(data_path, 50, 50)

        models_dir = tmp_path / "m"
        models_dir.mkdir()

        job = FinetuneJob(job_id="j1", config={"epochs": 1, "batch_size": 4})
        _run_training_worker(job, data_path, models_dir, _JobStore())

        assert job.state == "completed"
        call_args = mock_ft.call_args
        train_path = call_args[0][0]
        eval_path = call_args[1]["eval_path"]
        assert train_path.endswith("_train.jsonl")
        assert eval_path.endswith("_eval.jsonl")

    @patch(
        "director_ai.core.finetune.finetune_nli",
        side_effect=RuntimeError("GPU OOM"),
    )
    def test_worker_cleans_up_on_exception(
        self, mock_ft: MagicMock, tmp_path: Path
    ) -> None:
        data_path = tmp_path / "data.jsonl"
        _make_jsonl_file(data_path)

        models_dir = tmp_path / "m"
        models_dir.mkdir()

        job = FinetuneJob(job_id="j2", config={"epochs": 1, "batch_size": 4})
        _run_training_worker(job, data_path, models_dir, _JobStore())

        assert job.state == "failed"
        assert "GPU OOM" in job.error
        assert not data_path.exists()
